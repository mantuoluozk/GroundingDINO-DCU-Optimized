#include "hip/hip_runtime.h"
#include <hip/hip_fp16.h>
#include <cstdio>
#include <algorithm>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

// ============================================================================
// 【方案 B】极速就地转换核函数 (FP16 <-> FP32)
// ============================================================================
template <typename T_IN, typename T_OUT>
__global__ void FastCastKernel(const T_IN* input, T_OUT* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = static_cast<T_OUT>(input[i]);
    }
}

template <typename T_IN, typename T_OUT>
void RunFastCast(hipStream_t stream, const T_IN* input, T_OUT* output, int n) {
    if (n <= 0) return;
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    // 【关键修复 1】使用 <<< >>> 语法，完美避开 hipLaunchKernelGGL 宏解析带逗号模板的 Bug
    FastCastKernel<T_IN, T_OUT><<<blocks, threads, 0, stream>>>(input, output, n);
}

// ============================================================================
// FP32 双线性插值计算 (Device)
// ============================================================================
__device__ float ms_deform_attn_im2col_bilinear_fp32(const float* &bottom_data, 
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const float &h, const float &w, const int &m, const int &c)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1.0f - lh, hw = 1.0f - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    v1 = bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    v2 = bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    v3 = bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    v4 = bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr];
  }

  return (hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4);
}

// ============================================================================
// FP32 核心计算内核 (Global)
// ============================================================================
__global__ void ms_deformable_im2col_gpu_kernel_fp32(const int n,
                                                const float *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const float *data_sampling_loc,
                                                const float *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                float *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels; _temp /= channels;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads; _temp /= num_heads;
    const int q_col = _temp % num_query; _temp /= num_query;
    const int b_col = _temp;

    float *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    float col = 0.0f;
    
    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const float *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const float loc_w = data_sampling_loc[data_loc_w_ptr];
        const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const float weight = data_attn_weight[data_weight_ptr];

        const float h_im = loc_h * spatial_h - 0.5f;
        const float w_im = loc_w * spatial_w - 0.5f;

        if (h_im > -1.0f && w_im > -1.0f && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear_fp32(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

// ============================================================================
// 算子启动函数 (Launcher)
// ============================================================================
void ms_deformable_im2col_cuda_fp32(hipStream_t stream,
                              const float* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const float* data_sampling_loc,
                              const float* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              float* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  
  // 【关键修复 2】同样改用 <<< >>> 语法，避免宏展开的坑
  ms_deformable_im2col_gpu_kernel_fp32<<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);
}