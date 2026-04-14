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
// 【方案 A】极致向量化双线性插值 (一次计算 2 个 Channel)
// ============================================================================
__device__ float2 ms_deform_attn_im2col_bilinear_half2(const half2* bottom_data, 
                                                   const int height, const int width, const int nheads, const int channels_half2,
                                                   const float h, const float w, const int m, const int c_half2)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1.0f - lh, hw = 1.0f - lw;

  // 注意：这里的 stride 都是以 half2 为单位计算的
  const int w_stride = nheads * channels_half2;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels_half2 + c_half2;

  // 生成一个包含两个 0.0 的 half2
  half2 zero_h2 = __float2half2_rn(0.0f);

  // 极其高效的 32-bit (half2) 一次性拉取
  half2 v1_h2 = (h_low >= 0 && w_low >= 0) ? bottom_data[h_low_ptr_offset + w_low_ptr_offset + base_ptr] : zero_h2;
  half2 v2_h2 = (h_low >= 0 && w_high <= width - 1) ? bottom_data[h_low_ptr_offset + w_high_ptr_offset + base_ptr] : zero_h2;
  half2 v3_h2 = (h_high <= height - 1 && w_low >= 0) ? bottom_data[h_high_ptr_offset + w_low_ptr_offset + base_ptr] : zero_h2;
  half2 v4_h2 = (h_high <= height - 1 && w_high <= width - 1) ? bottom_data[h_high_ptr_offset + w_high_ptr_offset + base_ptr] : zero_h2;

  // 内部无损转换为 float2 进行高精度累加
  float2 v1 = __half22float2(v1_h2);
  float2 v2 = __half22float2(v2_h2);
  float2 v3 = __half22float2(v3_h2);
  float2 v4 = __half22float2(v4_h2);

  float2 res;
  res.x = hh * hw * v1.x + hh * lw * v2.x + lh * hw * v3.x + lh * lw * v4.x;
  res.y = hh * hw * v1.y + hh * lw * v2.y + lh * hw * v3.y + lh * lw * v4.y;
  return res;
}

// ============================================================================
// 【方案 A】原生 FP16 + Half2 向量化计算内核
// ============================================================================
__global__ void ms_deformable_im2col_gpu_kernel_half2(const int n,
                                                const half2 *data_value, 
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index, 
                                                const __half *data_sampling_loc,
                                                const __half *data_attn_weight,
                                                const int batch_size, 
                                                const int spatial_size, 
                                                const int num_heads,
                                                const int channels, 
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                half2 *data_col)
{
  // n 是 (batch * num_query * num_heads * (channels / 2))
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int channels_half2 = channels / 2; // 通道数砍半
    const int c_col_half2 = _temp % channels_half2; _temp /= channels_half2;
    const int sampling_index = _temp; 
    const int m_col = _temp % num_heads; _temp /= num_heads;
    const int q_col = _temp % num_query; _temp /= num_query;
    const int b_col = _temp;

    half2 *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    
    const int qid_stride_half2 = num_heads * channels_half2;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride_half2;
    
    // 用 float2 当作累加器，杜绝 FP16 加法溢出
    float2 col;
    col.x = 0.0f; col.y = 0.0f;
    
    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const half2 *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride_half2);
      
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        // 采样坐标和权重依然是单个的 __half，这里强转为 float
        const float loc_w = __half2float(data_sampling_loc[data_loc_w_ptr]);
        const float loc_h = __half2float(data_sampling_loc[data_loc_w_ptr + 1]);
        const float weight = __half2float(data_attn_weight[data_weight_ptr]);

        const float h_im = loc_h * spatial_h - 0.5f;
        const float w_im = loc_w * spatial_w - 0.5f;

        if (h_im > -1.0f && w_im > -1.0f && h_im < spatial_h && w_im < spatial_w)
        {
          float2 val = ms_deform_attn_im2col_bilinear_half2(data_value_ptr, spatial_h, spatial_w, num_heads, channels_half2, h_im, w_im, m_col, c_col_half2);
          // 一次乘加，同时更新 2 个 channel！
          col.x += val.x * weight;
          col.y += val.y * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    // 最后把高精度的 float2 打包回 half2 塞入显存
    *data_col_ptr = __float22half2_rn(col);
  }
}

// ============================================================================
// 启动 Launcher
// ============================================================================
void ms_deformable_im2col_cuda_half2(hipStream_t stream,
                              const __half* data_value,
                              const int64_t* data_spatial_shapes, 
                              const int64_t* data_level_start_index, 
                              const __half* data_sampling_loc,
                              const __half* data_attn_weight,
                              const int batch_size,
                              const int spatial_size, 
                              const int num_heads, 
                              const int channels, 
                              const int num_levels, 
                              const int num_query,
                              const int num_point,
                              __half* data_col)
{
  // 【绝杀】总线程数直接砍半，极大缓解 DCU 的线程调度压力
  const int num_kernels = batch_size * num_query * num_heads * (channels / 2);
  const int num_threads = CUDA_NUM_THREADS;
  
  // 巧妙地将 __half 指针强制转换为 half2* 传入，实现合并读写
  ms_deformable_im2col_gpu_kernel_half2<<<GET_BLOCKS(num_kernels, num_threads), num_threads, 0, stream>>>(
      num_kernels, (const half2*)data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight, 
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, (half2*)data_col);
}