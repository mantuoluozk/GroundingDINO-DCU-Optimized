// 【关键修复 1】定义这个宏，关闭 ORT 自动寻找 OrtGetApiBase 的行为
#define ORT_API_MANUAL_INIT

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

// 引入核心计算头文件
#include "ms_deform_im2col_cuda_hip.cuh"

// ============================================================================
// 1. 定义 Kernel
// ============================================================================
struct MsDeformAttnKernel {
    const OrtApi* api_;
    int im2col_step_;

    MsDeformAttnKernel(const OrtApi* api, const OrtKernelInfo* info) : api_(api) {
        int64_t step = 64;
        OrtStatus* status = api_->KernelInfoGetAttribute_int64(info, "im2col_step_i", &step);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            im2col_step_ = 64; // 【修复1】加上兜底赋值
        } else {
            im2col_step_ = static_cast<int>(step);
        }
        im2col_step_ = std::max(1, im2col_step_);
    }

    void CheckStatus(OrtStatus* status) {
        if (status != nullptr) {
            api_->ReleaseStatus(status);
        }
    }

    std::vector<int64_t> GetTensorDims(const OrtValue* tensor) {
        OrtTensorTypeAndShapeInfo* info;
        CheckStatus(api_->GetTensorTypeAndShape(tensor, &info));
        
        size_t dim_count;
        CheckStatus(api_->GetDimensionsCount(info, &dim_count));
        
        std::vector<int64_t> dims(dim_count);
        CheckStatus(api_->GetDimensions(info, dims.data(), dim_count));
        
        api_->ReleaseTensorTypeAndShapeInfo(info);
        return dims;
    }

    void Compute(OrtKernelContext* context) {
        OrtStatusPtr status = ComputeV2(context);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
        }
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* context) {
        const OrtValue* value_tensor;
        const OrtValue* spatial_shapes_tensor;
        const OrtValue* level_start_index_tensor;
        const OrtValue* sampling_loc_tensor;
        const OrtValue* attn_weight_tensor;

        CheckStatus(api_->KernelContext_GetInput(context, 0, &value_tensor));
        CheckStatus(api_->KernelContext_GetInput(context, 1, &spatial_shapes_tensor));
        CheckStatus(api_->KernelContext_GetInput(context, 2, &level_start_index_tensor));
        CheckStatus(api_->KernelContext_GetInput(context, 3, &sampling_loc_tensor));
        CheckStatus(api_->KernelContext_GetInput(context, 4, &attn_weight_tensor));

        void* value_ptr;
        void* spatial_shapes_ptr;
        void* level_start_index_ptr;
        void* sampling_loc_ptr;
        void* attn_weight_ptr;

        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(value_tensor), &value_ptr));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(spatial_shapes_tensor), &spatial_shapes_ptr));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(level_start_index_tensor), &level_start_index_ptr));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(sampling_loc_tensor), &sampling_loc_ptr));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(attn_weight_tensor), &attn_weight_ptr));

        const float* value_data = reinterpret_cast<const float*>(value_ptr);
        const int64_t* spatial_shapes_data = reinterpret_cast<const int64_t*>(spatial_shapes_ptr);
        const int64_t* level_start_index_data = reinterpret_cast<const int64_t*>(level_start_index_ptr);
        const float* sampling_loc_data = reinterpret_cast<const float*>(sampling_loc_ptr);
        const float* attn_weight_data = reinterpret_cast<const float*>(attn_weight_ptr);

        auto value_dims = GetTensorDims(value_tensor);
        auto spatial_shapes_dims = GetTensorDims(spatial_shapes_tensor);
        auto sampling_loc_dims = GetTensorDims(sampling_loc_tensor);

        const int batch = value_dims[0];
        const int spatial_size = value_dims[1];
        const int num_heads = value_dims[2];
        const int channels = value_dims[3];
        const int num_levels = spatial_shapes_dims[0];
        const int num_query = sampling_loc_dims[1];
        const int num_point = sampling_loc_dims[4];

        std::vector<int64_t> output_dims = {batch, num_query, num_heads * channels};
        OrtValue* output_tensor;
        CheckStatus(api_->KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size(), &output_tensor));
        
        void* output_ptr;
        CheckStatus(api_->GetTensorMutableData(output_tensor, &output_ptr));
        float* output_data = reinterpret_cast<float*>(output_ptr);

        void* stream_ptr;
        CheckStatus(api_->KernelContext_GetGPUComputeStream(context, &stream_ptr));
        hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

        // const int im2col_step_real = std::min(batch, im2col_step_);
        const int im2col_step_real = std::max(1, std::min(batch, im2col_step_));
        const int batch_n = im2col_step_real;
        
        auto per_value_size = spatial_size * num_heads * channels;
        auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
        auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
        auto per_output_size = batch_n * num_query * num_heads * channels;

        for (int n = 0; n < batch / im2col_step_real; ++n) {
            ms_deformable_im2col_cuda(
                stream,
                value_data + n * im2col_step_real * per_value_size,
                spatial_shapes_data,
                level_start_index_data,
                sampling_loc_data + n * im2col_step_real * per_sample_loc_size,
                attn_weight_data + n * im2col_step_real * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                output_data + n * per_output_size
            );
        }

        return nullptr;
    }
};

// ============================================================================
// 2. 算子接口封装
// ============================================================================
struct MsDeformAttnCustomOp : Ort::CustomOpBase<MsDeformAttnCustomOp, MsDeformAttnKernel> {
    
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new MsDeformAttnKernel(&api, info);
    }

    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** op_kernel) const {
        *op_kernel = new MsDeformAttnKernel(&api, info);
        return nullptr;
    }

    const char* GetName() const { return "ms_deform_attn"; }
    const char* GetExecutionProviderType() const { return "ROCMExecutionProvider"; }

    size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0 || index == 3 || index == 4) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

// ============================================================================
// 3. 注册入口
// ============================================================================
MsDeformAttnCustomOp ms_deform_attn_op;

extern "C" OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    // 【关键修复 2】手动初始化 C++ API，把 Python 传来的真实 API 指针交给 ORT
    Ort::InitApi(api->GetApi(ORT_API_VERSION));

    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

    if (auto status = ortApi->CreateCustomOpDomain("custom", &domain)) { return status; }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &ms_deform_attn_op)) { return status; }
    return ortApi->AddCustomOpDomain(options, domain);
}