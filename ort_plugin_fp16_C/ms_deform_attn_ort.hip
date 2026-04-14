#define ORT_API_MANUAL_INIT

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cmath>
#include <algorithm>

// 引入核心计算头文件
#include "ms_deform_im2col_cuda_hip.cuh"

// ============================================================================
// 1. 定义 Kernel (极致干净的 Native FP16 接口)
// ============================================================================
struct MsDeformAttnKernel {
    const OrtApi* api_;
    int im2col_step_;

    MsDeformAttnKernel(const OrtApi* api, const OrtKernelInfo* info) : api_(api) {
        int64_t step = 64;
        OrtStatus* status = api_->KernelInfoGetAttribute_int64(info, "im2col_step_i", &step);
        if (status != nullptr) {
            api_->ReleaseStatus(status);
            im2col_step_ = 64; 
        } else {
            im2col_step_ = static_cast<int>(step);
        }
        im2col_step_ = std::max(1, im2col_step_);
    }

    void CheckStatus(OrtStatus* status) {
        if (status != nullptr) api_->ReleaseStatus(status);
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
        if (status != nullptr) api_->ReleaseStatus(status);
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* context) {
        const OrtValue *v_t, *ss_t, *lsi_t, *sl_t, *aw_t;
        CheckStatus(api_->KernelContext_GetInput(context, 0, &v_t));
        CheckStatus(api_->KernelContext_GetInput(context, 1, &ss_t));
        CheckStatus(api_->KernelContext_GetInput(context, 2, &lsi_t));
        CheckStatus(api_->KernelContext_GetInput(context, 3, &sl_t));
        CheckStatus(api_->KernelContext_GetInput(context, 4, &aw_t));

        void *v_p, *ss_p, *lsi_p, *sl_p, *aw_p;
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(v_t), &v_p));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(ss_t), &ss_p));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(lsi_t), &lsi_p));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(sl_t), &sl_p));
        CheckStatus(api_->GetTensorMutableData(const_cast<OrtValue*>(aw_t), &aw_p));

        auto v_dims = GetTensorDims(v_t);
        auto sl_dims = GetTensorDims(sl_t);
        auto ss_dims = GetTensorDims(ss_t);

        const int batch = v_dims[0];
        const int spatial_size = v_dims[1];
        const int num_heads = v_dims[2];
        const int channels = v_dims[3];
        const int num_levels = ss_dims[0];
        const int num_query = sl_dims[1];
        const int num_point = sl_dims[4];

        std::vector<int64_t> output_dims = {batch, num_query, num_heads * channels};
        OrtValue* output_tensor;
        CheckStatus(api_->KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size(), &output_tensor));
        
        void* output_ptr;
        CheckStatus(api_->GetTensorMutableData(output_tensor, &output_ptr));

        void* stream_ptr;
        CheckStatus(api_->KernelContext_GetGPUComputeStream(context, &stream_ptr));
        hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

        // 强转为 __half 原生指针
        const __half* v_f16 = reinterpret_cast<const __half*>(v_p);
        const __half* sl_f16 = reinterpret_cast<const __half*>(sl_p);
        const __half* aw_f16 = reinterpret_cast<const __half*>(aw_p);
        __half* out_f16 = reinterpret_cast<__half*>(output_ptr);
        
        const int64_t* ss_i64 = reinterpret_cast<const int64_t*>(ss_p);
        const int64_t* lsi_i64 = reinterpret_cast<const int64_t*>(lsi_p);

        const int im2col_step_real = std::max(1, std::min(batch, im2col_step_));
        const int batch_n = im2col_step_real;
        
        auto per_value_size = spatial_size * num_heads * channels;
        auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
        auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
        auto per_output_size = batch_n * num_query * num_heads * channels;

        // 核心突破：摒弃所有 Cast，直接传给 half2 极速内核！
        for (int n = 0; n < batch / im2col_step_real; ++n) {
            ms_deformable_im2col_cuda_half2(
                stream,
                v_f16 + n * im2col_step_real * per_value_size,
                ss_i64,
                lsi_i64,
                sl_f16 + n * im2col_step_real * per_sample_loc_size,
                aw_f16 + n * im2col_step_real * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                out_f16 + n * per_output_size
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
        if (index == 0 || index == 3 || index == 4) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
};

// ============================================================================
// 3. 注册入口
// ============================================================================
MsDeformAttnCustomOp ms_deform_attn_op;

extern "C" OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    Ort::InitApi(api->GetApi(ORT_API_VERSION));
    OrtCustomOpDomain* domain = nullptr;
    const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

    if (auto status = ortApi->CreateCustomOpDomain("custom", &domain)) { return status; }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &ms_deform_attn_op)) { return status; }
    return ortApi->AddCustomOpDomain(options, domain);
}