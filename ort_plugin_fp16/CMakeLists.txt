cmake_minimum_required(VERSION 3.14)
project(GroundingDinoORTPlugin)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 告诉 CMake 使用 hipcc 编译器
set(CMAKE_CXX_COMPILER "hipcc")
set(CMAKE_C_COMPILER "hipcc")

# 1. 寻找 ONNX Runtime 头文件
# 请替换为你环境中实际的 onnxruntime_cxx_api.h 所在路径
set(ONNXRUNTIME_INCLUDE_DIR "/opt/dtk-25.04.2/onnxruntime/include")
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(.) # 把当前目录加进去，方便找到 .cuh

# 2. 编译我们的共享库 (.so)
add_library(ms_deform_attn_ort SHARED 
    ms_deform_attn_ort.hip
)

# =========================================================================
# 【关键修复】告诉 CMake 把 .hip 文件当作 C++ 语言来编译和链接
# =========================================================================
set_source_files_properties(ms_deform_attn_ort.hip PROPERTIES LANGUAGE CXX)
set_target_properties(ms_deform_attn_ort PROPERTIES LINKER_LANGUAGE CXX)

# 编译优化
# target_compile_options(ms_deform_attn_ort PRIVATE -O3 -fPIC)
target_compile_options(ms_deform_attn_ort PRIVATE -O3 -fPIC --gpu-max-threads-per-block=1024)