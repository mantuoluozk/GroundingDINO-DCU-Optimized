python -m onnxruntime.perf_test \
-m times \
-r 10 \
-s \
-e rocm \
-p profile.json \
-t random \
ResNet50.onnx \
perf_result_profile.txt