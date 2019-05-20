# Edge TPU Compiler API

A simple python api for Google Edge TPU online compiler.

## Usage

For cli:
```shell
python edge_tpu_comiler_api.py upload_tflite_model_file download_tpu_model_file -p proxy
python edge_tpu_comiler_api.py mobilenet.tflite mobilenet_tpu.tflite -p http://127.0.0.1:1080
```

For python:
```python
from edge_tpu_compiler_api import compile
# The proxy defaults to None
compile(upload_tflite_model_file, download_tpu_model_file, proxy)
```
