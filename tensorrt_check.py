import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())

import tensorrt_llm
print(tensorrt_llm.__version__)




