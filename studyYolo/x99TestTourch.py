import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加到代码开头

print(torch.__version__)
print(torch.cuda.is_available())
