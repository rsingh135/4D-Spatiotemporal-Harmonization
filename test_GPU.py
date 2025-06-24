# 在Python环境中运行
import torch
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU数量:", torch.cuda.device_count())
print("当前GPU:", torch.cuda.current_device())
print("GPU名称:", torch.cuda.get_device_name(0))