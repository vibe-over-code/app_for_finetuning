# Создайте файл test_gpu.py
import torch
# Добавьте в начало скрипта проверку версии
import transformers
print(f"Transformers version: {transformers.__version__}")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Проверка вычислений
x = torch.randn(3, 3).cuda()
print(f"✅ GPU вычисления работают: {x.mean()}")
