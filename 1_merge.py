import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gc

base_model_id = "Qwen/Qwen2.5-7B-Instruct" # Берется из кеша HF
adapter_path = "Qwen2.5-7B-Instruct-165929/lora_adapter"  # Твоя папка с адаптером
save_path = "merged_model_hf"                # Сюда сохранится результат

print("1. Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("2. Загрузка базовой модели на CPU (чтобы не превысить 10GB VRAM)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu", 
    low_cpu_mem_usage=True
)

print("3. Применение адаптера LoRA...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("4. Слияние весов (merge_and_unload)...")
merged_model = model.merge_and_unload()

print("5. Сохранение новой модели...")
merged_model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)

# Очистка
del base_model, model, merged_model
gc.collect()

print(f"✅ Готово! Модель сохранена в папку: {save_path}")