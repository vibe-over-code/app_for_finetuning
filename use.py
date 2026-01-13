from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-marx-003721/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 1. Добавляем конфиг квантования для инференса
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. Грузим базу в 4 битах
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    quantization_config=bnb_config, # ОБЯЗАТЕЛЬНО
    device_map="auto", 
    trust_remote_code=True
)

# 3. Накладываем адаптер
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

while True:
    inp = input()
    prompt = inp
    # Формат промпта Qwen
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))