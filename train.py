import torch
import os
import json
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ========== КОНФИГУРАЦИЯ ==========
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DATASET_PATH = "marx_dataset.jsonl" # Файл, который мы генерировали через Mistral
OUTPUT_DIR = f"./qwen-marx-{datetime.now().strftime('%H%M%S')}"
MAX_LENGTH = 1024 # Оптимально для 10GB VRAM
# ===================================

def main():
    print(f"🚀 Запуск обучения на {torch.cuda.get_device_name(0)}")

    # 1. Настройка квантования (чтобы влезть в 10 ГБ)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Загрузка токенизатора и модели
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Подготовка модели к PEFT
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Настройка LoRA
    # Для Qwen 2.5 важно указать правильные целевые модули
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Загрузка и подготовка датасета
    # Ожидается формат JSONL с полями instruction и output
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    def tokenize_function(examples):
        # Формируем текст в стиле чата Qwen
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"<|im_start|>user\n{examples['instruction'][i]}<|im_end|>\n<|im_start|>assistant\n{examples['output'][i]}<|im_end|>"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 6. Параметры обучения (оптимизировано под RTX 3080)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # Эффективный батч = 4
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        fp16=True, # На 3080 можно использовать bf16=True, если драйверы свежие
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_8bit", # Очень важно для экономии VRAM
        report_to="none"
    )

    # 7. Запуск
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("🛠 Начинаем процесс дообучения...")
    trainer.train()

    # 8. Сохранение адаптера
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Готово! Адаптер сохранен в {OUTPUT_DIR}/lora_adapter")

if __name__ == "__main__":
    main()