"""
Модуль для обучения модели с обработкой ошибок памяти
"""
import torch
import os
import json
import gc
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
from memory_estimator import (
    estimate_model_memory,
    estimate_training_memory,
    get_available_memory,
    check_memory_sufficiency
)


class MemoryError(Exception):
    """Исключение для ошибок памяти"""
    pass


def create_bitsandbytes_config(quantization_bits=4, use_double_quant=True):
    """
    Создает конфигурацию для bitsandbytes
    
    Args:
        quantization_bits: 4, 8 бит или None для без квантования
        use_double_quant: Использовать ли двойное квантование (только для 4-bit)
    
    Returns:
        BitsAndBytesConfig или None
    """
    if quantization_bits is None or quantization_bits == 0:
        return None
    elif quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    else:
        return None


def get_target_modules(model_name):
    """
    Определяет целевые модули для LoRA в зависимости от модели
    
    Args:
        model_name: Имя модели
    
    Returns:
        list: Список целевых модулей
    """
    model_lower = model_name.lower()
    
    # Общие модули для большинства моделей
    common_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    if "qwen" in model_lower or "llama" in model_lower or "mistral" in model_lower:
        return common_modules + ["gate_proj", "up_proj", "down_proj"]
    elif "phi" in model_lower:
        return common_modules + ["fc1", "fc2"]
    else:
        # Дефолтные модули
        return common_modules


def train_model(
    model_name,
    dataset_path,
    output_dir=None,
    max_length=1024,
    quantization_bits=4,
    use_double_quant=True,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    save_steps=50,
    logging_steps=10,
    use_gradient_checkpointing=True,
    use_8bit_optimizer=True,
    progress_callback=None
):
    """
    Основная функция обучения модели
    
    Args:
        model_name: Имя модели или путь
        dataset_path: Путь к датасету JSONL
        output_dir: Директория для сохранения (если None, создается автоматически)
        max_length: Максимальная длина последовательности
        quantization_bits: Битность квантования (4 или 8)
        use_double_quant: Использовать двойное квантование
        batch_size: Размер батча
        gradient_accumulation_steps: Шаги накопления градиентов
        learning_rate: Скорость обучения
        num_train_epochs: Количество эпох
        lora_r: Ранг LoRA
        lora_alpha: Alpha для LoRA
        lora_dropout: Dropout для LoRA
        save_steps: Шаги сохранения
        logging_steps: Шаги логирования
        use_gradient_checkpointing: Использовать gradient checkpointing
        use_8bit_optimizer: Использовать 8-bit оптимизатор
        progress_callback: Функция для обратного вызова прогресса
    
    Returns:
        dict: Результат обучения
    """
    try:
        # Проверка доступности CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. Требуется GPU для обучения.")
        
        device_name = torch.cuda.get_device_name(0)
        if progress_callback:
            progress_callback(f"🚀 Запуск обучения на {device_name}")
        
        # Оценка памяти перед началом
        if progress_callback:
            progress_callback("📊 Оценка требований к памяти...")
        
        model_memory = estimate_model_memory(model_name, quantization_bits)
        if 'error' in model_memory:
            if progress_callback:
                progress_callback(f"⚠️ Не удалось оценить память модели: {model_memory['error']}")
        else:
            training_memory = estimate_training_memory(
                model_memory['model_memory_gb'],
                batch_size,
                max_length,
                gradient_accumulation_steps,
                use_gradient_checkpointing,
                use_8bit_optimizer
            )
            available_memory = get_available_memory()
            memory_check = check_memory_sufficiency(
                training_memory['total_memory_gb'],
                available_memory['available_gb']
            )
            
            if progress_callback:
                progress_callback(f"📊 Оценка памяти:\n"
                                f"  Модель: {model_memory['model_memory_gb']} GB\n"
                                f"  Обучение: {training_memory['total_memory_gb']} GB\n"
                                f"  Доступно: {available_memory['available_gb']} GB\n"
                                f"  {memory_check['recommendation']}")
            
            if not memory_check['sufficient']:
                raise MemoryError(memory_check['recommendation'])
        
        # Настройка квантования
        bnb_config = create_bitsandbytes_config(quantization_bits, use_double_quant)
        
        # Загрузка токенизатора
        if progress_callback:
            progress_callback(f"📥 Загрузка токенизатора {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Загрузка модели с обработкой ошибок памяти
        if progress_callback:
            progress_callback(f"📥 Загрузка модели {model_name}...")
        
        try:
            load_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            if bnb_config is not None:
                load_kwargs["quantization_config"] = bnb_config
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise MemoryError(
                f"Недостаточно памяти для загрузки модели. "
                f"Попробуйте уменьшить quantization_bits или использовать меньшую модель. "
                f"Ошибка: {str(e)}"
            )
        
        # Подготовка модели к PEFT
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        if quantization_bits in [4, 8] and bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # Настройка LoRA
        target_modules = get_target_modules(model_name)
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        if progress_callback:
            model.print_trainable_parameters()
            progress_callback("✅ Модель подготовлена к обучению")
        
        # Загрузка датасета
        if progress_callback:
            progress_callback(f"📂 Загрузка датасета {dataset_path}...")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
        
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Определение формата чата на основе модели
        def get_chat_template(model_name):
            model_lower = model_name.lower()
            if "qwen" in model_lower:
                return lambda inst, out: f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            elif "llama" in model_lower:
                return lambda inst, out: f"<s>[INST] {inst} [/INST] {out} </s>"
            elif "mistral" in model_lower:
                return lambda inst, out: f"<s>[INST] {inst} [/INST] {out} </s>"
            else:
                return lambda inst, out: f"User: {inst}\nAssistant: {out}"
        
        chat_template = get_chat_template(model_name)
        
        def tokenize_function(examples):
            texts = []
            for i in range(len(examples['instruction'])):
                inst = examples['instruction'][i]
                out = examples.get('output', [''])[i] if 'output' in examples else ''
                text = chat_template(inst, out)
                texts.append(text)
            
            return tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        if progress_callback:
            progress_callback("🔄 Токенизация датасета...")
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Создание output_dir
        if output_dir is None:
            model_short = model_name.split('/')[-1]
            output_dir = f"./{model_short}-{datetime.now().strftime('%H%M%S')}"
        
        # Параметры обучения
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            fp16=True,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            optim="paged_adamw_8bit" if use_8bit_optimizer else "adamw_torch",
            report_to="none",
            remove_unused_columns=False
        )
        
        # Создание Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        
        if progress_callback:
            progress_callback("🛠 Начинаем процесс дообучения...")
        
        # Обучение с обработкой ошибок памяти
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            raise MemoryError(
                f"Недостаточно памяти во время обучения. "
                f"Попробуйте уменьшить batch_size, max_length или увеличить gradient_accumulation_steps. "
                f"Ошибка: {str(e)}"
            )
        
        # Сохранение адаптера
        if progress_callback:
            progress_callback(f"💾 Сохранение модели в {output_dir}...")
        
        adapter_path = os.path.join(output_dir, "lora_adapter")
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(output_dir)
        
        result = {
            'success': True,
            'output_dir': output_dir,
            'adapter_path': adapter_path,
            'message': f"✅ Готово! Адаптер сохранен в {adapter_path}"
        }
        
        if progress_callback:
            progress_callback(result['message'])
        
        return result
        
    except MemoryError as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка памяти: {str(e)}")
        return {
            'success': False,
            'error': 'memory',
            'message': str(e)
        }
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Ошибка: {str(e)}")
        return {
            'success': False,
            'error': 'other',
            'message': str(e)
        }
    finally:
        # Очистка памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
