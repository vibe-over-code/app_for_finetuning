"""
Визуальное приложение для Fine-Tuning моделей
"""
import gradio as gr
import os
import json
import threading
import subprocess
import shutil
import sys
import gc
from pathlib import Path
from trainer_module import train_model
from memory_estimator import (
    estimate_model_memory,
    estimate_training_memory,
    get_available_memory,
    check_memory_sufficiency,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# Глобальная переменная для обучения
training_progress = []
training_status = {"running": False, "result": None}

# Кэш для инференса
inference_cache = {}

PROJECT_ROOT = Path(__file__).resolve().parent


def estimate_memory_requirements(
    model_name,
    quantization_bits,
    batch_size,
    max_length,
    gradient_accumulation_steps,
    use_gradient_checkpointing,
    use_8bit_optimizer
):
    """Оценивает требования к памяти"""
    if not model_name:
        return "⚠️ Укажите имя модели"
    
    try:
        # Преобразуем quantization_bits
        qb = None if quantization_bits == "Нет" else int(quantization_bits)
        
        # Оценка памяти модели
        model_memory = estimate_model_memory(model_name, qb)
        
        if 'error' in model_memory:
            return f"❌ Ошибка оценки: {model_memory['error']}"
        
        # Оценка памяти для обучения
        training_memory = estimate_training_memory(
            model_memory['model_memory_gb'],
            batch_size,
            max_length,
            gradient_accumulation_steps,
            use_gradient_checkpointing,
            use_8bit_optimizer
        )
        
        # Получение доступной памяти
        available_memory = get_available_memory()
        
        if available_memory['available_gb'] == 0:
            memory_info = "⚠️ CUDA недоступна. Обучение будет невозможно."
        else:
            memory_check = check_memory_sufficiency(
                training_memory['total_memory_gb'],
                available_memory['available_gb']
            )
            
            memory_info = f"""
📊 **Оценка памяти:**

**Модель:**
- Параметров: ~{model_memory['num_params']/1e9:.2f}B
- Память модели: {model_memory['model_memory_gb']} GB

**Обучение:**
- Модель: {training_memory['model_memory_gb']} GB
- Активации: {training_memory['activation_memory_gb']} GB
- Градиенты: {training_memory['gradient_memory_gb']} GB
- Оптимизатор: {training_memory['optimizer_memory_gb']} GB
- Данные: {training_memory['data_memory_gb']} GB
- Overhead: {training_memory['overhead_gb']} GB
- **ИТОГО: {training_memory['total_memory_gb']} GB**

**Доступно:**
- GPU: {available_memory['device_name']}
- Всего памяти: {available_memory['total_gb']} GB
- Доступно: {available_memory['available_gb']} GB
- Используется: {available_memory['allocated_gb']} GB

**{memory_check['recommendation']}**
"""
        
        return memory_info
        
    except Exception as e:
        return f"❌ Ошибка при оценке памяти: {str(e)}"


def progress_callback(message):
    """Callback для обновления прогресса"""
    training_progress.append(message)
    print(message)  # Также выводим в консоль


def start_training(
    model_name,
    dataset_file,
    output_dir,
    adapter_path,
    continue_adapter,
    max_length,
    quantization_bits,
    use_double_quant,
    batch_size,
    gradient_accumulation_steps,
    learning_rate,
    num_train_epochs,
    lora_r,
    lora_alpha,
    lora_dropout,
    save_steps,
    logging_steps,
    use_gradient_checkpointing,
    use_8bit_optimizer,
):
    """Запускает обучение в отдельном потоке"""
    global training_status, training_progress
    
    if training_status["running"]:
        return "⚠️ Обучение уже запущено!"
    
    if not model_name:
        return "❌ Укажите имя модели"
    
    if dataset_file is None:
        return "❌ Загрузите датасет"

    # Преобразуем quantization_bits в число
    qb = None if quantization_bits == "Нет" else int(quantization_bits)

    # Путь к адаптеру (может быть пустым)
    adapter_path = adapter_path.strip() if isinstance(adapter_path, str) else ""
    if not continue_adapter:
        adapter_path = None

    # Сохраняем загруженный файл
    dataset_path = dataset_file.name if hasattr(dataset_file, "name") else dataset_file
    
    training_status["running"] = True
    training_progress = []
    
    def train_thread():
        global training_status
        try:
            result = train_model(
                model_name=model_name,
                dataset_path=dataset_path,
                output_dir=output_dir if output_dir else None,
                adapter_path=adapter_path,
                max_length=int(max_length),
                quantization_bits=qb,
                use_double_quant=use_double_quant,
                batch_size=int(batch_size),
                gradient_accumulation_steps=int(gradient_accumulation_steps),
                learning_rate=float(learning_rate),
                num_train_epochs=int(num_train_epochs),
                lora_r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                save_steps=int(save_steps),
                logging_steps=int(logging_steps),
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_8bit_optimizer=use_8bit_optimizer,
                progress_callback=progress_callback
            )
            training_status["result"] = result
        except Exception as e:
            training_status["result"] = {
                'success': False,
                'error': 'other',
                'message': str(e)
            }
        finally:
            training_status["running"] = False
    
    thread = threading.Thread(target=train_thread)
    thread.start()
    
    return "🚀 Обучение запущено! Следите за прогрессом в логах."


def load_inference_model(base_model_name, adapter_path):
    """
    Загружает (или берёт из кэша) модель для инференса с выбранным адаптером
    """
    key = (base_model_name, adapter_path)
    if key in inference_cache:
        return inference_cache[key]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Для инференса требуется GPU.")

    # Квантование для инференса
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.isdir(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    inference_cache[key] = (tokenizer, model)
    return tokenizer, model


def run_inference(
    base_model_name,
    adapter_path,
    prompt,
    max_new_tokens,
    temperature,
):
    """Инференс с выбранной моделью и адаптером"""
    if not base_model_name:
        return "❌ Укажите базовую модель"
    if not prompt:
        return "❗ Введите запрос"

    try:
        tokenizer, model = load_inference_model(base_model_name, adapter_path)

        # Формат промпта по умолчанию — как для Qwen
        text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"❌ Ошибка при инференсе: {str(e)}"


def get_progress():
    """Получает текущий прогресс обучения"""
    if training_progress:
        return "\n".join(training_progress[-50:])  # Последние 50 строк
    return "Ожидание запуска обучения..."


def check_status():
    """Проверяет статус обучения"""
    if training_status["running"]:
        return "🔄 Обучение выполняется..."
    elif training_status["result"]:
        result = training_status["result"]
        if result.get("success"):
            return f"✅ {result.get('message', 'Обучение завершено успешно!')}"
        else:
            return f"❌ {result.get('message', 'Ошибка при обучении')}"
    else:
        return "⏸ Ожидание запуска..."


def resolve_llama_cpp_dir(llama_cpp_path):
    """Возвращает директорию llama.cpp с fallback на корень текущего репозитория."""
    raw_path = llama_cpp_path.strip() if isinstance(llama_cpp_path, str) else ""
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve(), None
        return PROJECT_ROOT, (
            f"⚠️ Путь llama.cpp не найден: {raw_path}\n"
            f"Использую fallback: {PROJECT_ROOT}"
        )

    return PROJECT_ROOT, f"ℹ️ Путь llama.cpp не указан. Использую: {PROJECT_ROOT}"


def find_quantize_binary(llama_dir):
    """Ищет бинарник квантования llama.cpp в типичных путях."""
    candidates = [
        llama_dir / "llama-quantize.exe",
        llama_dir / "llama-quantize",
        llama_dir / "quantize.exe",
        llama_dir / "quantize",
        llama_dir / "build" / "bin" / "llama-quantize.exe",
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "quantize.exe",
        llama_dir / "build" / "bin" / "quantize",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def run_command(command, cwd):
    """Запускает команду и возвращает (успех, лог)."""
    process = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    command_line = " ".join(str(part) for part in command)
    log = f"$ {command_line}\n{process.stdout}"
    if process.stderr:
        log += f"\n[stderr]\n{process.stderr}"
    return process.returncode == 0, log.strip()


def merge_and_build_gguf(
    base_model_id,
    adapter_path,
    llama_cpp_path,
    quantization_type,
    merged_folder_name,
    fp16_filename,
    quantized_filename,
):
    """Полный пайплайн: merge LoRA -> convert в f16 GGUF -> quantize."""
    if not base_model_id or not str(base_model_id).strip():
        return "❌ Укажите базовую модель (HF ID или локальный путь)."

    adapter_path = adapter_path.strip() if isinstance(adapter_path, str) else ""
    if not adapter_path:
        return "❌ Укажите путь к LoRA-адаптеру."
    if not os.path.isdir(adapter_path):
        return f"❌ Папка адаптера не найдена: {adapter_path}"

    merged_folder_name = (
        merged_folder_name.strip()
        if isinstance(merged_folder_name, str) and merged_folder_name.strip()
        else "merged_model_hf"
    )
    fp16_filename = (
        fp16_filename.strip()
        if isinstance(fp16_filename, str) and fp16_filename.strip()
        else "model.f16.gguf"
    )
    quantized_filename = (
        quantized_filename.strip()
        if isinstance(quantized_filename, str) and quantized_filename.strip()
        else f"model.{quantization_type}.gguf"
    )

    llama_dir, fallback_message = resolve_llama_cpp_dir(llama_cpp_path)
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        return (
            f"❌ Не найден convert_hf_to_gguf.py в: {llama_dir}\n"
            "Проверьте путь к папке llama.cpp."
        )

    quantize_bin = find_quantize_binary(llama_dir)
    if not quantize_bin:
        return (
            f"❌ Не найден бинарник квантования (llama-quantize/quantize) в: {llama_dir}\n"
            "Соберите llama.cpp перед запуском."
        )

    merged_dir = llama_dir / merged_folder_name
    fp16_path = llama_dir / fp16_filename
    quantized_path = llama_dir / quantized_filename

    logs = []
    if fallback_message:
        logs.append(fallback_message)
    logs.append(f"📁 Рабочая директория llama.cpp: {llama_dir}")
    logs.append(f"📁 Папка merged модели: {merged_dir}")
    logs.append(f"📄 FP16 GGUF: {fp16_path}")
    logs.append(f"📄 Квантованный GGUF: {quantized_path}")
    logs.append(f"🛠 Квантайзер: {quantize_bin}")

    base_model = None
    model = None
    merged_model = None

    try:
        logs.append("\n[1/3] Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        logs.append("[1/3] Загрузка базовой модели на CPU...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        logs.append("[1/3] Применение LoRA-адаптера...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        logs.append("[1/3] Слияние весов (merge_and_unload)...")
        merged_model = model.merge_and_unload()

        if merged_dir.exists():
            shutil.rmtree(merged_dir)

        logs.append("[1/3] Сохранение merged модели в корень llama.cpp...")
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

        logs.append("\n[2/3] Конвертация HF -> FP16 GGUF...")
        convert_ok, convert_log = run_command(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(fp16_path),
                "--outtype",
                "f16",
            ],
            cwd=llama_dir,
        )
        logs.append(convert_log)
        if not convert_ok:
            logs.append("❌ Ошибка на этапе конвертации в GGUF.")
            return "\n\n".join(logs)

        logs.append(f"\n[3/3] Квантование в {quantization_type}...")
        quant_ok, quant_log = run_command(
            [
                str(quantize_bin),
                str(fp16_path),
                str(quantized_path),
                quantization_type,
            ],
            cwd=llama_dir,
        )
        logs.append(quant_log)
        if not quant_ok:
            logs.append("❌ Ошибка на этапе квантования.")
            return "\n\n".join(logs)

        logs.append("\n✅ Готово: merge + convert + quantize завершены успешно.")
        return "\n\n".join(logs)

    except Exception as e:
        logs.append(f"\n❌ Ошибка пайплайна: {e}")
        return "\n\n".join(logs)
    finally:
        del base_model, model, merged_model
        gc.collect()


# Создание интерфейса
with gr.Blocks() as app:
    gr.Markdown("# 🚀 Fine-Tuning Assistant")
    gr.Markdown("Приложение для дообучения языковых моделей с использованием LoRA")
    
    with gr.Tabs():
        with gr.TabItem("⚙️ Настройки"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Модель")
                    model_name = gr.Textbox(
                        label="Имя модели (HuggingFace ID или путь)",
                        value="Qwen/Qwen2.5-7B-Instruct",
                        placeholder="Например: Qwen/Qwen2.5-7B-Instruct или ./my-model"
                    )
                    
                    gr.Markdown("### Датасет")
                    dataset_file = gr.File(
                        label="Загрузите датасет (JSONL формат)",
                        file_types=[".jsonl"],
                        type="filepath"
                    )
                    
                    gr.Markdown("### Выходная директория")
                    output_dir = gr.Textbox(
                        label="Директория для сохранения (оставьте пустым для авто-генерации)",
                        value="",
                        placeholder="Например: ./my-trained-model"
                    )

                    gr.Markdown("### Адаптер")
                    adapter_path = gr.Textbox(
                        label="Путь к существующему LoRA-адаптеру (опционально)",
                        value="",
                        placeholder="./qwen-marx-003721/lora_adapter"
                    )
                    continue_adapter = gr.Checkbox(
                        label="Дообучать существующий адаптер (а не создавать новый)",
                        value=False
                    )
                
                with gr.Column():
                    gr.Markdown("### Параметры обучения")
                    max_length = gr.Slider(
                        label="Максимальная длина последовательности",
                        minimum=128,
                        maximum=4096,
                        value=1024,
                        step=128
                    )
                    
                    quantization_bits = gr.Radio(
                        label="Квантование (BitsAndBytes)",
                        choices=["4", "8", "Нет"],
                        value="4",
                        info="4-bit рекомендуется для экономии памяти"
                    )
                    
                    use_double_quant = gr.Checkbox(
                        label="Использовать двойное квантование (только для 4-bit)",
                        value=True
                    )
                    
                    batch_size = gr.Slider(
                        label="Размер батча",
                        minimum=1,
                        maximum=8,
                        value=1,
                        step=1
                    )
                    
                    gradient_accumulation_steps = gr.Slider(
                        label="Шаги накопления градиентов",
                        minimum=1,
                        maximum=32,
                        value=4,
                        step=1,
                        info="Эффективный батч = batch_size × gradient_accumulation_steps"
                    )
                    
                    learning_rate = gr.Number(
                        label="Скорость обучения",
                        value=2e-4,
                        precision=6
                    )
                    
                    num_train_epochs = gr.Slider(
                        label="Количество эпох",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Параметры LoRA")
                    lora_r = gr.Slider(
                        label="LoRA Rank (r)",
                        minimum=4,
                        maximum=128,
                        value=16,
                        step=4
                    )
                    
                    lora_alpha = gr.Slider(
                        label="LoRA Alpha",
                        minimum=4,
                        maximum=128,
                        value=32,
                        step=4
                    )
                    
                    lora_dropout = gr.Slider(
                        label="LoRA Dropout",
                        minimum=0.0,
                        maximum=0.5,
                        value=0.05,
                        step=0.01
                    )
                
                with gr.Column():
                    gr.Markdown("### Дополнительные параметры")
                    save_steps = gr.Slider(
                        label="Сохранять каждые N шагов",
                        minimum=10,
                        maximum=500,
                        value=50,
                        step=10
                    )
                    
                    logging_steps = gr.Slider(
                        label="Логировать каждые N шагов",
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1
                    )
                    
                    use_gradient_checkpointing = gr.Checkbox(
                        label="Использовать Gradient Checkpointing",
                        value=True,
                        info="Экономит память, но замедляет обучение"
                    )
                    
                    use_8bit_optimizer = gr.Checkbox(
                        label="Использовать 8-bit оптимизатор",
                        value=True,
                        info="Экономит память"
                    )
        
        with gr.TabItem("📊 Оценка памяти"):
            gr.Markdown("### Предсказание требований к памяти")
            estimate_btn = gr.Button("Оценить память", variant="primary")
            memory_info = gr.Markdown("Нажмите кнопку для оценки памяти")
            
            estimate_btn.click(
                fn=lambda qb, *args: estimate_memory_requirements(
                    args[0],
                    None if qb == "Нет" else int(qb),
                    *args[1:]
                ),
                inputs=[
                    quantization_bits,
                    model_name,
                    batch_size,
                    max_length,
                    gradient_accumulation_steps,
                    use_gradient_checkpointing,
                    use_8bit_optimizer
                ],
                outputs=memory_info
            )
        
        with gr.TabItem("🚀 Обучение"):
            gr.Markdown("### Запуск обучения")
            
            with gr.Row():
                start_btn = gr.Button("Начать обучение", variant="primary", size="lg")
                refresh_btn = gr.Button("🔄 Обновить", variant="secondary")
            
            status_text = gr.Textbox(
                label="Статус",
                value="⏸ Ожидание запуска...",
                interactive=False
            )
            
            progress_log = gr.Textbox(
                label="Логи обучения",
                lines=20,
                max_lines=50,
                interactive=False,
                value="Ожидание запуска обучения..."
            )
            
            # Автообновление прогресса
            def update_progress():
                return get_progress(), check_status()
            
            start_btn.click(
                fn=start_training,
                inputs=[
                    model_name,
                    dataset_file,
                    output_dir,
                    adapter_path,
                    continue_adapter,
                    max_length,
                    quantization_bits,
                    use_double_quant,
                    batch_size,
                    gradient_accumulation_steps,
                    learning_rate,
                    num_train_epochs,
                    lora_r,
                    lora_alpha,
                    lora_dropout,
                    save_steps,
                    logging_steps,
                    use_gradient_checkpointing,
                    use_8bit_optimizer
                ],
                outputs=status_text
            ).then(
                fn=update_progress,
                inputs=None,
                outputs=[progress_log, status_text]
            )
            
            # Кнопка обновления прогресса
            refresh_btn.click(
                fn=update_progress,
                inputs=None,
                outputs=[progress_log, status_text]
            )

            # Автообновление логов с помощью таймера
            # В текущей версии Gradio нет прямого метода .change() для gr.Timer.
            # Вместо этого, логи будут обновляться при загрузке страницы и по кнопке 'Обновить'.

        with gr.TabItem("💬 Инференс"):
            gr.Markdown("### Запуск модели с выбранным адаптером")

            with gr.Row():
                with gr.Column():
                    base_model_infer = gr.Textbox(
                        label="Базовая модель (HuggingFace ID или путь)",
                        value="Qwen/Qwen2.5-7B-Instruct",
                    )
                    adapter_infer = gr.Textbox(
                        label="Путь к адаптеру (LoRA)",
                        value="./qwen-marx-003721/lora_adapter",
                    )
                    max_new_tokens_infer = gr.Slider(
                        label="Максимум новых токенов",
                        minimum=16,
                        maximum=512,
                        value=250,
                        step=16,
                    )
                    temperature_infer = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.05,
                    )
                with gr.Column():
                    prompt_infer = gr.Textbox(
                        label="Вопрос / запрос",
                        lines=5,
                        placeholder="Введите запрос к модели...",
                    )
                    run_btn = gr.Button("Сгенерировать ответ", variant="primary")
                    output_infer = gr.Textbox(
                        label="Ответ модели",
                        lines=10,
                        interactive=False,
                    )

            run_btn.click(
                fn=run_inference,
                inputs=[
                    base_model_infer,
                    adapter_infer,
                    prompt_infer,
                    max_new_tokens_infer,
                    temperature_infer,
                ],
                outputs=output_infer,
            )

        with gr.TabItem("🧩 Merge + GGUF"):
            gr.Markdown("### Merge LoRA и сборка GGUF в папке llama.cpp")
            gr.Markdown(
                "Укажите путь к llama.cpp. Если поле пустое или путь неверный, будет использован корень текущего репозитория."
            )

            with gr.Row():
                with gr.Column():
                    merge_base_model = gr.Textbox(
                        label="Базовая модель (HF ID или локальный путь)",
                        value="Qwen/Qwen2.5-7B-Instruct",
                    )
                    merge_adapter_path = gr.Textbox(
                        label="Путь к LoRA-адаптеру",
                        value="./qwen-marx-003721/lora_adapter",
                    )
                    llama_cpp_path = gr.Textbox(
                        label="Путь к папке llama.cpp (опционально)",
                        value="",
                        placeholder="Например: D:/llama.cpp",
                    )
                    quantization_type = gr.Dropdown(
                        label="Квантование GGUF",
                        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "Q6_K", "Q4_0", "Q4_K_S"],
                        value="Q4_K_M",
                    )

                with gr.Column():
                    merged_folder_name = gr.Textbox(
                        label="Имя папки merged модели (в корне llama.cpp)",
                        value="merged_model_hf",
                    )
                    fp16_filename = gr.Textbox(
                        label="Имя FP16 GGUF файла",
                        value="model.f16.gguf",
                    )
                    quantized_filename = gr.Textbox(
                        label="Имя квантованного GGUF файла",
                        value="",
                        placeholder="Оставьте пустым для авто-имени: model.<quant>.gguf",
                    )
                    build_gguf_btn = gr.Button("Запустить merge + GGUF", variant="primary")

            merge_log = gr.Textbox(
                label="Лог пайплайна",
                lines=20,
                max_lines=60,
                interactive=False,
                value="Ожидание запуска...",
            )

            build_gguf_btn.click(
                fn=merge_and_build_gguf,
                inputs=[
                    merge_base_model,
                    merge_adapter_path,
                    llama_cpp_path,
                    quantization_type,
                    merged_folder_name,
                    fp16_filename,
                    quantized_filename,
                ],
                outputs=merge_log,
            )


if __name__ == "__main__":
    # Проверка CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA недоступна. Обучение будет невозможно без GPU.")
    
    print("\n🚀 Запуск приложения...")
    print("📱 Откройте браузер и перейдите по адресу: http://localhost:7860")
    print("💡 Если порт занят, приложение автоматически выберет другой порт\n")
    
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft()
    )
