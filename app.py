"""
Визуальное приложение для Fine-Tuning моделей
"""
import gradio as gr
import os
import json
import threading
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
