import logging
import torch
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- КОНФИГУРАЦИЯ ---
BOT_TOKEN = "7638412804:AAEeHev6ApucqfrJDBizejoY9GDX9bVycOc" # Не забудь перевыпустить токен!
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./Qwen2.5-7B-Instruct-012940/lora_adapter"
MAX_CONTEXT_TOKENS = 2048
MAX_NEW_TOKENS = 512

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- ЗАГРУЗКА МОДЕЛИ ---
print("Загрузка модели и адаптера...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, 
    quantization_config=bnb_config, 
    device_map="auto", 
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

user_histories = {}
generation_lock = asyncio.Lock()

# --- ФУНКЦИИ ---

def trim_history(history, max_tokens):
    while True:
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if inputs.input_ids.shape[1] + MAX_NEW_TOKENS <= max_tokens:
            return history
        if not history: return []
        history.pop(0)

def generate_response_sync(history):
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Жесткая очистка от артефактов
    response = response.split("<|im_end|>")[0]
    response = response.split("<|endoftext|>")[0]
    response = response.split("user\n")[0]
    response = response.replace("assistant\n", "").strip()
    return response

# --- ОБРАБОТЧИКИ (ВАЖНО: error_handler должен быть определен!) ---

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логирует ошибки, вызванные обновлениями."""
    logger.error(msg="Исключение при обработке обновления:", exc_info=context.error)
    # Если это сообщение, можно уведомить пользователя
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("☢️ Произошла внутренняя ошибка сервера (Timeout или GPU Error).")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_histories[update.effective_user.id] = []
    await update.message.reply_text("Контекст очищен. Я готов!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not update.message or not update.message.text: return

    async with generation_lock:
        if user_id not in user_histories: user_histories[user_id] = []
        user_histories[user_id].append({"role": "user", "content": update.message.text})
        user_histories[user_id] = trim_history(user_histories[user_id], MAX_CONTEXT_TOKENS)
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')

        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(None, generate_response_sync, user_histories[user_id])
        
        user_histories[user_id].append({"role": "assistant", "content": response_text})
        await update.message.reply_text(response_text)

# --- ЗАПУСК ---

if __name__ == '__main__':
    request_config = HTTPXRequest(
        connect_timeout=120.0, 
        read_timeout=300.0,    
        write_timeout=120.0,
        connection_pool_size=8
    )

    builder = ApplicationBuilder().token(BOT_TOKEN).request(request_config)
    # builder.proxy("http://...") # Раскомментируй, если нужно
    
    application = builder.build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('reset', start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Теперь ошибка NameError исчезнет, так как функция определена выше
    application.add_error_handler(error_handler)
    
    print("Бот запущен...")
    application.run_polling(drop_pending_updates=True)