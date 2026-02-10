import torch
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# --- ЧАСТЬ 1: НАСТРОЙКА ЛОКАЛЬНОЙ МОДЕЛИ (Qwen) ---

model_id = "Qwen/Qwen2.5-7B-Instruct"

# Конфигурация квантования (сжатия)
# ВЛИЯЕТ НА: Потребление видеопамяти.
# load_in_4bit=True заставляет модель весить в 3-4 раза меньше (около 5-6 ГБ VRAM вместо 16 ГБ),
# почти не теряя в уме (немного падает точность).
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Загружаем модель... (это может занять время)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, # Применяем сжатие
    device_map="auto" # Автоматически распределяет по GPU/CPU
)

# Пайплайн генерации
# max_new_tokens: Максимальная длина ответа модели (в словах/токенах).
# temperature: 0.1 делает модель строгой и фактологичной. 0.9 делает её креативной (но может выдумывать).
# repetition_penalty: 1.1 запрещает модели зацикливаться и повторять одни и те же фразы.
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=False,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# --- ЧАСТЬ 2: НАСТРОЙКА ЛОКАЛЬНЫХ ЭМБЕДДИНГОВ ---

# Используем модель, которая хорошо понимает много языков, включая русский.
# model_name: Это название модели на HuggingFace.
# device: 'cuda' для видеокарты, 'cpu' для процессора.
print("Загружаем эмбеддинги...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'} 
)

# --- ЧАСТЬ 3: ПОДГОТОВКА ДАННЫХ (RAG) ---

secret_text = """
Секретная информация о проекте "Омега":
1. Главный инженер проекта — Иван Петров.
2. Запуск запланирован на 5 марта 2025 года.
3. Бюджет составляет 10 миллионов рублей.
Код доступа к серверу: 9988-ALPHA-ZETA.
"""

# Нарезаем текст
# chunk_size=200: Размер одного кусочка текста. Если слишком мало — теряется смысл. Слишком много — забивается память.
# chunk_overlap=20: Перекрытие кусочков, чтобы не разрывать предложения посередине смысла.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.create_documents([secret_text])

# Создаем базу данных
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

# Создаем искателя (Retriever)
# k=2: Сколько самых похожих кусочков текста достать из базы. 
# Если поставить 1, может не хватить инфы. Если 10, модель запутается в лишнем тексте.
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- ЧАСТЬ 4: ЗАПУСК ЦИКЛА ---

# Шаблон промпта. Важно четко сказать модели, где контекст, а где вопрос.
template = """Ты полезный ассистент. Используй только следующий контекст, чтобы ответить на вопрос.
Если в контексте нет ответа, скажи "Я не знаю".

Контекст:
{context}

Вопрос: {question}

Ответ:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Тест
question = "Какой бюджет у проекта и когда запуск?"
print(f"\nВопрос: {question}\n")
response = chain.invoke(question)
print(f"Ответ Qwen: {response}")