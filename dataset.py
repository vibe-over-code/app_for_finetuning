import json
import time
import re
from requests import post

# --- КОНФИГУРАЦИЯ ---
API_KEY = "7xMTOT8C4GNXwTEUB2TQT8jJQYaxKMsg"  # Вставьте ваш ключ
INPUT_FILE = "mk.txt"
OUTPUT_FILE = "marx_dataset.jsonl"
URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-latest" # Лучшая модель для сложной логики

# Настройки нарезки текста
CHUNK_SIZE = 3000  # Символов на один кусок (примерно 700-800 токенов)
OVERLAP = 200      # Перекрытие, чтобы не терять смысл на стыках

def get_chunks(text, size, overlap):
    start = 0
    while start < len(text):
        yield text[start : start + size]
        start += size - overlap

def clean_json_response(content):
    """Очищает ответ от markdown (```json ... ```) если он есть"""
    if "```" in content:
        # Ищем контент между ```json и ``` или просто ```
        match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return content.strip()

def generate_qa(chunk_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # ПРОМПТ - САМАЯ ВАЖНАЯ ЧАСТЬ
    # Мы просим вернуть массив JSON объектов
    system_prompt = """
    Ты — архитектор данных для обучения LLM. Твоя задача — создать датасет для Fine-Tuning модели, которая станет цифровой копией политического лидера.
    
    1. Прочитай предоставленный отрывок из книги.
    2. Сформулируй 2-3 глубоких вопроса, на которые этот отрывок дает ответ. Вопросы могут быть как теоретическими, так и прикладными ("Почему рабочему платят мало?", "Что такое товар?").
    3. Напиши ответ от первого лица (как автор). Ответ должен быть идеологически выдержанным, использовать терминологию (стоимость, эксплуатация, пролетариат) и следовать логике текста.
    4. ВЕРНИ ТОЛЬКО ВАЛИДНЫЙ JSON-МАССИВ. Без лишних слов.
    
    Формат JSON:
    [
        {"instruction": "Вопрос пользователя", "input": "", "output": "Идеологический ответ"},
        {"instruction": "...", "input": "", "output": "..."}
    ]
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Текст для анализа:\n\n{chunk_text}"}
        ],
        "temperature": 0.4, # Пониже, чтобы JSON был валидным
        "response_format": {"type": "json_object"} # Принуждаем к JSON (фича Mistral API)
    }

    try:
        response = post(URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        # Mistral иногда возвращает обертку, например {"data": [...]}, проверяем это
        parsed = json.loads(clean_json_response(content))
        
        # Если вернулся словарь с ключом, вытаскиваем список, если список - оставляем
        if isinstance(parsed, dict):
            # Часто ключи бывают "pairs", "data", "qa"
            for key in parsed:
                if isinstance(parsed[key], list):
                    return parsed[key]
            return [] # Не нашли список
            
        return parsed # Это уже список

    except Exception as e:
        print(f"Ошибка API или парсинга: {e}")
        return []

def main():
    print(f"📖 Чтение {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print("Файл mk.txt не найден!")
        return

    chunks = list(get_chunks(full_text, CHUNK_SIZE, OVERLAP))
    total_chunks = len(chunks)
    print(f"🔪 Текст разбит на {total_chunks} частей.")

    print("🚀 Старт генерации...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f: # режим 'a' чтобы дописывать
        for i, chunk in enumerate(chunks):
            print(f"Обработка части {i+1}/{total_chunks}...", end=" ", flush=True)
            
            qa_pairs = generate_qa(chunk)
            
            if qa_pairs:
                count = 0
                for pair in qa_pairs:
                    # Простая валидация полей
                    if "instruction" in pair and "output" in pair:
                        # Записываем в JSONL (одна строка = один json объект)
                        json.dump(pair, out_f, ensure_ascii=False)
                        out_f.write("\n")
                        count += 1
                print(f"✅ Готово: +{count} примеров.")
            else:
                print("⚠️ Пустой ответ или ошибка.")
            
            # Пауза, чтобы не упереться в лимиты API (Rate Limit)
            time.sleep(0.1)

    print(f"\n🎉 Генерация завершена! Результат в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()