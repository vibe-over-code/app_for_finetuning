import json
import time
import re
import uuid
from requests import post

# --- КОНФИГУРАЦИЯ ---
API_KEY = "ftH40nqIasXUPPKYgWINxwdoEWQarNZ0" 
INPUT_FILE = "dataset1.txt"
OUTPUT_FILE = "johnny_expanded_dataset.jsonl"
URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-latest"

def clean_json_response(content):
    if "```" in content:
        match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
    return content.strip()

def generate_expanded_dialogues(chunk_text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # ПРОМПТ ДЛЯ ГЕНЕРАЦИИ 5 ДИАЛОГОВ НА ОДНУ ЦИТАТУ
    system_prompt = """
    Ты — продвинутый генератор контента для Cyberpunk 2077.
    Твоя цель: ВЗЯТЬ ПРЕДОСТАВЛЕННУЮ ЦИТАТУ И СОЗДАТЬ НА ЕЁ ОСНОВЕ 5 РАЗНЫХ ДИАЛОГОВЫХ ТРЕНИЙ.
    
    Стиль Джонни:
    - Ненавидит корпов (Арасака, Милитех).
    - Считает, что город — это выгребная яма[cite: 10].
    - Использует грубый сленг (чумба, хром, импланты).
    - Часто вспоминает войну и свои ошибки[cite: 26, 38].
    
    Стиль Ви:
    - Наемник, который либо соглашается, либо огрызается. Использует уличный жаргон.

    ФОРМАТ ВЫХОДА (JSON):
    {
      "dialogues": [
        {
          "msg_id": "uuid",
          "corp_name": "Johnny Silverhand",
          "npc_text": "Реплика Джонни",
          "replies": [{"id": 1, "text": "Ответ Ви 1"}, {"id": 2, "text": "Ответ Ви 2"}],
          "answers": {"1": "Реакция Джонни на 1", "2": "Реакция Джонни на 2"}
        },
        ... (еще 4 таких объекта)
      ]
    }
    
    ВЕРНИ ТОЛЬКО ЧИСТЫЙ JSON.
    """

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Создай 5 уникальных диалогов, вдохновленных этим текстом:\n\n{chunk_text}"}
        ],
        "temperature": 0.85, # Повышаем для разнообразия
        "response_format": {"type": "json_object"}
    }

    try:
        response = post(URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        data = json.loads(clean_json_response(content))
        return data.get("dialogues", [])
    except Exception as e:
        print(f"Ошибка: {e}")
        return []

def main():
    print("🚬 Загружаем чип с цитатами...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
        # Разделяем по блокам ❝...❞
        blocks = re.findall(r'❝(.*?)❞', raw_text, re.DOTALL)

    print(f"🧬 Исходных блоков: {len(blocks)}. Цель: ~{len(blocks)*5} диалогов.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, block in enumerate(blocks):
            print(f"📡 Масштабируем блок {i+1}/{len(blocks)}...", end=" ", flush=True)
            new_dialogues = generate_expanded_dialogues(block)
            
            if new_dialogues:
                for d in new_dialogues:
                    if "msg_id" not in d or d["msg_id"] == "uuid":
                        d["msg_id"] = str(uuid.uuid4())
                    json.dump(d, out_f, ensure_ascii=False)
                    out_f.write("\n")
                print(f"✅ (+{len(new_dialogues)})")
            else:
                print("❌")
            
            # Небольшая пауза, чтобы API не ругался
            time.sleep(1)

    print(f"\n🔥 Эволюция завершена. Новый датасет: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()