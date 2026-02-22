import json
import os
import uuid
import argparse
from pathlib import Path
from mistralai import Mistral


BASE_INSTRUCTION = (
    "Сгенерируй диалог в жанре киберпанк и верни ТОЛЬКО валидный JSON "
    "строго следующей структуры:\n"
    "{"
    '"msg_id": "<uuid>",'
    '"corp_name": "<имя персонажа>",'
    '"npc_text": "<первая реплика NPC>",'
    '"replies": [{"id": 1, "text": "..."}, {"id": 2, "text": "..."}],'
    '"answers": {"1": "...", "2": "..."}'
    "}"
)


EXTRACTION_PROMPT = """
Ты анализируешь киберпанк-диалог.

Верни строго JSON:

{
  "setting": "...",
  "mood": "...",
  "keywords": ["...", "...", "..."]
}

Никакого текста вне JSON.
"""


def extract_features(client, dialog_json):
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": dialog_json}
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        return {"setting": "", "mood": "", "keywords": []}


def build_input(features):
    parts = []

    if features["setting"]:
        parts.append(f"Локация: {features['setting']}")
    if features["mood"]:
        parts.append(f"Настроение: {features['mood']}")
    if features["keywords"]:
        parts.append("Ключевые слова: " + ", ".join(features["keywords"]))

    return "\n".join(parts)


def process_file(input_path: Path, output_path: Path):
    api_key = "ftH40nqIasXUPPKYgWINxwdoEWQarNZ0"
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")

    client = Mistral(api_key=api_key)

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            if not raw.get("msg_id"):
                raw["msg_id"] = str(uuid.uuid4())

            dialog_str = json.dumps(raw, ensure_ascii=False)

            features = extract_features(client, dialog_str)
            input_text = build_input(features)

            dataset_row = {
                "instruction": BASE_INSTRUCTION,
                "input": input_text,
                "output": dialog_str
            }

            # КЛЮЧЕВОЙ МОМЕНТ:
            # ensure_ascii=False и одна запись = одна строка
            outfile.write(json.dumps(dataset_row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    process_file(Path(args.input), Path(args.output))
    print("Done. JSONL dataset created.")


if __name__ == "__main__":
    main()