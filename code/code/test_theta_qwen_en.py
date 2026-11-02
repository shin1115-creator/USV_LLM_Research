
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama

from config import (
    MODEL_DIR,
    MODEL_FILENAME,
    MODEL_CONFIG,
    GENERATION_CONFIG,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

# シナリオを指定する。省略した場合は従来の横切りを使用する。
parser = argparse.ArgumentParser(description="Theta*経路の説明文をQwenで生成")
parser.add_argument(
    "--scenario",
    choices=("crossing", "headon", "overtaking"),
    default="crossing",
    help="対象シナリオ（初期値: crossing）",
)
args = parser.parse_args()

# 入力データ名・結果ファイル接頭辞はシナリオごとに切り替える。
SCENARIO_FILES = {
    "crossing": ("theta_001_llm_input.json", "theta_001"),
    "headon": ("theta_headon_001_llm_input.json", "theta_headon_001"),
    "overtaking": ("theta_overtaking_001_llm_input.json", "theta_overtaking_001"),
}
input_filename, output_prefix = SCENARIO_FILES[args.scenario]


INPUT_PATH = (
    PROJECT_DIR / "data" / "theta_star"
    / input_filename
)

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    source = json.load(f)

# 評価用の模範理由は渡さない
data = source["llm_input"]
routes = data["routes"]
correct_route = data["correct_route"]

if correct_route not in ("A", "B"):
    raise ValueError("Invalid correct route.")

# 英語で経路情報を整理
def route_summary(route_id):
    route = routes[route_id]
    valid = (
        "Yes"
        if route["valid_under_baseline_bumper"]
        else "No"
    )

    return (
        f"Route {route_id}:\n"
        f"- Route length: {route['route_length_m']} m\n"
        f"- Minimum separation: {route['min_separation_m']} m\n"
        f"- Meets baseline Bumper constraint: {valid}"
    )

# 元の日本語プロンプトと同じ情報・指示を使用
system_message = (
    "You are an AI that explains collision-avoidance "
    "routes for a small unmanned surface vessel. "
    "Explain using only the supplied information. "
    "Do not invent facts."
)

prompt = f"""
The following data compares two collision-avoidance
routes generated using Theta*.

Encounter type: {data["encounter_type"]}

Selection criterion:
A route must not intrude into the baseline Bumper zone.

Route comparison:
{route_summary("A")}

{route_summary("B")}

Predefined correct route: Route {correct_route}

Task:
Explain briefly in English why the predefined
correct route is appropriate.

Requirements:
- Do not choose a route yourself.
- Use the baseline Bumper assessment as evidence.
- Compare Route A and Route B.
- Do not introduce facts or numbers not provided.
- Do not claim that real-world navigation safety
  is guaranteed.
- Output only the explanation.
""".strip()

model_path = MODEL_DIR / MODEL_FILENAME

if not model_path.exists():
    raise FileNotFoundError(model_path)

print("Model:", MODEL_FILENAME)
print("Loading model...")

llm = Llama(
    model_path=str(model_path),
    **MODEL_CONFIG
)

print("Model loaded successfully!")
print("\nGenerating English explanation...")

start = time.perf_counter()

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ],
    **GENERATION_CONFIG
)

elapsed = time.perf_counter() - start

choice = response["choices"][0]
generated_text = choice["message"]["content"] or ""
finish_reason = choice["finish_reason"]
usage = response.get("usage", {})

print("\n【Generated explanation】")
print(generated_text)
print(f"\nElapsed time: {elapsed:.2f} seconds")
print("Completion tokens:", usage.get("completion_tokens"))
print("Finish reason:", finish_reason)

# モデル名・言語・日時でファイルを識別
model_name = Path(MODEL_FILENAME).stem
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

output_path = (
    RESULTS_DIR
    / f"{output_prefix}_qwen_{model_name}_en_{timestamp}.json"
)

result = {
    "timestamp": datetime.now().isoformat(),
    "scenario_id": data["scenario_id"],
    "model": MODEL_FILENAME,
    "language": "en",
    "system_message": system_message,
    "prompt": prompt,
    "correct_route": correct_route,
    "generated_text": generated_text,
    "elapsed_sec": round(elapsed, 2),
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "finish_reason": finish_reason,
    "generation_config": GENERATION_CONFIG,
    "source_file": str(INPUT_PATH)
}

with open(output_path, "x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nResult saved:", output_path)
