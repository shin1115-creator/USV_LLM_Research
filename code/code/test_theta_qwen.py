
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
    SYSTEM_MESSAGE,
)

# =====================================
# 1. 入出力ファイル
# =====================================

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
    PROJECT_DIR
    / "data"
    / "theta_star"
    / input_filename
)

# モデル名を取得（拡張子.ggufを除く）
model_name = Path(MODEL_FILENAME).stem

# 実行日時を取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# モデル名と日時を含めた保存ファイル名
output_filename = (
    f"{output_prefix}_qwen_{model_name}_{timestamp}.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / output_filename
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================
# 2. Theta*のデータを読み込む
# =====================================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    source_data = json.load(f)

# 評価用の正解理由は読み込ませない
data = source_data["llm_input"]

routes = data["routes"]
correct_route = data["correct_route"]

if correct_route not in ("A", "B"):
    raise ValueError("正解経路が不正です。")

# =====================================
# 3. Qwenへのプロンプトを作成
# =====================================

def route_summary(route_id):
    route = routes[route_id]

    baseline_valid = (
        "満たす"
        if route["valid_under_baseline_bumper"]
        else "満たさない"
    )

    return (
        f"経路{route_id}:\n"
        f"- 経路長: {route['route_length_m']} m\n"
        f"- 最小離隔距離: {route['min_separation_m']} m\n"
        f"- 基準Bumper条件: {baseline_valid}"
    )


prompt = f"""
あなたは小型USVの避航経路について説明するAIです。

以下はTheta*による避航経路の比較データです。

【遭遇状況】
{data["encounter_type"]}

【経路の判断基準】
{data["selection_criterion"]}

【比較経路】
{route_summary("A")}

{route_summary("B")}

【指定された正解経路】
経路{correct_route}

【指示】
指定された正解経路が適切である理由を、
日本語で簡潔に説明してください。

次の点を守ってください。
- 経路の選択は行わない。
- 基準Bumperの判定結果を根拠にする。
- 経路AとBの違いを説明する。
- 入力されていない事実や数値を追加しない。
- 実際の航行安全性を保証する表現は使わない。
- 説明文だけを出力する。
""".strip()

# =====================================
# 4. モデルを読み込む
# =====================================

model_path = MODEL_DIR / MODEL_FILENAME

if not model_path.exists():
    raise FileNotFoundError(
        f"モデルが見つかりません: {model_path}"
    )

print("モデル:", MODEL_FILENAME)
print("モデルを読み込んでいます...")

llm = Llama(
    model_path=str(model_path),
    **MODEL_CONFIG
)

print("モデルの読み込み成功！")

# =====================================
# 5. 説明文を生成
# =====================================

print("\nQwenで説明文を生成しています...")

start = time.perf_counter()

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    **GENERATION_CONFIG
)

elapsed = time.perf_counter() - start

choice = response["choices"][0]
generated_text = choice["message"]["content"] or ""
finish_reason = choice["finish_reason"]

usage = response.get("usage", {})

print("\n【生成された説明文】")
print(generated_text)

print(f"\n処理時間: {elapsed:.2f} 秒")
print("生成トークン数:", usage.get("completion_tokens"))
print("終了理由:", finish_reason)

# =====================================
# 6. 実験結果を保存
# =====================================

result = {
    "timestamp": datetime.now().isoformat(),
    "scenario_id": data["scenario_id"],
    "model": MODEL_FILENAME,
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

with open(OUTPUT_PATH, "x", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n実験結果保存完了:", OUTPUT_PATH)
