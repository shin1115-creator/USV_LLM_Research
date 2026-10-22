
import json
import csv
import time
from pathlib import Path
from datetime import datetime

from llama_cpp import Llama


# =====================================
# 1. ファイルの場所を設定
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

DATA_PATH = PROJECT_DIR / "data" / "scenario_001.json"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "experiment_results.csv"


# =====================================
# 2. 航海情報をJSONから読み込む
# =====================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    scenario = json.load(f)

nav = scenario["navigation"]
routes = scenario["routes"]
correct = scenario["correct_route"]

print("シナリオ読み込み成功！")
print("ID:", scenario["scenario_id"])


# =====================================
# 3. 共通の入力データを作成
# =====================================

input_data = f"""
【航海情報】
遭遇状況：{scenario['encounter_type']}
船種：{scenario['ship_type']}
視認状態：{scenario['visibility']}

相対距離：{nav['relative_distance_nm']} NM
相対方位：{nav['relative_bearing_deg']} 度
ベアリング変化率：{nav['bearing_rate_deg_min']} 度/分
TCPA：{nav['tcpa_sec']} 秒
DCPA：{nav['dcpa_nm']} NM

【避航経路】
ルートA：{routes['A']['description']}
ルートAの最小離隔距離：{routes['A']['min_clearance_m']} m

ルートB：{routes['B']['description']}
ルートBの最小離隔距離：{routes['B']['min_clearance_m']} m

【指定された正解経路】
ルート{correct}
"""


# =====================================
# 4. 2種類のプロンプトを作成
# =====================================

common_rules = """
【共通条件】
・正解経路を変更しないでください。
・入力情報にない事実を断定しないでください。
・根拠が不足する場合は明示してください。
・日本語で簡潔に説明してください。
"""

# P1：基本型
prompt_p1 = f"""
あなたは小型USVの避航経路を説明するAIです。

{input_data}

【指示】
指定された正解経路が適切である理由を、
航海情報と両経路の特徴に基づいて説明してください。

{common_rules}
"""

# P2：構造化型
prompt_p2 = f"""
あなたは小型USVの避航経路を説明するAIです。

{input_data}

【指示】
指定された正解経路が適切である理由を、
次の順番で説明してください。

1. 衝突リスクの説明
2. ルートAとBの特徴の比較
3. 正解経路が適切である理由

{common_rules}
"""

prompts = {
    "P1_basic": prompt_p1,
    "P2_structured": prompt_p2,
}


# =====================================
# 5. Qwenモデルを読み込む
# =====================================

print("\nモデルを読み込んでいます...")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)

print("モデルの読み込み成功！")


# =====================================
# 6. 文章生成・結果保存
# =====================================

fieldnames = [
    "timestamp",
    "scenario_id",
    "model",
    "prompt_type",
    "prompt",
    "correct_route",
    "generated_text",
    "elapsed_sec",
    "completion_tokens",
    "finish_reason",
    "temperature",
    "max_tokens",
    "seed"
]

for prompt_type, prompt in prompts.items():

    print(f"\n===== {prompt_type} =====")

    start = time.perf_counter()

    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "日本語で正確かつ簡潔に説明してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=450,
        temperature=0.2,
        seed=42
    )

    elapsed = time.perf_counter() - start

    answer = result["choices"][0]["message"]["content"] or ""
    finish_reason = result["choices"][0].get("finish_reason", "")
    tokens = result.get("usage", {}).get("completion_tokens")

    print("\n【生成文章】")
    print(answer)

    print(f"\n処理時間：{elapsed:.2f}秒")
    print(f"生成トークン数：{tokens}")
    print(f"終了理由：{finish_reason}")

    # CSVへ追記
    file_exists = CSV_PATH.exists() and CSV_PATH.stat().st_size > 0

    with open(
        CSV_PATH,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scenario_id": scenario["scenario_id"],
            "model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
            "prompt_type": prompt_type,
            "prompt": prompt,
            "correct_route": correct,
            "generated_text": answer,
            "elapsed_sec": round(elapsed, 3),
            "completion_tokens": tokens,
            "finish_reason": finish_reason,
            "temperature": 0.2,
            "max_tokens": 450,
            "seed": 42            
        })

print("\nすべての実験が完了しました。")
print(f"CSV保存先：{CSV_PATH}")
