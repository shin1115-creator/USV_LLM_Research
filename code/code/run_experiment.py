
import json
import csv
import time
from datetime import datetime
from uuid import uuid4

from llama_cpp import Llama

from config import (
    MODEL_DIR,
    DATA_DIR,
    RESULTS_DIR,
    MODEL_FILENAME,
    MODEL_CONFIG,
    SCENARIO_FILES,
    PROMPT_TYPES,
    REPEAT_COUNT,
    GENERATION_CONFIG,
    SYSTEM_MESSAGE,
    RESULTS_FILENAME,
)

from prompts import build_prompts


# =====================================
# 1. 保存先の準備
# =====================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / MODEL_FILENAME
csv_path = RESULTS_DIR / RESULTS_FILENAME

if not model_path.is_file():
    raise FileNotFoundError(
        f"モデルが見つかりません: {model_path}"
    )

# =====================================
# 2. CSV列の定義
# =====================================

fieldnames = [
    "run_id",
    "timestamp",
    "scenario_id",
    "model_file",
    "prompt_type",
    "repeat_index",
    "prompt",
    "system_message",
    "correct_route",
    "generated_text",
    "elapsed_sec",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_sec",
    "finish_reason",
    "n_ctx",
    "n_gpu_layers",
    "temperature",
    "max_tokens",
    "repeat_penalty",
    "seed",
]

# =====================================
# 3. Qwenの読み込み
# =====================================

print("モデルを読み込んでいます...")

llm = Llama(
    model_path=str(model_path),
    **MODEL_CONFIG
)

print("モデルの読み込み成功！")


# =====================================
# 4. CSVファイルを準備
# =====================================

file_exists = (
    csv_path.exists()
    and csv_path.stat().st_size > 0
)

if file_exists:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])

    if existing_header != fieldnames:
        raise ValueError(
            "既存CSVの列構成が異なります。"
            "RESULTS_FILENAMEを変更してください。"
        )


# =====================================
# 5. 実験処理
# =====================================

with open(
    csv_path,
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
        f.flush()

    # シナリオごとに実行
    for scenario_file in SCENARIO_FILES:

        scenario_path = DATA_DIR / scenario_file

        with open(
            scenario_path,
            "r",
            encoding="utf-8"
        ) as sf:
            scenario = json.load(sf)

        scenario_id = scenario["scenario_id"]
        correct_route = scenario["correct_route"]

        # P1・P2・P3を作成
        prompts = build_prompts(scenario)

        print(f"\nシナリオ：{scenario_id}")

        # プロンプトごとに実行
        for prompt_type in PROMPT_TYPES:

            if prompt_type not in prompts:
                raise ValueError(
                    f"未定義のプロンプト: {prompt_type}"
                )

            prompt = prompts[prompt_type]

            # 繰り返し実験
            for repeat_index in range(1, REPEAT_COUNT + 1):

                print(
                    f"\n===== {prompt_type}"
                    f" / {repeat_index}回目 ====="
                )

                messages = [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                start = time.perf_counter()

                result = llm.create_chat_completion(
                    messages=messages,
                    **GENERATION_CONFIG
                )

                elapsed = time.perf_counter() - start

                choice = result["choices"][0]

                answer = choice["message"]["content"] or ""
                finish_reason = choice.get("finish_reason", "")

                usage = result.get("usage", {})

                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get(
                    "completion_tokens"
                )

                if completion_tokens is not None and elapsed > 0:
                    tokens_per_sec = round(
                        completion_tokens / elapsed, 2
                    )
                else:
                    tokens_per_sec = None

                # 結果表示
                print("\n【生成文章】")
                print(answer)

                print(f"\n処理時間：{elapsed:.2f}秒")
                print(f"生成トークン数：{completion_tokens}")
                print(f"生成速度：{tokens_per_sec} tokens/sec")
                print(f"終了理由：{finish_reason}")

                # CSV保存
                writer.writerow({
                    "run_id": uuid4().hex,
                    "timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "scenario_id": scenario_id,
                    "model_file": MODEL_FILENAME,
                    "prompt_type": prompt_type,
                    "repeat_index": repeat_index,
                    "prompt": prompt,
                    "system_message": SYSTEM_MESSAGE,
                    "correct_route": correct_route,
                    "generated_text": answer,
                    "elapsed_sec": round(elapsed, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_sec": tokens_per_sec,
                    "finish_reason": finish_reason,
                    "n_ctx": MODEL_CONFIG["n_ctx"],
                    "n_gpu_layers": MODEL_CONFIG["n_gpu_layers"],
                    **GENERATION_CONFIG,
                })

                # 実験ごとに保存を確定
                f.flush()

print("\nすべての実験が完了しました！")
print(f"結果保存先：{csv_path}")
