
from llama_cpp import Llama
from pathlib import Path
import time
from datetime import datetime

# モデルファイルの場所
project_dir = Path(__file__).resolve().parent.parent
model_path = project_dir / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Qwenの読み込み
print("モデルを読み込んでいます...")

llm = Llama(
    model_path=str(model_path),
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)

print("モデルの読み込み成功！")

# テスト用の避航経路情報
prompt = """
あなたは小型USVの避航経路を説明するAIです。

【航海情報】
遭遇状況：横切り
相対距離：1.6 NM
TCPA：360秒
DCPA：0.08 NM

【経路情報】
正解経路：ルートA
ルートA：他船の船尾側を通過
最小離隔距離：830m

ルートB：他船の船首側を横切る
最小離隔距離：280m

【指示】
指定された正解経路Aが適切である理由を
入力された航海情報と経路情報に基づいて
100～150文字程度の日本語で説明してください。

条件：
・入力情報にない事実を追加しない
・数値を正確に使用する
・正解経路を変更しない
・同じ説明を繰り返さない
"""

# 文章生成
start = time.perf_counter()

result = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "日本語で簡潔に説明してください。"},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    temperature=0.2
)

elapsed = time.perf_counter() - start

# 結果表示
print("\n【Qwenの生成結果】")
print(result["choices"][0]["message"]["content"])
print(f"\n生成処理時間：{elapsed:.2f}秒")

# 結果保存先
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

# 実行日時をファイル名にする
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

# 保存ファイル
output_path = results_dir / f"qwen_result_{timestamp}.txt"

# 生成結果と実験条件を保存
with open(output_path, "w", encoding="utf-8") as f:
    f.write(f"実験日時：{timestamp}\n")
    f.write("モデル：Qwen2.5-0.5B-Instruct Q4_K_M\n")
    f.write("実行条件：n_ctx=2048, max_tokens=400, temperature=0.2\n")
    f.write(f"生成処理時間：{elapsed:.2f}秒\n\n")
    f.write("【入力プロンプト】\n")
    f.write(prompt)
    f.write("\n\n【生成文章】\n")
    f.write(result["choices"][0]["message"]["content"] or "")

print(f"\n実験結果を保存しました：{output_path}")
