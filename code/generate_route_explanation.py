r"""保存済みの経路比較データから、Qwenで採用経路の説明文を生成する。

配置先: USV_LLM_Research/code/generate_route_explanation.py
実行例: python code\generate_route_explanation.py --objective distance_priority
プレビュー: python code\generate_route_explanation.py --objective distance_priority --preview

注意:
- 採用経路の決定・変更はこのスクリプトでは行わない。
- LLMには初期計画経路の評価値と採用済み経路を与え、採用理由の文章を生成させる。
- 結果は日時入りJSONに保存し、従来の結果ファイルを上書きしない。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path


SCENARIO_ID = "crossing_all130_u5"
DEFAULT_MODEL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
OBJECTIVES = ("distance_priority", "separation_priority")

# 初回の動作検証ではモデル・生成条件を固定する。今後の比較実験で変更する。
N_CTX = 2048
N_GPU_LAYERS = 0
N_THREADS = 4
MAX_TOKENS = 256
TEMPERATURE = 0.0


def project_root() -> Path:
    """このファイルをプロジェクト直下のcode/に配置した場合のルート。"""
    here = Path(__file__).resolve()
    if here.parent.name == "code":
        return here.parent.parent
    # テストのため単体で起動する場合は現在のディレクトリを使う。
    return Path.cwd()


def require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label}は数値である必要があります: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}が有限数ではありません")
    return result


def load_input(path: Path, objective: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LLM入力JSONが見つかりません: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("scenario_id") != SCENARIO_ID:
        raise ValueError(f"シナリオIDが想定と異なります: {data.get('scenario_id')!r}")
    if data.get("navigation_objective", {}).get("id") != objective:
        raise ValueError("入力JSONと指定した航行目的が一致しません")
    if data.get("selected_route") not in ("A", "B"):
        raise ValueError("採用経路がA/Bとして確定していません")
    routes = data.get("routes", {})
    for rid in ("A", "B"):
        if rid not in routes or not isinstance(routes[rid], dict):
            raise ValueError(f"経路{rid}のデータがありません")
        info = routes[rid]
        for field in ("route_length_m", "estimated_travel_time_s", "minimum_separation_m"):
            require_number(info.get(field), f"経路{rid}.{field}")
        if info.get("bypass_side") not in ("右舷側", "左舷側", "直進付近"):
            raise ValueError(f"経路{rid}の迂回方向を確認してください: {info.get('bypass_side')!r}")
    return data


def route_text(rid: str, data: dict) -> str:
    """計算済みの事実のみ列挙し、差分や説明文は事前に作らない。"""
    route = data["routes"][rid]
    return (
        f"経路{rid}:\n"
        f"  経路長: {route['route_length_m']:.2f} m\n"
        f"  推定航行時間: {route['estimated_travel_time_s']:.2f} 秒\n"
        f"  最小離隔距離: {route['minimum_separation_m']:.2f} m\n"
        f"  迂回方向: {route['bypass_side']}"
    )


def build_prompt(data: dict) -> str:
    objective = data["navigation_objective"]
    return (
        "以下は小型USVのTheta*初期計画経路の比較データです。\n"
        "経路Aと経路Bを比較した結果、採用する経路は既にPythonで決定されています。\n\n"
        f"【航行目的】{objective['label']}\n"
        f"【採用基準】{objective['rule']}\n"
        f"【採用済み経路】経路{data['selected_route']}\n\n"
        f"【経路情報】\n{route_text('A', data)}\n\n{route_text('B', data)}\n\n"
        "【指示】\n"
        "採用済み経路が、指定された航行目的に沿って選ばれた理由を日本語で2～3文で説明してください。\n"
        "- 採用経路を選び直さず、必ず既に指定された採用経路について述べる。\n"
        "- 航行目的に対応する指標を優先して、A/Bの実数値を正確に比較する。\n"
        "- 経路長と最小離隔距離を混同しない。\n"
        "- 迂回方向は自船初期針路から見た左右であり、他船の通過舷とは異なる。\n"
        "- 入力されていない安全基準、Bumper判定、COLREGs適合、危険性の断定を追加しない。\n"
        "- 説明文だけを出力し、箇条書きや指示文の繰り返しをしない。"
    )


def model_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / "models" / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="採用済みの経路についてQwenで説明を生成")
    parser.add_argument("--objective", choices=OBJECTIVES, default="distance_priority",
                        help="航行目的（初期値: 距離重視）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="modelsフォルダ内のGGUFファイル名（フルパスも可）")
    parser.add_argument("--preview", action="store_true", help="LLMを実行せずプロンプトだけ表示")
    parser.add_argument("--project-root", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    root = args.project_root.resolve() if args.project_root else project_root()
    input_path = root / "data" / "llm_inputs" / f"{SCENARIO_ID}_{args.objective}_llm_input.json"
    data = load_input(input_path, args.objective)
    prompt = build_prompt(data)
    print(f"シナリオ: {data['scenario_id']}")
    print(f"航行目的: {data['navigation_objective']['label']}")
    print(f"Pythonによる採用経路: {data['selected_route']}")
    print(f"入力ファイル: {input_path}")

    if args.preview:
        print("\n【LLMへの入力内容（プレビュー）】")
        print(prompt)
        print("\nプレビュー終了：LLM推論・結果保存は行っていません。")
        return

    gguf = model_path(root, args.model)
    if not gguf.is_file():
        raise FileNotFoundError(f"GGUFモデルが見つかりません: {gguf}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-pythonを現在の仮想環境で読み込めません") from exc

    print(f"モデル: {gguf.name}")
    print("モデルを読み込んでいます...")
    llm = Llama(model_path=str(gguf), n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS, n_threads=N_THREADS, verbose=False)

    start = time.perf_counter()
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "あなたはUSVの経路選択結果を、提供された客観的データだけで説明するAIです。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=1.0,
    )
    elapsed = time.perf_counter() - start

    choice = response["choices"][0]
    explanation = (choice.get("message", {}).get("content") or "").strip()
    if not explanation:
        print("警告：生成文が空です。結果ファイルにそのまま記録します。")
    print("\n【Qwenが生成した説明文】")
    print(explanation or "（空）")
    print(f"推論時間: {elapsed:.2f} 秒")
    print(f"終了理由: {choice.get('finish_reason', '不明')}")

    output_dir = root / "results" / "llm_explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_name = f"{SCENARIO_ID}_{args.objective}_{gguf.stem}_ja_{timestamp}.json"
    output_path = output_dir / output_name
    saved = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "scenario_id": data["scenario_id"],
        "objective_id": args.objective,
        "selected_route_by_python": data["selected_route"],
        "model_file": gguf.name,
        "prompt_version": "objective_baseline_ja_v1",
        "parameters": {
            "n_ctx": N_CTX, "n_gpu_layers": N_GPU_LAYERS, "n_threads": N_THREADS,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "top_p": 1.0,
        },
        "input_file": str(input_path.relative_to(root)),
        "prompt": prompt,
        "generated_explanation": explanation,
        "inference_seconds": round(elapsed, 4),
        "finish_reason": choice.get("finish_reason"),
        "token_usage": response.get("usage", {}),
        "evaluation_status": "未評価",
    }
    # 'x'モードによって過去の結果を上書きしない。
    with output_path.open("x", encoding="utf-8") as f:
        json.dump(saved, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"結果保存: {output_path}")
    print("説明の正確性は未評価です。採用結果との整合を次工程で確認します。")


if __name__ == "__main__":
    main()
