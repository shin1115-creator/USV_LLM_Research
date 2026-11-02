
"""既存のTheta*・Qwen実験を、プロンプトを固定したまま繰り返し実行する。

配置先：<プロジェクトフォルダ>/code/run_repeated_experiment.py
実行予定の確認：python run_repeated_experiment.py
実際の推論を実行：python run_repeated_experiment.py --execute
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


MODELS = [
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
]
LANGUAGES = {
    "ja": "test_theta_qwen.py",
    "en": "test_theta_qwen_en.py",
}


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description="既存のTheta*・Qwen実験を繰り返し実行します。")
    parser.add_argument("--runs", type=int, default=3, help="モデル・言語の組み合わせごとの試行回数（初期値：3）")
    parser.add_argument("--execute", action="store_true", help="指定すると推論を実行します。省略時は予定表示のみです。")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs には1以上の整数を指定してください")

    code_dir = Path(__file__).resolve().parent
    root_dir = code_dir.parent
    results_dir = root_dir / "results"
    jobs = [
        (model, lang, script, trial)
        for model in MODELS
        for lang, script in LANGUAGES.items()
        for trial in range(1, args.runs + 1)
    ]

    print(f"実験予定: {len(jobs)}回（モデル2 × 言語2 × 各{args.runs}回）")
    for index, (model, lang, script, trial) in enumerate(jobs, 1):
        print(f"{index:02d}. {model} / {lang} / 試行{trial} / {script}")
    if not args.execute:
        print("\nプレビューのみ：推論は行っていません。実行時は --execute を指定します。")
        return

    for model in MODELS:
        if not (root_dir / "models" / model).is_file():
            parser.error(f"モデルが見つかりません: {root_dir / 'models' / model}")
    for script in LANGUAGES.values():
        if not (code_dir / script).is_file():
            parser.error(f"実行用スクリプトが見つかりません: {code_dir / script}")

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "repeated_run_logs"
    log_dir.mkdir(exist_ok=True)
    batch_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = results_dir / f"theta_001_repeat_manifest_{batch_id}.json"
    manifest = {
        "experiment_id": "theta_AB_001_repeat_preview",
        "started_at": now_iso(),
        "runs_per_condition": args.runs,
        "model_names": MODELS,
        "languages": list(LANGUAGES),
        "jobs": [],
        "note": "既存の実験スクリプトとプロンプトは変更せずに使用する。"
                "各試行は新しいPythonプロセスで実行され、処理時間は各スクリプト側の計測方法に従う。",
    }

    # 各試行の別プロセスで、既存スクリプトの読み込み前にモデル名だけを切り替える。
    # プロンプトとその他の生成パラメータは変更しない。
    child_code = (
        "import sys, runpy, config; "
        "config.MODEL_FILENAME = sys.argv[1]; "
        "runpy.run_path(sys.argv[2], run_name='__main__')"
    )
    failures = 0
    for index, (model, lang, script, trial) in enumerate(jobs, 1):
        print(f"\n実行 {index}/{len(jobs)}: {model} / {lang} / {trial}回目", flush=True)
        before = set(results_dir.glob("theta_001_qwen*.json"))
        started = now_iso()
        proc = subprocess.run(
            [sys.executable, "-c", child_code, model, str(code_dir / script)],
            cwd=code_dir, capture_output=True, text=True, errors="replace", check=False,
        )
        after = set(results_dir.glob("theta_001_qwen*.json"))
        new_files = sorted(str(p.relative_to(root_dir)) for p in after - before)
        log_path = log_dir / f"{batch_id}_{index:02d}_{lang}.log"
        log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
        row = {
            "model": model, "language": lang, "trial": trial,
            "started_at": started, "finished_at": now_iso(),
            "exit_code": proc.returncode, "new_result_files": new_files,
            "log": str(log_path.relative_to(root_dir)),
        }
        manifest["jobs"].append(row)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if proc.returncode:
            failures += 1
            print(f"失敗: ログ {log_path}", flush=True)
        else:
            print(f"完了: 結果ファイル {new_files or '検出なし（ログを確認）'}", flush=True)

    print(f"\n終了：成功 {len(jobs) - failures} 件 / 失敗 {failures} 件")
    print(f"実行記録: {manifest_path}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
