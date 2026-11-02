
from pathlib import Path

# =====================================
# 1. ディレクトリ設定
# =====================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

# =====================================
# 2. モデル設定
# =====================================

MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

MODEL_CONFIG = {
    "n_ctx": 2048,
    "n_gpu_layers": 0,
    "n_threads": None,
    "verbose": False,
}

# =====================================
# 3. 実験対象
# =====================================

SCENARIO_FILES = [
    "scenario_001.json",
]

PROMPT_TYPES = [
    "P1_basic",
    "P2_structured",
    "P3_constrained",
]

REPEAT_COUNT = 1

# =====================================
# 4. 文章生成パラメータ
# =====================================

GENERATION_CONFIG = {
    "max_tokens": 400,
    "temperature": 0.0,
    "repeat_penalty": 1.1,
    "seed": 42,
}

# =====================================
# 5. システムメッセージ
# =====================================

SYSTEM_MESSAGE = (
    "あなたは小型USVの避航経路説明AIです。"
    "入力情報に基づいて日本語で説明してください。"
    "入力にない事実を断定しないでください。"
)

# =====================================
# 6. CSV保存設定
# =====================================

RESULTS_FILENAME = "experiment_results_v2.csv"
