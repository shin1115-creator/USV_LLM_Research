
import json
from pathlib import Path

# 研究フォルダのパス
project_dir = Path(__file__).resolve().parent.parent

# JSONファイルの場所
json_path = project_dir / "data" / "scenario_001.json"

# JSONデータの読み込み
with open(json_path, "r", encoding="utf-8") as f:
    scenario = json.load(f)

# 読み込んだ情報を表示
print("【シナリオID】")
print(scenario["scenario_id"])

print("\n【遭遇状況】")
print(scenario["encounter_type"])

print("\n【航海情報】")
print("相対距離:", scenario["navigation"]["relative_distance_nm"], "NM")
print("TCPA:", scenario["navigation"]["tcpa_sec"], "秒")
print("DCPA:", scenario["navigation"]["dcpa_nm"], "NM")

print("\n【避航経路】")
print("ルートA:", scenario["routes"]["A"]["description"])
print("ルートB:", scenario["routes"]["B"]["description"])

print("\n【正解経路】")
print(scenario["correct_route"])
