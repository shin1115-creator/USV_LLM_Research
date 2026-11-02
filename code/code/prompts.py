# =====================================
# 指示文
# =====================================

def build_prompts(scenario):

    nav = scenario["navigation"]
    routes = scenario["routes"]
    correct = scenario["correct_route"]

    # 共通の航海情報
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

    common_rules = """
【共通条件】
・正解経路を変更しないでください。
・入力情報にない事実を断定しないでください。
・根拠が不足する場合は明示してください。
・日本語で簡潔に説明してください。
"""

    # =================================
    # P1：基本型
    # =================================

    prompt_p1 = f"""
あなたは小型USVの避航経路を説明するAIです。

{input_data}

【指示】
指定された正解経路が適切である理由を、
航海情報と両経路の特徴に基づいて説明してください。

{common_rules}
"""

    # =================================
    # P2：構造化型
    # =================================

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

    # =================================
    # P3：制約付き型
    # =================================

    prompt_p3 = f"""
あなたは小型USVの避航経路の説明担当です。

{input_data}

次の条件に従い、指定された正解経路について説明文を作成してください。

【説明する内容】
・TCPAとDCPAから分かる接近状況
・ルートAとBの最小離隔距離の違い
・指定された正解経路の特徴と、その根拠

【重要な制約】
・説明文のみを出力する
・指示内容をそのまま書き写さない
・最小離隔距離と相対距離を混同しない
・入力されていない安全基準を作らない
・根拠が不足する場合は不足していると述べる
・150文字程度の日本語で記述する

【説明文】
"""
    return {
        "P1_basic": prompt_p1,
        "P2_structured": prompt_p2,
        "P3_constrained": prompt_p3,
    }
