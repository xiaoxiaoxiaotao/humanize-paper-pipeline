import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional


def analyze_chinese_text(text: str, use_advanced: bool = True) -> Tuple[int, Optional[Dict]]:
    """
    增强版中文AI检测器 (2026更新版 - 知网3.0算法适配)
    
    优化内容:
    1. 扩充AI常用词库 (来自知网、万方、维普最新检测研究)
    2. 轻量化技术特征检测 (适合2核CPU环境)
    3. 增加更多中文AI特征模式
    4. 引入最新检测指标: 困惑度代理、熵值、词突发性
    5. 新增知网3.0核心维度: 句长分布、段落结构、信息密度、连接词分布
    """
    details = {'metrics': {}}
    score = 0

    sentences = [s.strip() for s in re.split(r'[。！？!?]+', text) if s.strip()]
    if not sentences:
        return 0, None

    text_len = len(text)
    if text_len == 0:
        return 0, None

    # 知网3.0核心维度 - 优先检测
    score, details = _analyze_sentence_length_distribution(text, sentences, score, details)
    score, details = _analyze_paragraph_structure_similarity(text, sentences, score, details)
    score, details = _analyze_info_density_distribution(text, sentences, score, details)
    score, details = _analyze_transition_word_distribution(text, sentences, score, details)
    
    # 原有检测维度
    score, details = _analyze_sentence_uniformity(text, sentences, score, details)
    score, details = _analyze_transitions(text, score, details)
    score, details = _analyze_abstract_language(text, score, details)
    score, details = _analyze_hedging(text, score, details)
    score, details = _analyze_de_density(text, text_len, score, details)
    score, details = _analyze_sentence_openings(sentences, score, details)
    score, details = _analyze_idioms(text, score, details)
    score, details = _analyze_symmetry(text, score, details)
    score, details = _analyze_punctuation(text, score, details)
    score, details = _analyze_templates(text, sentences, score, details)
    
    if use_advanced:
        score, details = _analyze_entropy(text, score, details)
        score, details = _analyze_bigram_ttr(text, score, details)
        score, details = _analyze_clause_density(sentences, text, score, details)
        score, details = _analyze_de_chains(text, score, details)
        score, details = _analyze_word_burstiness(text, score, details)
    
    score, details = _analyze_concluding_formula(text, score, details)
    score, details = _analyze_passive_zh(text, text_len, score, details)
    score, details = _analyze_lexical_repetition(text, score, details)
    score, details = _analyze_citation_distribution(sentences, text, score, details)

    final_score = min(100, max(0, int(score)))
    details['overall_score'] = final_score
    return final_score, details


def _analyze_sentence_uniformity(text, sentences, score, details):
    """句子长度均匀性分析 - 强AI特征"""
    lengths = [len(s) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    variance_ratio = std_dev / avg_length if avg_length > 0 else 0

    uniformity_score = 0
    if avg_length < 15:
        if variance_ratio < 0.18:
            uniformity_score = 25
            details['metrics']['sentence_uniformity'] = {"score": 0.95, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极规律)"}
        elif variance_ratio < 0.25:
            uniformity_score = 15
            details['metrics']['sentence_uniformity'] = {"score": 0.8, "variance_ratio": round(variance_ratio, 3), "details": "高"}
        elif variance_ratio < 0.35:
            uniformity_score = 8
            details['metrics']['sentence_uniformity'] = {"score": 0.5, "variance_ratio": round(variance_ratio, 3), "details": "中等"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.1, "variance_ratio": round(variance_ratio, 3), "details": "低"}
    else:
        if variance_ratio < 0.22:
            uniformity_score = 30
            details['metrics']['sentence_uniformity'] = {"score": 0.95, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极规律)"}
        elif variance_ratio < 0.32:
            uniformity_score = 18
            details['metrics']['sentence_uniformity'] = {"score": 0.75, "variance_ratio": round(variance_ratio, 3), "details": "高"}
        elif variance_ratio < 0.42:
            uniformity_score = 10
            details['metrics']['sentence_uniformity'] = {"score": 0.5, "variance_ratio": round(variance_ratio, 3), "details": "中等"}
        elif variance_ratio < 0.55:
            uniformity_score = 5
            details['metrics']['sentence_uniformity'] = {"score": 0.25, "variance_ratio": round(variance_ratio, 3), "details": "低偏中"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.05, "variance_ratio": round(variance_ratio, 3), "details": "低"}
    score += uniformity_score
    return score, details


AI_TRANSITIONS_ZH = [
    "综上所述", "总而言之", "总体来看", "总体而言", "整体来看", "整体而言",
    "由此可见", "综上", "概而言之", "简而言之",
    "此外", "另外", "再者", "而且", "不仅如此", "与此同时", "在此基础上",
    "毋庸置疑", "值得注意的是", "值得一提的是", "需要强调的是", "必须指出",
    "不可否认", "显而易见", "不难看出", "可想而知", "不言而喻",
    "首先", "其次", "再次", "最后", "第一", "第二", "第三",
    "一方面", "另一方面", "一方面...另一方面...",
    "进一步而言", "具体来说", "一般来说", "通常来说", "从...来看",
    "事实上", "实际上", "本质上", "归根结底", "从根本上说",
    "因此", "然而", "但是", "不过", "可是", "尽管如此",
    "例如", "比如", "举例来说", "譬如说",
    "重要的是", "关键在于", "核心在于",
    "从宏观角度", "从微观角度", "从整体视角",
    "随着...的发展", "随着...的进步", "随着...的深入",
    "在当前背景下", "在...的背景下", "在...的趋势下",
    "进入...时代", "步入...阶段"
]


def _analyze_transitions(text, score, details):
    """过渡词过度使用检测"""
    transition_found = []
    transition_count = 0
    for p in AI_TRANSITIONS_ZH:
        cnt = text.count(p)
        if cnt > 0:
            transition_count += cnt
            transition_found.append((p, cnt))
    details['metrics']['transition_overuse'] = {
        "count": transition_count,
        "found": transition_count,
        "items": transition_found[:12],
        "details": f"检测到 {transition_count} 个过渡词"
    }
    score += min(transition_count * 6, 28)
    return score, details


ABSTRACT_ZH = [
    "多种因素", "各个方面", "深远的影响", "发挥着至关重要的作用",
    "扮演着重要的角色", "不可忽视", "具有重要意义",
    "毫无疑问", "深刻揭示了", "综合运用",
    "提供了理论支撑", "为后续研究提供基础", "完善了理论体系",
    "开启了新篇章", "此案例印证了", "深入探讨",
    "具有一定的局限性", "全面阐述", "系统梳理", "深入分析",
    "这不难理解", "随着社会的不断发展", "在当前背景下",
    "有着广泛的应用", "为...指明了方向",
    "良好平衡", "某种平衡", "取得了平衡",
    "强有力的", "强有力的支撑", "坚实的基础",
    "深远意义", "重要价值", "现实意义", "理论意义",
    "取得了良好的效果", "取得了显著的效果",
    "表现出色", "表现优异", "效果显著",
    "具有较强的", "具有较高的", "具有良好的",
    "发挥了重要作用", "起到了关键作用",
    "从...角度来看", "在...方面", "从...层面来说",
    "不仅...而且...", "既...又...", "一方面...另一方面...",
    "为...做出了贡献", "推动了...的发展", "促进了...的进步",
    "智能化", "自动化", "数字化", "信息化", "智慧化",
    "技术方案", "核心痛点", "关键问题",
    "高效", "优异", "显著", "卓越", "出色",
    "具有里程碑意义", "具有划时代的意义",
    "在...中发挥重要作用", "对...具有重要价值",
    "极大地", "显著地", "有效地", "大大地",
    "提供了...的思路", "提供了...的方法", "提供了...的技术方案",
    "为...奠定了基础", "为...创造了条件", "为...提供了保障",
    "具有...的特点", "具有...的优势", "具有...的特性",
    "从某种意义上说", "在一定程度上", "在某些情况下",
    "可以说", "可以认为", "可以推断",
    "综上所述", "总而言之", "由此可见",
    "因此", "由此", "因而"
]


def _analyze_abstract_language(text, score, details):
    """空泛套话和大词检测"""
    abstract_found = []
    abstract_count = 0
    for p in ABSTRACT_ZH:
        cnt = text.count(p)
        if cnt > 0:
            abstract_count += cnt
            abstract_found.append((p, cnt))
    details['metrics']['abstract_language'] = {
        "count": abstract_count,
        "found": abstract_count,
        "items": abstract_found[:15],
        "details": f"检测到 {abstract_count} 个空泛套话/大词"
    }
    score += min(abstract_count * 7, 32)
    return score, details


AI_HEDGING_ZH = [
    "似乎", "似乎在", "似乎表明", "似乎说明", "似乎验证",
    "可能表明", "可能说明", "可能意味着", "或许", "或许可以", "或可", "或可为",
    "潜在地", "倾向于", "有可能", "不排除", "不能完全排除",
    "可以认为", "可以推测", "可以推断", "可以看出",
    "初步表明", "初步显示", "初步验证",
    "似乎在某种程度上", "可能在一定程度上", "似乎在一定意义上",
    "从某种角度看", "在一定意义上", "从某种意义上说",
    "某种程度上", "一定程度上", "或多或少",
    "相对来说", "比较而言", "一般来说",
    "可能存在", "可能有", "或许有"
]


def _analyze_hedging(text, score, details):
    """对冲词过度使用检测 - 知网重点检测特征"""
    hedge_found = []
    hedge_count = 0
    for p in AI_HEDGING_ZH:
        cnt = text.count(p)
        if cnt > 0:
            hedge_count += cnt
            hedge_found.append((p, cnt))

    if hedge_count >= 6:
        over_hedge_penalty = min((hedge_count - 5) * 7, 28)
        score += over_hedge_penalty
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:12],
            "details": f"过度对冲词 ({hedge_count}个), 惩罚 +{over_hedge_penalty}"
        }
    elif hedge_count >= 4:
        over_hedge_penalty = min((hedge_count - 3) * 4, 12)
        score += over_hedge_penalty
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:12],
            "details": f"对冲词较多 ({hedge_count}个), 轻微惩罚 +{over_hedge_penalty}"
        }
    elif hedge_count >= 1:
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:12],
            "details": f"存在少量对冲词 ({hedge_count}个)"
        }
    else:
        details['metrics']['over_hedging'] = {"count": 0, "items": [], "details": "未检测到对冲词"}

    genuine_hedging = ["有待进一步", "尚需验证", "仍需探讨", "需进一步研究", "有待考证"]
    genuine_count = sum(text.count(p) for p in genuine_hedging)
    if genuine_count > 0 and hedge_count <= 2:
        score -= min(genuine_count * 4, 8)
        details['metrics']['scholarly_hedging'] = {"count": genuine_count, "details": f"检测到{genuine_count}个真实学术对冲词"}

    return score, details


def _analyze_de_density(text, text_len, score, details):
    """'的'字密度检测 - 中文AI写作重要特征"""
    de_count = text.count("的")
    shi_count = text.count("是")
    if de_count / text_len > 0.065:
        penalty = min(int((de_count / text_len - 0.065) * 450), 22)
        score += penalty
        details['metrics']['dense_adj'] = {"density": round(de_count / text_len, 4), "details": f"'的'字密度过高, 惩罚 +{penalty}"}
    elif de_count / text_len > 0.055:
        penalty = min(int((de_count / text_len - 0.055) * 300), 10)
        score += penalty
        details['metrics']['dense_adj'] = {"density": round(de_count / text_len, 4), "details": f"'的'字密度偏高, 轻微惩罚 +{penalty}"}
    
    if shi_count / text_len > 0.028:
        penalty = min(int((shi_count / text_len - 0.028) * 400), 18)
        score += penalty
        details['metrics']['dense_be'] = {"density": round(shi_count / text_len, 4), "details": f"'是'字密度过高, 惩罚 +{penalty}"}
    return score, details


def _analyze_sentence_openings(sentences, score, details):
    """句首模式重复检测"""
    sentence_starts = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 4:
            sentence_starts.append(s_clean[:4])

    if len(sentence_starts) >= 3:
        start_counter = Counter(sentence_starts)
        most_common_start = start_counter.most_common(1)[0]
        if most_common_start[1] >= 4:
            repetition_ratio = most_common_start[1] / len(sentence_starts)
            if repetition_ratio > 0.45:
                penalty = 18
            elif repetition_ratio > 0.3:
                penalty = 12
            else:
                penalty = 8
            score += penalty
            details['metrics']['sentence_opening_repetition'] = {
                "top_pattern": most_common_start[0],
                "count": most_common_start[1],
                "ratio": round(repetition_ratio, 3),
                "details": f"句首模式重复严重, 惩罚 +{penalty}"
            }
        elif most_common_start[1] >= 3:
            repetition_ratio = most_common_start[1] / len(sentence_starts)
            penalty = 8
            score += penalty
            details['metrics']['sentence_opening_repetition'] = {
                "top_pattern": most_common_start[0],
                "count": most_common_start[1],
                "ratio": round(repetition_ratio, 3),
                "details": f"句首模式重复, 轻微惩罚 +{penalty}"
            }
    return score, details


COMMON_AI_IDIOMS = [
    "不可或缺", "举足轻重", "至关重要", "显而易见", "毋庸置疑",
    "与日俱增", "日新月异", "蓬勃发展", "方兴未艾", "如火如荼",
    "层出不穷", "琳琅满目", "丰富多彩", "千变万化", "错综复杂",
    "相辅相成", "密不可分", "息息相关", "休戚与共", "一脉相承",
    "卓有成效", "行之有效", "有的放矢", "对症下药", "因地制宜",
    "循序渐进", "稳扎稳打", "精益求精", "孜孜不倦", "锲而不舍",
    "前所未有", "史无前例", "开创性", "突破性", "里程碑式",
    "不断深入", "持续推进", "逐步完善", "日益完善",
    "全面提升", "显著提高", "大幅改善", "有效解决",
    "积极探索", "不断创新", "勇于实践", "锐意进取"
]


def _analyze_idioms(text, score, details):
    """成语堆砌检测 - AI写作典型特征"""
    idiom_count = 0
    idiom_found = []
    for idiom in COMMON_AI_IDIOMS:
        cnt = text.count(idiom)
        if cnt > 0:
            idiom_count += cnt
            idiom_found.append((idiom, cnt))

    if idiom_count >= 5:
        penalty = min((idiom_count - 4) * 6, 20)
        score += penalty
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:12],
            "details": f"成语/四字词组严重堆砌 ({idiom_count}个), 惩罚 +{penalty}"
        }
    elif idiom_count >= 3:
        penalty = min((idiom_count - 2) * 4, 12)
        score += penalty
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:12],
            "details": f"成语/四字词组较多 ({idiom_count}个), 惩罚 +{penalty}"
        }
    elif idiom_count >= 1:
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:12],
            "details": f"成语使用 {idiom_count}个"
        }
    return score, details


SYMMETRY_PATTERNS = [
    r'不仅[^，。]{2,15}，而且[^，。]{2,15}',
    r'既[^，。]{2,10}，又[^，。]{2,10}',
    r'一方面[^，。]{2,20}，另一方面[^，。]{2,20}',
    r'既[^，。]{2,10}也[^，。]{2,10}',
    r'一方面...另一方面...',
    r'不仅...而且...',
    r'既...又...',
    r'一边...一边...',
    r'有的...有的...',
    r'有时...有时...'
]


def _analyze_symmetry(text, score, details):
    """对称句式检测"""
    symmetry_count = 0
    for pat in SYMMETRY_PATTERNS:
        symmetry_count += len(re.findall(pat, text))

    if symmetry_count >= 4:
        penalty = min(symmetry_count * 5, 18)
        score += penalty
        details['metrics']['structural_symmetry'] = {
            "count": symmetry_count,
            "details": f"句式对称/排比严重 ({symmetry_count}处), 惩罚 +{penalty}"
        }
    elif symmetry_count >= 2:
        penalty = min(symmetry_count * 4, 12)
        score += penalty
        details['metrics']['structural_symmetry'] = {
            "count": symmetry_count,
            "details": f"句式对称/排比较多 ({symmetry_count}处), 轻微惩罚 +{penalty}"
        }
    return score, details


def _analyze_punctuation(text, score, details):
    """标点符号分布检测"""
    comma_count = text.count('，')
    period_count = text.count('。')
    semicolon_count = text.count('；')

    if period_count > 0:
        comma_period_ratio = comma_count / period_count
        if comma_period_ratio > 4.0:
            penalty = min(int((comma_period_ratio - 4.0) * 6), 18)
            score += penalty
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比过高, 惩罚 +{penalty}"
            }
        elif comma_period_ratio > 3.2:
            penalty = min(int((comma_period_ratio - 3.2) * 4), 10)
            score += penalty
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比偏高, 轻微惩罚 +{penalty}"
            }
        else:
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比正常"
            }
    return score, details


SUIZHE_PATTERNS = [
    r'随着[^，。]{2,25}的[^，。]{2,25}',
    r'基于[^，。]{2,25}的[^，。]{2,25}',
    r'通过[^，。]{2,25}的[^，。]{2,25}',
    r'利用[^，。]{2,25}的[^，。]{2,25}',
    r'根据[^，。]{2,25}的[^，。]{2,25}',
    r'针对[^，。]{2,25}的[^，。]{2,25}',
    r'对于[^，。]{2,25}的[^，。]{2,25}',
    r'伴随[^，。]{2,25}的[^，。]{2,25}',
    r'在[^，。]{2,20}的[^，。]{2,20}',
    r'从[^，。]{2,20}的[^，。]{2,20}'
]


def _analyze_templates(text, sentences, score, details):
    """模板句式检测 - 知网核心检测特征"""
    suizhe_count = 0
    suizhe_found = []
    for pat in SUIZHE_PATTERNS:
        matches = re.findall(pat, text)
        suizhe_count += len(matches)
        suizhe_found.extend(matches)

    if suizhe_count >= 3:
        penalty = min(suizhe_count * 7, 25)
        score += penalty
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:6],
            "details": f"'随着/基于/通过...的...'模板 ({suizhe_count}处), 惩罚 +{penalty}"
        }
    elif suizhe_count >= 2:
        penalty = min(suizhe_count * 5, 15)
        score += penalty
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:6],
            "details": f"'随着/基于...的...'模板 ({suizhe_count}处), 轻微惩罚 +{penalty}"
        }
    elif suizhe_count >= 1:
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:6],
            "details": f"'随着/基于...的...'模板 ({suizhe_count}处)"
        }

    paragraph_structure_markers = 0
    bg_markers = ["是...的重要", "在...中发挥", "作为...的", "近年来", "随着", "在当前背景下"]
    for m in bg_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    problem_markers = ["然而", "但是", "不足", "局限", "问题", "挑战", "困难"]
    for m in problem_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    value_markers = ["具有重要意义", "具有重要价值", "现实意义", "理论意义", "应用价值"]
    for m in value_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    paper_markers = ["本文", "本研究", "本文旨在", "本文拟", "本文将", "本文提出"]
    for m in paper_markers:
        if m in text:
            paragraph_structure_markers += 1
            break

    if paragraph_structure_markers >= 4:
        penalty = (paragraph_structure_markers - 3) * 8
        score += penalty
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"模板结构严重 (检测到 {paragraph_structure_markers}/4标记), 惩罚 +{penalty}"
        }
    elif paragraph_structure_markers >= 3:
        penalty = (paragraph_structure_markers - 2) * 6
        score += penalty
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"模板结构较多 (检测到 {paragraph_structure_markers}/4标记), 轻微惩罚 +{penalty}"
        }
    else:
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"段落结构自然 ({paragraph_structure_markers}/4标记)"
        }

    definition_pattern = r'[^，。]{3,25}是[^，。]{3,35}的[^，。]{2,20}'
    definition_matches = re.findall(definition_pattern, text)
    if len(definition_matches) >= 3:
        penalty = min(len(definition_matches) * 5, 18)
        score += penalty
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:4],
            "details": f"'是...的'定义句 ({len(definition_matches)}处), 惩罚 +{penalty}"
        }
    elif len(definition_matches) >= 2:
        penalty = min(len(definition_matches) * 4, 12)
        score += penalty
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:4],
            "details": f"'是...的'定义句 ({len(definition_matches)}处), 轻微惩罚 +{penalty}"
        }
    elif len(definition_matches) >= 1:
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:4],
            "details": f"'是...的'定义句 ({len(definition_matches)}处)"
        }

    return score, details


def _analyze_entropy(text, score, details):
    """信息熵分析 - 轻量级困惑度代理指标"""
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；\"\"''《》()（）【】· \n\t"]
    if len(chars_only) > 1:
        freqs = Counter(chars_only)
        entropy = -sum((cnt / len(chars_only)) * math.log2(cnt / len(chars_only)) for cnt in freqs.values())
        details['metrics']['shannon_entropy'] = {
            "value": round(entropy, 2),
            "details": f"信息熵: {round(entropy, 2)}"
        }
        if entropy < 3.8:
            score += 12
            details['metrics']['shannon_entropy']['details'] += " [极低, 惩罚 +12]"
        elif entropy < 4.2:
            score += 8
            details['metrics']['shannon_entropy']['details'] += " [偏低, 惩罚 +8]"
        elif entropy < 4.6:
            score += 4
            details['metrics']['shannon_entropy']['details'] += " [略低, 轻微惩罚 +4]"
    return score, details


def _analyze_bigram_ttr(text, score, details):
    """二元词类型-词例比 - 词汇多样性检测"""
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；\"\"''《》()（）【】· \n\t"]
    if len(chars_only) > 10:
        bigrams = ["".join(chars_only[i:i + 2]) for i in range(len(chars_only) - 1)]
        ttr = len(set(bigrams)) / len(bigrams) if bigrams else 1
        details['metrics']['bigram_ttr'] = {
            "value": round(ttr, 3),
            "details": f"2-gram TTR: {round(ttr, 3)}"
        }
        if ttr < 0.52:
            score += 15
            details['metrics']['bigram_ttr']['details'] += " [词汇贫乏, 惩罚 +15]"
        elif ttr < 0.58:
            score += 10
            details['metrics']['bigram_ttr']['details'] += " [词汇偏单调, 惩罚 +10]"
        elif ttr < 0.65:
            score += 5
            details['metrics']['bigram_ttr']['details'] += " [词汇略单调, 轻微惩罚 +5]"
    return score, details


def _analyze_clause_density(sentences, text, score, details):
    """从句嵌套密度 - AI写作典型特征"""
    clauses = [c for c in re.split(r'[，。！？；：]+', text) if c.strip()]
    if sentences:
        clauses_per_sentence = len(clauses) / len(sentences)
        details['metrics']['clause_chain_density'] = {
            "value": round(clauses_per_sentence, 2),
            "details": f"从句密度: {round(clauses_per_sentence, 2)}"
        }
        if clauses_per_sentence > 4.5:
            score += 14
            details['metrics']['clause_chain_density']['details'] += " [严重嵌套, 惩罚 +14]"
        elif clauses_per_sentence > 3.8:
            score += 10
            details['metrics']['clause_chain_density']['details'] += " [中度嵌套, 惩罚 +10]"
        elif clauses_per_sentence > 3.2:
            score += 6
            details['metrics']['clause_chain_density']['details'] += " [偏多嵌套, 轻微惩罚 +6]"
    return score, details


def _analyze_de_chains(text, score, details):
    """'的'字链检测"""
    de_chain_pattern = r'的[^的]{0,4}的[^的]{0,4}的'
    de_chains = re.findall(de_chain_pattern, text)
    if len(de_chains) >= 3:
        penalty = min(len(de_chains) * 6, 20)
        score += penalty
        details['metrics']['de_chain'] = {
            "count": len(de_chains),
            "examples": de_chains[:4],
            "details": f"'的'字链严重 ({len(de_chains)}处), 惩罚 +{penalty}"
        }
    elif len(de_chains) >= 2:
        penalty = min(len(de_chains) * 4, 12)
        score += penalty
        details['metrics']['de_chain'] = {
            "count": len(de_chains),
            "examples": de_chains[:4],
            "details": f"'的'字链较多 ({len(de_chains)}处), 轻微惩罚 +{penalty}"
        }
    return score, details


CONCLUDING_PATTERNS = [
    "总体来看", "总而言之", "综上所述", "总的来说",
    "由此可见", "综上", "概而言之", "简而言之",
    "本文将重点研究", "本文旨在", "本文拟",
    "本文通过", "本文基于", "本文提出",
    "具有重要的现实意义", "具有重要的理论意义"
]


def _analyze_concluding_formula(text, score, details):
    """总结式套话检测"""
    concluding_count = 0
    concluding_found = []
    for p in CONCLUDING_PATTERNS:
        cnt = text.count(p)
        if cnt > 0:
            concluding_count += cnt
            concluding_found.append((p, cnt))

    if concluding_count >= 3:
        penalty = min((concluding_count - 2) * 6, 20)
        score += penalty
        details['metrics']['concluding_formula'] = {
            "count": concluding_count,
            "items": concluding_found[:12],
            "details": f"总结套话严重 ({concluding_count}处), 惩罚 +{penalty}"
        }
    elif concluding_count >= 2:
        penalty = min((concluding_count - 1) * 5, 14)
        score += penalty
        details['metrics']['concluding_formula'] = {
            "count": concluding_count,
            "items": concluding_found[:12],
            "details": f"总结套话较多 ({concluding_count}处), 惩罚 +{penalty}"
        }
    elif concluding_count >= 1:
        details['metrics']['concluding_formula'] = {
            "count": concluding_count,
            "items": concluding_found[:12],
            "details": f"总结套话 ({concluding_count}处)"
        }
    return score, details


PASSIVE_ZH = ["被", "受到", "得到", "得以", "予以", "加以", "由...", "为...所..."]


def _analyze_passive_zh(text, text_len, score, details):
    """被动语态过度使用检测"""
    passive_count = sum(text.count(p) for p in PASSIVE_ZH[:6])
    passive_count += text.count("由...所...") + text.count("为...所...")
    if passive_count >= 5 and text_len > 0:
        passive_density = passive_count / text_len
        if passive_density > 0.022:
            penalty = 10
            score += penalty
            details['metrics']['passive_overuse'] = {
                "count": passive_count,
                "density": round(passive_density, 4),
                "details": f"被动语态过度 ({passive_count}个), 惩罚 +{penalty}"
            }
    return score, details


def _analyze_lexical_repetition(text, score, details):
    """词汇重复检测"""
    content_chars_segments = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    if content_chars_segments:
        all_bigrams_text = []
        for seg in content_chars_segments:
            for i in range(len(seg) - 1):
                all_bigrams_text.append(seg[i:i + 2])
        if all_bigrams_text:
            bigram_counter = Counter(all_bigrams_text)
            top_bigram = bigram_counter.most_common(1)[0]
            if top_bigram[1] >= 5 and len(all_bigrams_text) > 0:
                repetition_ratio = top_bigram[1] / len(all_bigrams_text)
                if repetition_ratio > 0.10:
                    penalty = 12
                    score += penalty
                    details['metrics']['lexical_repetition'] = {
                        "top_bigram": top_bigram[0],
                        "count": top_bigram[1],
                        "ratio": round(repetition_ratio, 3),
                        "details": f"高频词 '{top_bigram[0]}' 严重重复({top_bigram[1]}次), 惩罚 +{penalty}"
                    }
                elif repetition_ratio > 0.07:
                    penalty = 8
                    score += penalty
                    details['metrics']['lexical_repetition'] = {
                        "top_bigram": top_bigram[0],
                        "count": top_bigram[1],
                        "ratio": round(repetition_ratio, 3),
                        "details": f"高频词 '{top_bigram[0]}' 重复较多({top_bigram[1]}次), 惩罚 +{penalty}"
                    }
    return score, details


def _analyze_citation_distribution(sentences, text, score, details):
    """引用分布检测 - AI文本常把所有引用放在句末"""
    citation_pattern = r'\[\d+(?:-\d+)?\]'
    citations = re.findall(citation_pattern, text)
    if len(citations) >= 3:
        sentences_with_citations = 0
        sentence_end_citations = 0
        for s in sentences:
            if re.search(citation_pattern, s):
                sentences_with_citations += 1
                if re.search(citation_pattern + r'[^\u4e00-\u9fa5a-zA-Z]{0,3}$', s):
                    sentence_end_citations += 1
        if sentences_with_citations > 0 and sentence_end_citations / sentences_with_citations > 0.85:
            penalty = 12
            score += penalty
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2),
                "details": f"引用过度集中句末 ({sentence_end_citations}/{sentences_with_citations}), 惩罚 +{penalty}"
            }
        elif sentences_with_citations > 0 and sentence_end_citations / sentences_with_citations > 0.7:
            penalty = 8
            score += penalty
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2),
                "details": f"引用偏集中句末 ({sentence_end_citations}/{sentences_with_citations}), 轻微惩罚 +{penalty}"
            }
        else:
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2) if sentences_with_citations > 0 else 0,
                "details": f"引用分布正常"
            }
    return score, details


def _analyze_info_density(sentences, score, details):
    """信息密度均匀性检测 - AI文本通常信息密度分布非常均匀"""
    if len(sentences) >= 3:
        info_densities = []
        for s in sentences:
            content_chars = len(re.findall(r'[\u4e00-\u9fa5]', s))
            total_chars = len(s)
            if total_chars > 0:
                info_densities.append(content_chars / total_chars)
        if len(info_densities) >= 3:
            mean_density = sum(info_densities) / len(info_densities)
            variance = sum((d - mean_density) ** 2 for d in info_densities) / len(info_densities)
            std = variance ** 0.5
            if std < 0.04:
                penalty = 14
                score += penalty
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度极均匀, 惩罚 +{penalty}"
                }
            elif std < 0.06:
                penalty = 10
                score += penalty
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度很均匀, 惩罚 +{penalty}"
                }
            elif std < 0.08:
                penalty = 6
                score += penalty
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度偏均匀, 轻微惩罚 +{penalty}"
                }
            else:
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度有变化"
                }
    return score, details


def _analyze_word_burstiness(text, score, details):
    """词突发性(Burstiness) - 人类文本会有某些词突发性频繁出现"""
    chars_segments = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    if len(chars_segments) > 5:
        words = []
        for seg in chars_segments:
            for i in range(len(seg) - 1):
                words.append(seg[i:i+2])
        
        if len(words) > 10:
            counter = Counter(words)
            counts = list(counter.values())
            
            if len(counts) > 1:
                import statistics
                mean_count = statistics.mean(counts)
                std_count = statistics.stdev(counts) if len(counts) > 1 else 0
                
                if mean_count > 0:
                    cv = std_count / mean_count
                    details['metrics']['word_burstiness'] = {
                        "cv": round(cv, 3),
                        "details": f"词突发性 CV: {round(cv, 3)}"
                    }
                    
                    if cv < 0.6:
                        score += 10
                        details['metrics']['word_burstiness']['details'] += " [突发性过低, 惩罚 +10]"
                    elif cv < 0.8:
                        score += 6
                        details['metrics']['word_burstiness']['details'] += " [突发性偏低, 轻微惩罚 +6]"
    return score, details


def _analyze_sentence_length_distribution(text, sentences, score, details):
    """知网3.0 - 维度1: 句长分布分析 (最强判别信号)"""
    if len(sentences) < 3:
        return score, details
    
    import statistics
    
    # 计算每句话的长度（不含标点）
    lengths = []
    for sent in sentences:
        clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
        if clean_sent:
            lengths.append(len(clean_sent))
    
    if len(lengths) < 3:
        return score, details
    
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
    
    # AI特征A: 句长集中在15-25字区间
    ai_peak_count = sum(1 for l in lengths if 15 <= l <= 25)
    ai_peak_ratio = ai_peak_count / len(lengths)
    
    # AI特征B: 标准差过小
    cv = std_len / mean_len if mean_len > 0 else 0
    
    details['metrics']['sentence_length_distribution'] = {
        'mean': round(mean_len, 1),
        'std': round(std_len, 1),
        'cv': round(cv, 3),
        'ai_peak_ratio': round(ai_peak_ratio, 2),
        'details': f'句长均值{round(mean_len, 1)}，变异系数{round(cv, 3)}，15-25字占比{round(ai_peak_ratio*100)}%'
    }
    
    # 综合评分
    penalty = 0
    if ai_peak_ratio > 0.7:
        penalty += 15
        details['metrics']['sentence_length_distribution']['details'] += ' [句长过于集中，强烈惩罚 +15]'
    elif ai_peak_ratio > 0.55:
        penalty += 8
        details['metrics']['sentence_length_distribution']['details'] += ' [句长较集中，惩罚 +8]'
    
    if cv < 0.25:
        penalty += 12
        details['metrics']['sentence_length_distribution']['details'] += ' [变异系数过低，惩罚 +12]'
    elif cv < 0.35:
        penalty += 6
        details['metrics']['sentence_length_distribution']['details'] += ' [变异系数偏低，轻微惩罚 +6]'
    
    score += penalty
    return score, details


def _analyze_paragraph_structure_similarity(text, sentences, score, details):
    """知网3.0 - 维度2: 段落内部结构相似度"""
    # 按段落分割
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    
    if len(paragraphs) < 2:
        return score, details
    
    # 简化分析：分析每段的句子数量、连接词分布等结构特征
    paragraph_features = []
    
    for para in paragraphs:
        para_sentences = [s.strip() for s in re.split(r'[。！？!?]+', para) if s.strip()]
        if para_sentences:
            # 特征1: 句子数量
            sent_count = len(para_sentences)
            # 特征2: 平均句长
            avg_len = sum(len(s) for s in para_sentences) / sent_count
            # 特征3: 是否有典型连接词
            has_transition = any(t in para for t in AI_TRANSITIONS_ZH[:10])
            paragraph_features.append((sent_count, avg_len, has_transition))
    
    if len(paragraph_features) >= 2:
        # 简单相似度计算：统计特征相同的段落对
        similar_pairs = 0
        total_pairs = len(paragraph_features) * (len(paragraph_features) - 1) / 2
        
        for i in range(len(paragraph_features)):
            for j in range(i + 1, len(paragraph_features)):
                # 判断两个段落结构是否相似
                sent_count_diff = abs(paragraph_features[i][0] - paragraph_features[j][0])
                avg_len_diff = abs(paragraph_features[i][1] - paragraph_features[j][1])
                transition_same = paragraph_features[i][2] == paragraph_features[j][2]
                
                if sent_count_diff <= 1 and avg_len_diff < 8 and transition_same:
                    similar_pairs += 1
        
        similarity_ratio = similar_pairs / total_pairs if total_pairs > 0 else 0
        
        details['metrics']['paragraph_structure_similarity'] = {
            'similarity_ratio': round(similarity_ratio, 2),
            'paragraph_count': len(paragraphs),
            'details': f'段落结构相似度{round(similarity_ratio*100)}%，共{len(paragraphs)}段'
        }
        
        # 知网阈值从0.75降到0.7
        if similarity_ratio >= 0.7:
            score += 18
            details['metrics']['paragraph_structure_similarity']['details'] += ' [结构高度相似，强烈惩罚 +18]'
        elif similarity_ratio >= 0.55:
            score += 10
            details['metrics']['paragraph_structure_similarity']['details'] += ' [结构较相似，惩罚 +10]'
        elif similarity_ratio >= 0.4:
            score += 5
            details['metrics']['paragraph_structure_similarity']['details'] += ' [结构有相似性，轻微惩罚 +5]'
    
    return score, details


def _analyze_info_density_distribution(text, sentences, score, details):
    """知网3.0 - 维度3: 信息密度分布"""
    if len(sentences) < 3:
        return score, details
    
    import statistics
    
    info_densities = []
    
    for sent in sentences:
        # 实义词：统计中文字符数占比
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', sent))
        total_chars = len(sent)
        if total_chars > 0:
            info_densities.append(chinese_chars / total_chars)
    
    if len(info_densities) < 3:
        return score, details
    
    mean_density = statistics.mean(info_densities)
    std_density = statistics.stdev(info_densities) if len(info_densities) > 1 else 0
    
    # AI文本特征：信息密度稳定在65-75%，且波动极小
    in_ai_zone = sum(1 for d in info_densities if 0.65 <= d <= 0.75)
    ai_zone_ratio = in_ai_zone / len(info_densities)
    
    details['metrics']['info_density_distribution'] = {
        'mean': round(mean_density, 2),
        'std': round(std_density, 3),
        'ai_zone_ratio': round(ai_zone_ratio, 2),
        'details': f'信息密度均值{round(mean_density*100)}%，标准差{round(std_density, 3)}，65-75%区间占比{round(ai_zone_ratio*100)}%'
    }
    
    penalty = 0
    
    # 检查是否大量落在AI典型区间
    if ai_zone_ratio > 0.85:
        penalty += 15
        details['metrics']['info_density_distribution']['details'] += ' [信息密度过于集中在AI区间，强烈惩罚 +15]'
    elif ai_zone_ratio > 0.7:
        penalty += 8
        details['metrics']['info_density_distribution']['details'] += ' [信息密度较集中在AI区间，惩罚 +8]'
    
    # 检查波动是否过小
    if std_density < 0.05:
        penalty += 10
        details['metrics']['info_density_distribution']['details'] += ' [信息密度波动过小，惩罚 +10]'
    elif std_density < 0.08:
        penalty += 5
        details['metrics']['info_density_distribution']['details'] += ' [信息密度波动偏小，轻微惩罚 +5]'
    
    score += penalty
    return score, details


def _analyze_transition_word_distribution(text, sentences, score, details):
    """知网3.0 - 维度4: 连接词频率与分布均匀性"""
    if len(sentences) < 3:
        return score, details
    
    import statistics
    
    # 统计所有连接词
    transition_count = 0
    transition_positions = []
    
    for idx, sent in enumerate(sentences):
        sent_transitions = 0
        for t in AI_TRANSITIONS_ZH:
            if t in sent:
                sent_transitions += 1
                transition_count += 1
                transition_positions.append(idx)
    
    # 计算连接词密度（每千字）
    text_chars = len(text)
    density_per_1000 = (transition_count / text_chars) * 1000 if text_chars > 0 else 0
    
    # 计算分布均匀性
    uniformity_score = 0
    if len(transition_positions) >= 3:
        # 计算相邻连接词之间的距离标准差
        gaps = [transition_positions[i+1] - transition_positions[i] 
                for i in range(len(transition_positions)-1)]
        if gaps:
            mean_gap = statistics.mean(gaps)
            std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
            gap_cv = std_gap / mean_gap if mean_gap > 0 else 0
            uniformity_score = 1 - gap_cv  # 越小越均匀
    
    details['metrics']['transition_word_distribution'] = {
        'total_count': transition_count,
        'density_per_1000': round(density_per_1000, 1),
        'uniformity_score': round(uniformity_score, 2),
        'details': f'连接词{transition_count}个，密度{round(density_per_1000, 1)}个/千字，均匀性{round(uniformity_score*100)}%'
    }
    
    penalty = 0
    
    # AI文本特征：每千字8-15个连接词
    if density_per_1000 > 12:
        penalty += 12
        details['metrics']['transition_word_distribution']['details'] += ' [连接词过度使用，惩罚 +12]'
    elif density_per_1000 > 8:
        penalty += 6
        details['metrics']['transition_word_distribution']['details'] += ' [连接词使用较多，轻微惩罚 +6]'
    
    # 检查分布均匀性
    if uniformity_score > 0.85:
        penalty += 10
        details['metrics']['transition_word_distribution']['details'] += ' [分布过于均匀，惩罚 +10]'
    elif uniformity_score > 0.7:
        penalty += 5
        details['metrics']['transition_word_distribution']['details'] += ' [分布较均匀，轻微惩罚 +5]'
    
    score += penalty
    return score, details
