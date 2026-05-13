import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional


def analyze_chinese_text(text: str, use_advanced: bool = True) -> Tuple[int, Optional[Dict]]:
    """
    增强版中文AI检测器
    
    优化内容:
    1. 扩充AI常用词库(来自知网检测研究和网络资源)
    2. 轻量化技术特征检测(适合2核CPU环境)
    3. 增加更多中文AI特征模式
    """
    details = {'metrics': {}}
    score = 0

    sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
    if not sentences:
        return 0, None

    text_len = len(text)
    if text_len == 0:
        return 0, None

    score, details = _analyze_sentence_uniformity(text, sentences, score, details)

    score, details = _analyze_transitions(text, score, details)

    score, details = _analyze_abstract_language(text, score, details)

    score, details = _analyze_hedging(text, score, details)

    score, details = _analyze_de_density(text, text_len, score, details)

    score, details = _analyze_sentence_openings(sentences, score, details)

    score, details = _analyze_idioms(text, score, details)

    score, details = _analyze_symmetry(text, score, details)

    score, details = _analyze_punctuation(text, score, details)

    if use_advanced:
        score, details = _analyze_entropy(text, score, details)

        score, details = _analyze_bigram_ttr(text, score, details)

        score, details = _analyze_clause_density(sentences, text, score, details)

        score, details = _analyze_de_chains(text, score, details)

    score, details = _analyze_concluding_formula(text, score, details)

    score, details = _analyze_passive_zh(text, text_len, score, details)

    score, details = _analyze_lexical_repetition(text, score, details)

    score, details = _analyze_suizhe_template(text, score, details)

    score, details = _analyze_paragraph_template(text, score, details)

    score, details = _analyze_definition_pattern(text, score, details)

    score, details = _analyze_citation_distribution(sentences, text, score, details)

    if use_advanced:
        score, details = _analyze_info_density(sentences, score, details)

        score, details = _analyze_template_sentences(text, score, details)

    final_score = min(100, max(0, int(score)))
    details['overall_score'] = final_score
    return final_score, details


def _analyze_sentence_uniformity(text, sentences, score, details):
    lengths = [len(s) for s in sentences]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    variance_ratio = std_dev / avg_length if avg_length > 0 else 0

    uniformity_score = 0
    if avg_length < 15:
        if variance_ratio < 0.20:
            uniformity_score = 20
            details['metrics']['sentence_uniformity'] = {"score": 0.9, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极其规律)"}
        elif variance_ratio < 0.35:
            uniformity_score = 10
            details['metrics']['sentence_uniformity'] = {"score": 0.6, "variance_ratio": round(variance_ratio, 3), "details": "中等"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.1, "variance_ratio": round(variance_ratio, 3), "details": "低"}
    else:
        if variance_ratio < 0.25:
            uniformity_score = 35
            details['metrics']['sentence_uniformity'] = {"score": 0.9, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极其规律)"}
        elif variance_ratio < 0.40:
            uniformity_score = 20
            details['metrics']['sentence_uniformity'] = {"score": 0.6, "variance_ratio": round(variance_ratio, 3), "details": "中等"}
        elif variance_ratio < 0.55:
            uniformity_score = 8
            details['metrics']['sentence_uniformity'] = {"score": 0.3, "variance_ratio": round(variance_ratio, 3), "details": "低偏中"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.1, "variance_ratio": round(variance_ratio, 3), "details": "低"}
    score += uniformity_score
    return score, details


AI_TRANSITIONS_ZH = [
    "总而言之", "综上所述", "总体来看", "总体而言", "整体来看", "整体而言",
    "此外", "另外", "不仅如此", "与此同时", "在此基础上",
    "毋庸置疑", "值得注意的是", "必须指出", "不可否认", "显而易见",
    "首先", "其次", "再次", "最后", "第一", "第二", "第三",
    "一方面", "另一方面", "由此可见", "正如前述",
    "更重要的是", "需要强调的是", "关键在于",
    "换言之", "简而言之", "概括来说", "从宏观来看", "从微观来看",
    "进一步说", "具体而言", "一般而言", "通常来说",
    "事实上", "实际上", "本质上", "归根结底",
    "因此", "然而", "但是", "不过", "可是",
    "例如", "比如", "举例来说",
]


def _analyze_transitions(text, score, details):
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
        "items": transition_found[:10],
        "details": f"检测到 {transition_count} 个过渡词"
    }
    score += min(transition_count * 5, 25)
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
    "良好平衡", "某种平衡",
    "强有力的", "强有力的支撑", "坚实的基础",
    "深远意义", "重要价值", "现实意义", "理论意义",
    "取得了良好的效果", "取得了显著的效果",
    "表现出色", "表现优异", "效果显著",
    "具有较强的", "具有较高的", "具有良好的",
    "发挥了重要作用", "起到了关键作用",
    "从...角度来看", "在...方面",
    "不仅...而且", "既...又", "一方面...另一方面",
    "为...做出了贡献", "推动了...的发展",
    "智能化", "自动化", "数字化", "信息化",
    "技术方案", "核心痛点", "取得了平衡",
    "高效", "优异", "显著",
]


def _analyze_abstract_language(text, score, details):
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
    score += min(abstract_count * 7, 35)
    return score, details


AI_HEDGING_ZH = [
    "似乎", "似乎在", "似乎表明", "似乎说明", "似乎验证",
    "可能表明", "可能说明", "可能意味着", "或许", "或许可以", "或可", "或可为",
    "潜在地", "倾向于", "有可能", "不排除", "不能完全排除",
    "可以认为", "可以推测", "可以推断",
    "初步表明", "初步显示",
    "似乎在某种程度上", "可能在一定程度上", "似乎在一定意义上",
]


def _analyze_hedging(text, score, details):
    hedge_found = []
    hedge_count = 0
    for p in AI_HEDGING_ZH:
        cnt = text.count(p)
        if cnt > 0:
            hedge_count += cnt
            hedge_found.append((p, cnt))

    if hedge_count >= 5:
        over_hedge_penalty = min((hedge_count - 4) * 8, 25)
        score += over_hedge_penalty
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:10],
            "details": f"过度对冲词 ({hedge_count}个), 惩罚 +{over_hedge_penalty}"
        }
    elif hedge_count >= 3:
        over_hedge_penalty = min((hedge_count - 2) * 4, 10)
        score += over_hedge_penalty
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:10],
            "details": f"对冲词较多 ({hedge_count}个), 轻微惩罚 +{over_hedge_penalty}"
        }
    elif hedge_count >= 1:
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:10],
            "details": f"存在少量对冲词 ({hedge_count}个)"
        }
    else:
        details['metrics']['over_hedging'] = {"count": 0, "items": [], "details": "未检测到对冲词"}

    genuine_hedging = ["有待进一步", "尚需验证", "仍需探讨", "需进一步研究"]
    genuine_count = sum(text.count(p) for p in genuine_hedging)
    if genuine_count > 0 and hedge_count <= 2:
        score -= min(genuine_count * 3, 6)
        details['metrics']['scholarly_hedging'] = {"count": genuine_count, "details": f"检测到{genuine_count}个真实学术对冲词"}

    return score, details


def _analyze_de_density(text, text_len, score, details):
    de_count = text.count("的")
    shi_count = text.count("是")
    if de_count / text_len > 0.06:
        penalty = min(int((de_count / text_len - 0.06) * 500), 20)
        score += penalty
        details['metrics']['dense_adj'] = {"density": round(de_count / text_len, 4), "details": f"'的'字密度过高, 惩罚 +{penalty}"}
    if shi_count / text_len > 0.025:
        penalty = min(int((shi_count / text_len - 0.025) * 400), 15)
        score += penalty
        details['metrics']['dense_be'] = {"density": round(shi_count / text_len, 4), "details": f"'是'字密度过高, 惩罚 +{penalty}"}
    return score, details


def _analyze_sentence_openings(sentences, score, details):
    sentence_starts = []
    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) >= 4:
            sentence_starts.append(s_clean[:4])

    if len(sentence_starts) >= 3:
        start_counter = Counter(sentence_starts)
        most_common_start = start_counter.most_common(1)[0]
        if most_common_start[1] >= 3:
            repetition_ratio = most_common_start[1] / len(sentence_starts)
            if repetition_ratio > 0.4:
                penalty = 15
            elif repetition_ratio > 0.25:
                penalty = 10
            else:
                penalty = 5
            score += penalty
            details['metrics']['sentence_opening_repetition'] = {
                "top_pattern": most_common_start[0],
                "count": most_common_start[1],
                "ratio": round(repetition_ratio, 3),
                "details": f"句首模式重复, 惩罚 +{penalty}"
            }
    return score, details


COMMON_AI_IDIOMS = [
    "不可或缺", "举足轻重", "至关重要", "显而易见", "毋庸置疑",
    "与日俱增", "日新月异", "蓬勃发展", "方兴未艾", "如火如荼",
    "层出不穷", "琳琅满目", "丰富多彩", "千变万化", "错综复杂",
    "相辅相成", "密不可分", "息息相关", "休戚与共", "一脉相承",
    "卓有成效", "行之有效", "有的放矢", "对症下药", "因地制宜",
    "循序渐进", "稳扎稳打", "精益求精", "孜孜不倦", "锲而不舍",
    "前所未有", "史无前例", "开创性", "突破性", "里程碑",
]


def _analyze_idioms(text, score, details):
    idiom_count = 0
    idiom_found = []
    for idiom in COMMON_AI_IDIOMS:
        cnt = text.count(idiom)
        if cnt > 0:
            idiom_count += cnt
            idiom_found.append((idiom, cnt))

    if idiom_count >= 3:
        penalty = min((idiom_count - 2) * 5, 15)
        score += penalty
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:10],
            "details": f"成语/四字词组 {idiom_count}个, 惩罚 +{penalty}"
        }
    elif idiom_count > 0:
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:10],
            "details": f"成语使用 {idiom_count}个"
        }
    return score, details


SYMMETRY_PATTERNS = [
    r'不仅[^，。]{2,15}，而且[^，。]{2,15}',
    r'既[^，。]{2,10}，又[^，。]{2,10}',
    r'一方面[^，。]{2,20}，另一方面[^，。]{2,20}',
    r'既[^，。]{2,10}也[^，。]{2,10}',
]


def _analyze_symmetry(text, score, details):
    symmetry_count = 0
    for pat in SYMMETRY_PATTERNS:
        symmetry_count += len(re.findall(pat, text))

    if symmetry_count >= 3:
        penalty = min(symmetry_count * 5, 15)
        score += penalty
        details['metrics']['structural_symmetry'] = {
            "count": symmetry_count,
            "details": f"句式对称/排比 {symmetry_count}处, 惩罚 +{penalty}"
        }
    return score, details


def _analyze_punctuation(text, score, details):
    comma_count = text.count('，')
    period_count = text.count('。')
    semicolon_count = text.count('；')

    if period_count > 0:
        comma_period_ratio = comma_count / period_count
        if comma_period_ratio > 3.5:
            penalty = min(int((comma_period_ratio - 3.5) * 5), 15)
            score += penalty
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比过高, 惩罚 +{penalty}"
            }
        else:
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比正常"
            }
    return score, details


def _analyze_entropy(text, score, details):
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；\"\"''《》()（）【】· \n\t"]
    if len(chars_only) > 1:
        freqs = Counter(chars_only)
        entropy = -sum((cnt / len(chars_only)) * math.log2(cnt / len(chars_only)) for cnt in freqs.values())
        details['metrics']['shannon_entropy'] = {
            "value": round(entropy, 2),
            "details": f"信息熵: {round(entropy, 2)}"
        }
        if entropy < 4.0:
            score += 10
            details['metrics']['shannon_entropy']['details'] += " [低, 惩罚 +10]"
    return score, details


def _analyze_bigram_ttr(text, score, details):
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；\"\"''《》()（）【】· \n\t"]
    if len(chars_only) > 10:
        bigrams = ["".join(chars_only[i:i + 2]) for i in range(len(chars_only) - 1)]
        ttr = len(set(bigrams)) / len(bigrams) if bigrams else 1
        details['metrics']['bigram_ttr'] = {
            "value": round(ttr, 3),
            "details": f"2-gram TTR: {round(ttr, 3)}"
        }
        if ttr < 0.55:
            score += 15
            details['metrics']['bigram_ttr']['details'] += " [词汇贫乏, +15]"
        elif ttr < 0.65:
            score += 8
            details['metrics']['bigram_ttr']['details'] += " [词汇偏单调, +8]"
    return score, details


def _analyze_clause_density(sentences, text, score, details):
    clauses = [c for c in re.split(r'[，。！？；：]+', text) if c.strip()]
    if sentences:
        clauses_per_sentence = len(clauses) / len(sentences)
        details['metrics']['clause_chain_density'] = {
            "value": round(clauses_per_sentence, 2),
            "details": f"从句密度: {round(clauses_per_sentence, 2)}"
        }
        if clauses_per_sentence > 4.0:
            score += 12
            details['metrics']['clause_chain_density']['details'] += " [重度嵌套, +12]"
        elif clauses_per_sentence > 3.2:
            score += 6
            details['metrics']['clause_chain_density']['details'] += " [中度嵌套, +6]"
    return score, details


def _analyze_de_chains(text, score, details):
    de_chain_pattern = r'的[^的]{0,4}的[^的]{0,4}的'
    de_chains = re.findall(de_chain_pattern, text)
    if len(de_chains) >= 2:
        penalty = min(len(de_chains) * 5, 15)
        score += penalty
        details['metrics']['de_chain'] = {
            "count": len(de_chains),
            "examples": de_chains[:3],
            "details": f"'的'字链 {len(de_chains)}处, 惩罚 +{penalty}"
        }
    return score, details


CONCLUDING_PATTERNS = [
    "总体来看", "总而言之", "综上所述", "总的来说",
    "由此可见", "综上", "概而言之",
    "本文将重点研究", "本文旨在", "本文拟",
    "本文通过", "本文基于", "本文提出",
    "具有重要的现实意义", "具有重要的理论意义",
]


def _analyze_concluding_formula(text, score, details):
    concluding_count = 0
    concluding_found = []
    for p in CONCLUDING_PATTERNS:
        cnt = text.count(p)
        if cnt > 0:
            concluding_count += cnt
            concluding_found.append((p, cnt))

    if concluding_count >= 2:
        penalty = min((concluding_count - 1) * 6, 18)
        score += penalty
        details['metrics']['concluding_formula'] = {
            "count": concluding_count,
            "items": concluding_found[:10],
            "details": f"总结套话 {concluding_count}处, 惩罚 +{penalty}"
        }
    return score, details


PASSIVE_ZH = ["被", "受到", "得到", "得以", "予以", "加以"]


def _analyze_passive_zh(text, text_len, score, details):
    passive_count = sum(text.count(p) for p in PASSIVE_ZH)
    if passive_count >= 4 and text_len > 0:
        passive_density = passive_count / text_len
        if passive_density > 0.02:
            penalty = 8
            score += penalty
            details['metrics']['passive_overuse'] = {
                "count": passive_count,
                "density": round(passive_density, 4),
                "details": f"被动句式 {passive_count}个, 惩罚 +{penalty}"
            }
    return score, details


def _analyze_lexical_repetition(text, score, details):
    content_chars_segments = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
    if content_chars_segments:
        all_bigrams_text = []
        for seg in content_chars_segments:
            for i in range(len(seg) - 1):
                all_bigrams_text.append(seg[i:i + 2])
        if all_bigrams_text:
            bigram_counter = Counter(all_bigrams_text)
            top_bigram = bigram_counter.most_common(1)[0]
            if top_bigram[1] >= 4 and len(all_bigrams_text) > 0:
                repetition_ratio = top_bigram[1] / len(all_bigrams_text)
                if repetition_ratio > 0.08:
                    penalty = 8
                    score += penalty
                    details['metrics']['lexical_repetition'] = {
                        "top_bigram": top_bigram[0],
                        "count": top_bigram[1],
                        "ratio": round(repetition_ratio, 3),
                        "details": f"高频词 \"{top_bigram[0]}\" 重复{top_bigram[1]}次, 惩罚 +{penalty}"
                    }
    return score, details


SUIZHE_PATTERNS = [
    r'随着[^，。]{2,20}的[^，。]{2,20}',
    r'基于[^，。]{2,20}的[^，。]{2,20}',
    r'通过[^，。]{2,20}的[^，。]{2,20}',
    r'利用[^，。]{2,20}的[^，。]{2,20}',
]


def _analyze_suizhe_template(text, score, details):
    suizhe_count = 0
    suizhe_found = []
    for pat in SUIZHE_PATTERNS:
        matches = re.findall(pat, text)
        suizhe_count += len(matches)
        suizhe_found.extend(matches)

    if suizhe_count >= 2:
        penalty = min(suizhe_count * 6, 20)
        score += penalty
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:5],
            "details": f"'随着/基于/通过...的...'模板 {suizhe_count}处, 惩罚 +{penalty}"
        }
    elif suizhe_count > 0:
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:5],
            "details": f"'随着/基于...的...'模板 {suizhe_count}处"
        }
    return score, details


def _analyze_paragraph_template(text, score, details):
    paragraph_structure_markers = 0
    bg_markers = ["是...的重要", "在...中发挥", "作为...的", "近年来", "随着"]
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
    paper_markers = ["本文", "本研究", "本文旨在", "本文拟", "本文将"]
    for m in paper_markers:
        if m in text:
            paragraph_structure_markers += 1
            break

    if paragraph_structure_markers >= 3:
        penalty = (paragraph_structure_markers - 2) * 8
        score += penalty
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"模板结构 (检测到 {paragraph_structure_markers}/4标记), 惩罚 +{penalty}"
        }
    else:
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"段落结构自然 ({paragraph_structure_markers}/4标记)"
        }
    return score, details


def _analyze_definition_pattern(text, score, details):
    definition_pattern = r'[^，。]{3,20}是[^，。]{3,30}的[^，。]{2,20}'
    definition_matches = re.findall(definition_pattern, text)
    if len(definition_matches) >= 2:
        penalty = min(len(definition_matches) * 5, 15)
        score += penalty
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:3],
            "details": f"'是...的'定义句 {len(definition_matches)}处, 惩罚 +{penalty}"
        }
    elif len(definition_matches) > 0:
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:3],
            "details": f"'是...的'定义句 {len(definition_matches)}处"
        }
    return score, details


def _analyze_citation_distribution(sentences, text, score, details):
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
        if sentences_with_citations > 0 and sentence_end_citations / sentences_with_citations > 0.8:
            penalty = 10
            score += penalty
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2),
                "details": f"引用集中句末 {sentence_end_citations}/{sentences_with_citations}, 惩罚 +{penalty}"
            }
        else:
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2) if sentences_with_citations > 0 else 0,
                "details": f"引用分布正常"
            }
    return score, details


def _analyze_info_density(sentences, score, details):
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
            if std < 0.05:
                penalty = 12
                score += penalty
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度过于均匀, 惩罚 +{penalty}"
                }
            else:
                details['metrics']['info_density_uniformity'] = {
                    "std": round(std, 4),
                    "details": f"信息密度有变化"
                }
    return score, details


TEMPLATE_SENTENCE_STARTS = [
    "随着", "基于", "通过", "利用", "采用",
    "旨在", "为了", "通过", "根据",
]


def _analyze_template_sentences(text, score, details):
    template_count = 0
    for pattern in TEMPLATE_SENTENCE_STARTS:
        template_count += text.count(pattern)

    if template_count >= 5:
        penalty = min((template_count - 4) * 3, 15)
        score += penalty
        details['metrics']['template_sentence_starts'] = {
            "count": template_count,
            "details": f"模板化句首 {template_count}处, 惩罚 +{penalty}"
        }
    return score, details
