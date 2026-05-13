import re
import math
import numpy as np
from collections import Counter


def analyze_chinese_text(text):
    details = {'metrics': {}}
    score = 0

    sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
    if not sentences:
        return 0, None

    text_len = len(text)
    if text_len == 0:
        return 0, None

    # ============================================================
    # 1. 句长均匀度 (Burstiness / Sentence Uniformity)
    # ============================================================
    lengths = [len(s) for s in sentences]
    avg_length = np.mean(lengths)
    std_dev = np.std(lengths)
    variance_ratio = std_dev / avg_length if avg_length > 0 else 0
    cv = std_dev / avg_length if avg_length > 0 else 0

    uniformity_score = 0
    # 短文本（平均句长<15字，如口语化文本）对均匀度要求更宽松
    if avg_length < 15:
        if variance_ratio < 0.20:
            uniformity_score = 20
            details['metrics']['sentence_uniformity'] = {"score": 0.9, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极其规律，极度符合AI生成特征)"}
        elif variance_ratio < 0.35:
            uniformity_score = 10
            details['metrics']['sentence_uniformity'] = {"score": 0.6, "variance_ratio": round(variance_ratio, 3), "details": "中等 (句长偏规律，缺乏人类错落感)"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.1, "variance_ratio": round(variance_ratio, 3), "details": "低 (句长参差错落，呈现人类 Burstiness)"}
    else:
        if variance_ratio < 0.25:
            uniformity_score = 35
            details['metrics']['sentence_uniformity'] = {"score": 0.9, "variance_ratio": round(variance_ratio, 3), "details": "高 (句长极其规律，极度符合AI生成特征)"}
        elif variance_ratio < 0.40:
            uniformity_score = 20
            details['metrics']['sentence_uniformity'] = {"score": 0.6, "variance_ratio": round(variance_ratio, 3), "details": "中等 (句长偏规律，缺乏人类错落感)"}
        elif variance_ratio < 0.55:
            uniformity_score = 8
            details['metrics']['sentence_uniformity'] = {"score": 0.3, "variance_ratio": round(variance_ratio, 3), "details": "低偏中 (句长有一定变化)"}
        else:
            details['metrics']['sentence_uniformity'] = {"score": 0.1, "variance_ratio": round(variance_ratio, 3), "details": "低 (句长参差错落，呈现人类 Burstiness)"}
    score += uniformity_score

    # ============================================================
    # 2. 机器式过渡词 (Transition Overuse)
    # ============================================================
    ai_transitions_zh = [
        "总而言之", "综上所述", "总体来看", "总体而言", "整体来看", "整体而言",
        "此外", "另外", "不仅如此", "与此同时", "在此基础上",
        "毋庸置疑", "值得注意的是", "必须指出", "不可否认", "显而易见",
        "首先", "其次", "再次", "最后", "第一", "第二", "第三",
        "一方面", "另一方面", "由此可见", "正如前述",
        "更重要的是", "需要强调的是", "关键在于",
        "换言之", "简而言之", "概括来说", "从宏观来看", "从微观来看",
        "进一步说", "换言之", "具体而言", "一般而言", "通常来说",
        "事实上", "实际上", "本质上", "归根结底",
        "因此", "然而", "但是", "不过", "可是",
        "例如", "比如", "如", "举例来说",
        "随着", "基于", "通过", "利用", "采用",
        "为了", "旨在", "以期", "以期达到",
    ]
    transition_found = []
    transition_count = 0
    for p in ai_transitions_zh:
        cnt = text.count(p)
        if cnt > 0:
            transition_count += cnt
            transition_found.append((p, cnt))
    details['metrics']['transition_overuse'] = {
        "count": transition_count,
        "found": transition_count,
        "items": transition_found[:10],
        "details": f"检测到 {transition_count} 个机器常滥用的过渡词"
    }
    score += min(transition_count * 5, 25)

    # ============================================================
    # 3. 空泛套话/大词/卖弄词 (Abstract Language)
    # ============================================================
    abstract_zh = [
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
    abstract_found = []
    abstract_count = 0
    for p in abstract_zh:
        cnt = text.count(p)
        if cnt > 0:
            abstract_count += cnt
            abstract_found.append((p, cnt))
    details['metrics']['abstract_language'] = {
        "count": abstract_count,
        "found": abstract_count,
        "items": abstract_found[:15],
        "details": f"检测到 {abstract_count} 个空泛套话/大词短语"
    }
    score += min(abstract_count * 7, 35)

    # ============================================================
    # 4. AI过度对冲词检测 (Over-Hedging)
    # ============================================================
    ai_hedging_zh = [
        "似乎", "似乎在一定程度上", "或可", "或可为", "或许",
        "可能表明", "可能说明", "潜在地", "暗示了",
        "倾向于", "在一定程度上", "在某种程度上",
        "某种程度上", "某种意义上",
        "有可能", "不排除", "不能完全排除",
        "可以认为", "可以推测", "可以推断",
        "有理由相信", "有理由认为",
        "初步表明", "初步显示",
        "似乎表明", "似乎说明", "似乎验证",
    ]
    hedge_found = []
    hedge_count = 0
    for p in ai_hedging_zh:
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
            "details": f"AI过度对冲词堆砌 (检测到 {hedge_count} 个，如'似乎'、'可能表明'等，这是AI模仿学术写作的典型痕迹，惩罚 +{over_hedge_penalty})"
        }
    elif hedge_count >= 3:
        over_hedge_penalty = min((hedge_count - 2) * 4, 10)
        score += over_hedge_penalty
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:10],
            "details": f"存在较多对冲词 ({hedge_count} 个)，可能偏AI化，轻微惩罚 +{over_hedge_penalty}"
        }
    elif hedge_count >= 1:
        details['metrics']['over_hedging'] = {
            "count": hedge_count,
            "items": hedge_found[:10],
            "details": f"存在少量对冲词 ({hedge_count} 个)，属正常学术表达"
        }
    else:
        details['metrics']['over_hedging'] = {
            "count": 0,
            "items": [],
            "details": "未检测到对冲词"
        }

    # 保留真正的学术严谨对冲词（适度使用是人类特征）
    genuine_hedging = ["有待进一步", "尚需验证", "仍需探讨", "需进一步研究"]
    genuine_count = sum(text.count(p) for p in genuine_hedging)
    if genuine_count > 0 and hedge_count <= 2:
        score -= min(genuine_count * 3, 6)
        details['metrics']['scholarly_hedging'] = {"count": genuine_count, "details": f"检测到 {genuine_count} 个真实学术对冲词 (人类严谨特征，减免AI分)"}

    # ============================================================
    # 5. "的"字密度 / 形容词堆砌
    # ============================================================
    de_count = text.count("的")
    shi_count = text.count("是")
    if de_count / text_len > 0.06:
        penalty = min(int((de_count / text_len - 0.06) * 500), 20)
        score += penalty
        details['metrics']['dense_adj'] = {"density": round(de_count / text_len, 4), "details": f"'的'字密度过高 ({round(de_count/text_len*100, 1)}%)，典型AI形容词堆砌，惩罚 +{penalty}"}
    if shi_count / text_len > 0.025:
        penalty = min(int((shi_count / text_len - 0.025) * 400), 15)
        score += penalty
        details['metrics']['dense_be'] = {"density": round(shi_count / text_len, 4), "details": f"'是'字密度过高 ({round(shi_count/text_len*100, 1)}%)，AI定义式回答特征，惩罚 +{penalty}"}

    # ============================================================
    # 6. 句首模式重复检测 (Sentence Opening Repetition)
    # ============================================================
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
                "details": f"句首模式重复 (\"{most_common_start[0]}\" 出现 {most_common_start[1]} 次，占比 {round(repetition_ratio*100,1)}%)，惩罚 +{penalty}"
            }

    # ============================================================
    # 7. 成语/四字词组滥用检测 (Idiom Overuse)
    # ============================================================
    common_ai_idioms = [
        "不可或缺", "举足轻重", "至关重要", "显而易见", "毋庸置疑",
        "与日俱增", "日新月异", "蓬勃发展", "方兴未艾", "如火如荼",
        "层出不穷", "琳琅满目", "丰富多彩", "千变万化", "错综复杂",
        "相辅相成", "密不可分", "息息相关", "休戚与共", "一脉相承",
        "卓有成效", "行之有效", "有的放矢", "对症下药", "因地制宜",
        "循序渐进", "稳扎稳打", "精益求精", "孜孜不倦", "锲而不舍",
        "前所未有", "史无前例", "开创性", "突破性", "里程碑",
    ]
    idiom_count = 0
    idiom_found = []
    for idiom in common_ai_idioms:
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
            "details": f"成语/四字词组滥用 (检测到 {idiom_count} 个)，AI生成文本典型特征，惩罚 +{penalty}"
        }
    elif idiom_count > 0:
        details['metrics']['idiom_overuse'] = {
            "count": idiom_count,
            "items": idiom_found[:10],
            "details": f"成语使用适中 ({idiom_count} 个)"
        }

    # ============================================================
    # 8. 句式对称性/排比检测 (Structural Symmetry)
    # ============================================================
    symmetry_patterns = [
        r'不仅[^，。]{2,15}，而且[^，。]{2,15}',
        r'既[^，。]{2,10}，又[^，。]{2,10}',
        r'一方面[^，。]{2,20}，另一方面[^，。]{2,20}',
        r'既[^，。]{2,10}也[^，。]{2,10}',
    ]
    symmetry_count = 0
    for pat in symmetry_patterns:
        symmetry_count += len(re.findall(pat, text))

    if symmetry_count >= 3:
        penalty = min(symmetry_count * 5, 15)
        score += penalty
        details['metrics']['structural_symmetry'] = {
            "count": symmetry_count,
            "details": f"句式对称/排比结构过多 ({symmetry_count} 处)，AI典型模式，惩罚 +{penalty}"
        }

    # ============================================================
    # 9. 标点密度分析 (Punctuation Density)
    # ============================================================
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
                "details": f"逗号/句号比过高 ({round(comma_period_ratio, 2)})，说明单句内嵌套过多从句，AI特征，惩罚 +{penalty}"
            }
        else:
            details['metrics']['punctuation_density'] = {
                "comma_period_ratio": round(comma_period_ratio, 2),
                "details": f"逗号/句号比正常 ({round(comma_period_ratio, 2)})"
            }

    # ============================================================
    # 10. NLP 技术特征 (Shannon Entropy, Bigram TTR, Clause Density)
    # ============================================================
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；\"\"''《》()（）【】· \n\t"]
    if len(chars_only) > 1:
        # 10a. Shannon Entropy
        freqs = Counter(chars_only)
        entropy = -sum((cnt / len(chars_only)) * math.log2(cnt / len(chars_only)) for cnt in freqs.values())
        details['metrics']['shannon_entropy'] = {
            "value": round(entropy, 2),
            "details": f"文本信息熵: {round(entropy, 2)}"
        }
        if entropy < 4.0:
            score += 10
            details['metrics']['shannon_entropy']['details'] += " [触发低概率分布惩罚 +10]"

        # 10b. Bigram TTR
        bigrams = ["".join(chars_only[i:i + 2]) for i in range(len(chars_only) - 1)]
        ttr = len(set(bigrams)) / len(bigrams) if bigrams else 1
        details['metrics']['bigram_ttr'] = {
            "value": round(ttr, 3),
            "details": f"二元词汇多样度 (2-gram TTR): {round(ttr, 3)}"
        }
        if ttr < 0.55:
            score += 15
            details['metrics']['bigram_ttr']['details'] += " [触发词汇贫乏惩罚 +15]"
        elif ttr < 0.65:
            score += 8
            details['metrics']['bigram_ttr']['details'] += " [触发词汇偏单调惩罚 +8]"

        # 10c. Clause Chain Density
        clauses = [c for c in re.split(r'[，。！？；：]+', text) if c.strip()]
        if len(actual_sentences := [s for s in re.split(r'[。！？.!?]+', text) if s.strip()]) > 0:
            clauses_per_sentence = len(clauses) / len(actual_sentences)
            details['metrics']['clause_chain_density'] = {
                "value": round(clauses_per_sentence, 2),
                "details": f"单句从句嵌套密度: {round(clauses_per_sentence, 2)}"
            }
            if clauses_per_sentence > 4.0:
                score += 12
                details['metrics']['clause_chain_density']['details'] += " [触发重度嵌套惩罚 +12]"
            elif clauses_per_sentence > 3.2:
                score += 6
                details['metrics']['clause_chain_density']['details'] += " [触发中度嵌套惩罚 +6]"

    # ============================================================
    # 11. "的"字链检测 (Multiple "的" in sequence)
    # ============================================================
    de_chain_pattern = r'的[^的]{0,4}的[^的]{0,4}的'
    de_chains = re.findall(de_chain_pattern, text)
    if len(de_chains) >= 2:
        penalty = min(len(de_chains) * 5, 15)
        score += penalty
        details['metrics']['de_chain'] = {
            "count": len(de_chains),
            "examples": de_chains[:3],
            "details": f"检测到 {len(de_chains)} 处'的'字链 (如'{de_chains[0]}')，AI翻译腔强信号，惩罚 +{penalty}"
        }

    # ============================================================
    # 12. 段末总结套话检测 (Concluding Formulaic Patterns)
    # ============================================================
    concluding_patterns = [
        "总体来看", "总而言之", "综上所述", "总的来说",
        "由此可见", "综上", "概而言之",
        "为...提供了", "为...做出了", "推动了...的发展",
        "实现了从...到...的", "完成了从...到...的",
        "本文将重点研究", "本文旨在", "本文拟",
        "本文通过", "本文基于", "本文提出",
        "具有重要的现实意义", "具有重要的理论意义",
        "对于...具有重大的现实意义",
    ]
    concluding_count = 0
    concluding_found = []
    for p in concluding_patterns:
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
            "details": f"段末总结套话/本文指向句过多 ({concluding_count} 处)，AI典型模式，惩罚 +{penalty}"
        }

    # ============================================================
    # 13. 被动/使役句式过度使用 (Passive/Causative Overuse)
    # ============================================================
    passive_zh = ["被", "受到", "得到", "得以", "予以", "加以"]
    passive_count = sum(text.count(p) for p in passive_zh)
    if passive_count >= 4 and text_len > 0:
        passive_density = passive_count / text_len
        if passive_density > 0.02:
            penalty = 8
            score += penalty
            details['metrics']['passive_overuse'] = {
                "count": passive_count,
                "density": round(passive_density, 4),
                "details": f"被动/使役句式密度偏高 ({passive_count} 个)，惩罚 +{penalty}"
            }

    # ============================================================
    # 14. 词汇重复度检测 (Lexical Repetition)
    # ============================================================
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
                        "details": f"高频二元词重复 (\"{top_bigram[0]}\" 出现 {top_bigram[1]} 次)，惩罚 +{penalty}"
                    }

    # ============================================================
    # 15. 【新增】"随着...的..." AI模板句式检测
    # 这是AI写中文论文的标志性句式
    # ============================================================
    suizhe_patterns = [
        r'随着[^，。]{2,20}的[^，。]{2,20}',
        r'基于[^，。]{2,20}的[^，。]{2,20}',
        r'通过[^，。]{2,20}的[^，。]{2,20}',
        r'利用[^，。]{2,20}的[^，。]{2,20}',
    ]
    suizhe_count = 0
    suizhe_found = []
    for pat in suizhe_patterns:
        matches = re.findall(pat, text)
        suizhe_count += len(matches)
        suizhe_found.extend(matches)

    if suizhe_count >= 2:
        penalty = min(suizhe_count * 6, 20)
        score += penalty
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:5],
            "details": f"检测到 {suizhe_count} 处'随着/基于/通过...的...'模板句式，AI论文写作标志性特征，惩罚 +{penalty}"
        }
    elif suizhe_count > 0:
        details['metrics']['suizhe_template'] = {
            "count": suizhe_count,
            "examples": suizhe_found[:5],
            "details": f"检测到 {suizhe_count} 处'随着/基于...的...'模板句式"
        }

    # ============================================================
    # 16. 【新增】段落结构模板检测 (Paragraph Structure Template)
    # AI论文引言常用"背景→问题→意义→本文方案"的机械结构
    # ============================================================
    paragraph_structure_markers = 0
    # 背景引入标记
    bg_markers = ["是...的重要", "在...中发挥", "作为...的", "近年来", "随着"]
    for m in bg_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    # 问题/不足标记
    problem_markers = ["然而", "但是", "不足", "局限", "问题", "挑战", "困难"]
    for m in problem_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    # 意义/价值标记
    value_markers = ["具有重要意义", "具有重要价值", "现实意义", "理论意义", "应用价值"]
    for m in value_markers:
        if m in text:
            paragraph_structure_markers += 1
            break
    # 本文方案标记
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
            "details": f"段落结构过于模板化 (检测到 {paragraph_structure_markers}/4 个结构标记：背景→问题→意义→本文方案)，AI论文引言典型机械结构，惩罚 +{penalty}"
        }
    else:
        details['metrics']['paragraph_template'] = {
            "marker_count": paragraph_structure_markers,
            "details": f"段落结构自然度尚可 ({paragraph_structure_markers}/4 个结构标记)"
        }

    # ============================================================
    # 17. 【新增】"是"字定义句式检测 (Definition Pattern)
    # AI喜欢用"XX是YY"的机械定义句式
    # ============================================================
    definition_pattern = r'[^，。]{3,20}是[^，。]{3,30}的[^，。]{2,20}'
    definition_matches = re.findall(definition_pattern, text)
    if len(definition_matches) >= 2:
        penalty = min(len(definition_matches) * 5, 15)
        score += penalty
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:3],
            "details": f"检测到 {len(definition_matches)} 处'是...的'定义式句式，AI机械表达特征，惩罚 +{penalty}"
        }
    elif len(definition_matches) > 0:
        details['metrics']['definition_pattern'] = {
            "count": len(definition_matches),
            "examples": definition_matches[:3],
            "details": f"检测到 {len(definition_matches)} 处'是...的'定义式句式"
        }

    # ============================================================
    # 18. 【新增】引用分布均匀度检测 (Citation Distribution)
    # AI生成的引用往往均匀分布在句末
    # ============================================================
    citation_pattern = r'\[\d+(?:-\d+)?\]'
    citations = re.findall(citation_pattern, text)
    if len(citations) >= 3:
        # 检查引用是否都出现在句末
        sentences_with_citations = 0
        sentence_end_citations = 0
        for s in sentences:
            if re.search(citation_pattern, s):
                sentences_with_citations += 1
                # 检查引用是否在句子最后5个字符内
                if re.search(citation_pattern + r'[^\u4e00-\u9fa5a-zA-Z]{0,3}$', s):
                    sentence_end_citations += 1
        if sentences_with_citations > 0 and sentence_end_citations / sentences_with_citations > 0.8:
            penalty = 10
            score += penalty
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2),
                "details": f"引用过度集中在句末 ({sentence_end_citations}/{sentences_with_citations} 句)，AI机械排版特征，惩罚 +{penalty}"
            }
        else:
            details['metrics']['citation_distribution'] = {
                "total_citations": len(citations),
                "end_citation_ratio": round(sentence_end_citations / sentences_with_citations, 2) if sentences_with_citations > 0 else 0,
                "details": f"引用分布正常 ({len(citations)} 个引用)"
            }

    # ============================================================
    # 19. 【新增】信息密度均匀度检测 (Information Density Uniformity)
    # AI每句话的信息量均匀，人类有起伏
    # ============================================================
    if len(sentences) >= 3:
        # 用每句话中实词（名词、动词）的比例来估算信息密度
        info_densities = []
        for s in sentences:
            # 简单的启发式：内容字符数 / 总字符数
            content_chars = len(re.findall(r'[\u4e00-\u9fa5]', s))
            total_chars = len(s)
            if total_chars > 0:
                info_densities.append(content_chars / total_chars)
        if len(info_densities) >= 3:
            info_density_std = np.std(info_densities)
            if info_density_std < 0.05:
                penalty = 12
                score += penalty
                details['metrics']['info_density_uniformity'] = {
                    "std": round(info_density_std, 4),
                    "details": f"信息密度过于均匀 (std={round(info_density_std, 4)})，AI文本每句话信息量相近，人类写作有起伏，惩罚 +{penalty}"
                }
            else:
                details['metrics']['info_density_uniformity'] = {
                    "std": round(info_density_std, 4),
                    "details": f"信息密度有变化 (std={round(info_density_std, 4)})"
                }

    final_score = min(100, max(0, int(score)))
    details['overall_score'] = final_score
    return final_score, details
