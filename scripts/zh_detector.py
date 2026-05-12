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
    # 2. 机器式过渡词 (Transition Overuse) — 大幅扩容
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
    score += min(transition_count * 8, 30)

    # ============================================================
    # 3. 空泛套话/大词/卖弄词 (Abstract Language) — 极大扩容
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
    score += min(abstract_count * 6, 30)

    # ============================================================
    # 4. AI过度对冲词检测 (Over-Hedging) — 关键修正
    # 旧逻辑把对冲词当人类特征减分，这是错误的。
    # AI改写后的文本恰恰会堆砌"似乎"、"可能表明"等对冲词，
    # 这是AI模仿人类学术写作的典型痕迹，应予惩罚。
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
    # AI倾向用相同句式开头，如连续多个"本文..."、"该模型..."
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
    # AI生成中文文本时倾向大量堆砌四字成语，人类使用更克制
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

    if idiom_count >= 4:
        penalty = min((idiom_count - 3) * 4, 15)
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
    # AI极度喜欢排比和对仗结构，人类写作更随意
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
    # AI文本逗号密度偏高（因为喜欢长定语从句），
    # 人类文本句号比例更高（更多短句）
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
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；""''《》()（）【】· \n\t"]
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
    # "XX的XX的XX" 是AI翻译腔/拼凑腔的强信号
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
    # AI极度喜欢在段落末尾加总结性套话
    # ============================================================
    concluding_patterns = [
        "总体来看", "总而言之", "综上所述", "总的来说",
        "由此可见", "综上", "概而言之",
        "为...提供了", "为...做出了", "推动了...的发展",
        "实现了从...到...的", "完成了从...到...的",
    ]
    concluding_count = 0
    concluding_found = []
    for p in concluding_patterns:
        cnt = text.count(p)
        if cnt > 0:
            concluding_count += cnt
            concluding_found.append((p, cnt))

    if concluding_count >= 3:
        penalty = min((concluding_count - 2) * 5, 15)
        score += penalty
        details['metrics']['concluding_formula'] = {
            "count": concluding_count,
            "items": concluding_found[:10],
            "details": f"段末总结套话过多 ({concluding_count} 处)，AI典型模式，惩罚 +{penalty}"
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
    # AI在短文本中容易反复使用相同的关键词
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

    final_score = min(100, max(0, int(score)))
    details['overall_score'] = final_score
    return final_score, details
