import re
import math
import numpy as np
from collections import Counter

def analyze_chinese_text(text):
    details = {'metrics': {}}
    score = 0
    
    # 拆分句子
    sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
    if not sentences:
        return 0, None
        
    # 1. Uniformity (句长均匀度)
    lengths = [len(s) for s in sentences]
    avg_length = np.mean(lengths)
    std_dev = np.std(lengths)
    variance_ratio = std_dev / avg_length if avg_length > 0 else 0
    
    uniformity_score = 0
    if variance_ratio < 0.35:
        uniformity_score = 40
        details['metrics']['sentence_uniformity'] = {"score": 0.9, "details": "高 (句长极其规律，呆板机械，极度符合AI生成特征)"}
    elif variance_ratio < 0.50:
        uniformity_score = 20
        details['metrics']['sentence_uniformity'] = {"score": 0.6, "details": "中等 (句长有一定规律，缺乏人类错落感)"}
    else:
        details['metrics']['sentence_uniformity'] = {"score": 0.1, "details": "低 (句长参差错落，呈现人类 Burstiness)"}
    score += uniformity_score
    
    # 2. Transitions (机器式过渡词，扩大词库与比重)
    ai_phrases_zh = [
        "总而言之", "此外", "另外", "毋庸置疑", "值得注意的是", "必须指出", 
        "不可否认", "总的来说", "综上所述", "从各个方面", "首先", "其次", 
        "再次", "最后", "一方面", "另一方面", "由此可见", "正如", "不仅如此"
    ]
    transition_count = sum(text.count(p) for p in ai_phrases_zh)
    details['metrics']['transition_overuse'] = {"count": transition_count, "details": f"检测到 {transition_count} 个机器常滥用的过渡词"}
    score += min(transition_count * 15, 45)  # 严格惩罚
    
    # 3. Abstract Language (套话/空话/卖弄词/大词，极大扩容防抓漏)
    abstract_zh = [
        "多种因素", "各个方面", "深远的影响", "发挥着至关重要的作用", "扮演着重要的角色", 
        "不可忽视", "具有重要意义", "不可剥夺", "显而易见", "毫无疑问", 
        "深刻揭示了", "不可或缺", "综合运用", "提供了理论支撑", "为后续研究提供基础",
        "完善了理论体系", "开启了新篇章", "此案例印证了", "验证了可行性", "深入探讨",
        "具有一定的局限性", "全面阐述", "系统梳理", "深入分析", "这不难理解",
        "随着社会的不断发展", "在当前背景下", "为...指明了方向", "有着广泛的应用"
    ]
    abstract_count = sum(text.count(p) for p in abstract_zh)
    details['metrics']['abstract_language'] = {"count": abstract_count, "details": f"检测到 {abstract_count} 个空泛套话短语"}
    score += min(abstract_count * 15, 45) # 严格惩罚
    
    # 3.5 密集程度惩罚 (的、是 被过度使用)
    de_count = text.count("的")
    shi_count = text.count("是")
    text_len = len(text)
    if text_len > 0:
        if de_count / text_len > 0.07:  # 超过7%是"的"，这通常由于AI堆砌形容词
            score += 15
            details['metrics']['dense_adj'] = {"details": "高度堆砌的形容词结构（'的'字出现密度极高，典型的机器翻译腔/拼凑腔）"}
        if shi_count / text_len > 0.03: # 过多使用判断动词"是"
            score += 10
            details['metrics']['dense_be'] = {"details": "过多使用系动词'是'（AI喜欢用是来下定义式回答）"}
            
    # 3.8 市面主流检测器对齐：NLP 技术特征测算 (Technical Metrics)
    chars_only = [c for c in text if c.strip() and c not in "，。！？、：；“”‘’《》()（）【】· \n\t"]
    if len(chars_only) > 1:
        # 1. 字符信息熵 (Shannon Entropy) -> 反映模型概率分布的平滑性
        freqs = Counter(chars_only)
        entropy = -sum((cnt/len(chars_only)) * math.log2(cnt/len(chars_only)) for cnt in freqs.values())
        details['metrics']['shannon_entropy'] = {
            "value": round(entropy, 2), 
            "details": f"文本散度/信息熵 (Entropy): {round(entropy, 2)} (AI生成文本的字频分布往往高度集中在常见字区间)"
        }
        if entropy < 4.0: # 假设阈值，分布过于集中
            score += 10
            details['metrics']['shannon_entropy']['details'] += " [触发低概率分布惩罚]"
            
        # 2. 二元词汇多样度 (Bigram Type-Token Ratio, TTR) -> 衡量机器复读机特性
        bigrams = ["".join(chars_only[i:i+2]) for i in range(len(chars_only)-1)]
        ttr = len(set(bigrams)) / len(bigrams) if bigrams else 1
        details['metrics']['bigram_ttr'] = {
            "value": round(ttr, 3), 
            "details": f"高级二元多样度 (2-gram TTR): {round(ttr, 3)} (市面核心技术指标，低TTR说明系统在反复调用相似固定词对)"
        }
        if ttr < 0.65:
            score += 15
            details['metrics']['bigram_ttr']['details'] += " [触发词汇贫乏与复用惩罚]"
            
        # 3. 句法嵌套深度代理 (Clause Chain Density) -> AI的“特长定语句”特性
        clauses = [c for c in re.split(r'[，。！？；：]+', text) if c.strip()]
        actual_sentences = [s for s in re.split(r'[。！？.!?]+', text) if s.strip()]
        if len(actual_sentences) > 0:
            clauses_per_sentence = len(clauses) / len(actual_sentences)
            details['metrics']['clause_chain_density'] = {
                "value": round(clauses_per_sentence, 2), 
                "details": f"单句从句嵌套密度 (Clause Density): {round(clauses_per_sentence, 2)} (大模型偏爱层层嵌套的绵长定语/状语从句)"
            }
            if clauses_per_sentence > 3.8:
                score += 10
                details['metrics']['clause_chain_density']['details'] += " [触发重度巨长从句结构惩罚]"
    
    # 4. Hedging (学术对冲词，减分项 - 人类常用)
    hedging_zh = ["可能表明", "似乎", "潜在地", "有待进一步", "暗示了", "在某种程度上", "倾向于"]
    hedge_count = sum(text.count(p) for p in hedging_zh)
    details['metrics']['scholarly_hedging'] = {"count": hedge_count, "details": f"检测到 {hedge_count} 个学术对冲词 (人类严谨特征，减免AI分)"}
    score -= min(hedge_count * 5, 15)
    
    final_score = min(100, max(0, int(score)))
    details['overall_score'] = final_score
    return final_score, details