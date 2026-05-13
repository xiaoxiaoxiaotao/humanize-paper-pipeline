import re
from typing import Dict, List, Tuple


class AdversarialRewriter:
    """
    启发式对抗改写规则集 - 基于困惑度概念的对抗性策略
    
    这些规则模拟了困惑度检测的工作原理:
    - 困惑度衡量文本的"意外程度"，AI文本通常困惑度低（太"平滑"）
    - 人类写作有更多变化、意外和不确定性
    
    本模块通过启发式规则来对抗AI特征，增加文本的不确定性和多样性
    """

    ZH_ADVERSARIAL_PATTERNS = [
        (r'随着([\u4e00-\u9fa5]{2,10})的([\u4e00-\u9fa5]{2,10})', r'\1之后，\2'),
        (r'基于([\u4e00-\u9fa5]{2,10})的([\u4e00-\u9fa5]{2,10})', r'使用\1的\2'),
        (r'通过([\u4e00-\u9fa5]{2,10})的([\u4e00-\u9fa5]{2,10})', r'\1后，\2'),
        (r'利用([\u4e00-\u9fa5]{2,10})进行([\u4e00-\u9fa5]{2,10})', r'\1后，\2'),
        (r'旨在', '为了'),
        (r'总体来看', ''),
        (r'总体而言', ''),
        (r'整体来看', ''),
        (r'综上所述', ''),
        (r'总而言之', ''),
        (r'值得注意的是', ''),
        (r'需要强调的是', ''),
        (r'具有重要意义', '有意义'),
        (r'具有重要的现实意义', '有实际价值'),
        (r'具有重要的理论意义', '有理论价值'),
        (r'发挥着重要作用', '起到作用'),
        (r'发挥着至关重要的作用', '起到关键作用'),
        (r'不可或缺', '必要'),
        (r'至关重要', '关键'),
        (r'举足轻重', '重要'),
        (r'显而易见', '明显'),
        (r'毋庸置疑', '无需置疑'),
        (r'与此同时', '同时'),
        (r'此外', '另外'),
        (r'不仅如此', '而且'),
        (r'进一步说', '而且'),
        (r'换言之', '也就是说'),
        (r'简而言之', '简单说'),
        (r'核心痛点', '主要问题'),
        (r'技术方案', '解决方案'),
        (r'良好平衡', '平衡'),
        (r'强有力的支撑', '有力支撑'),
        (r'坚实的基础', '基础'),
        (r'高效', '有效'),
        (r'显著', '明显'),
        (r'优异', '好'),
        (r'本文将重点研究', '本文研究'),
        (r'本文旨在', ''),
        (r'本文拟', '本文'),
        (r'取得了良好的效果', '有效果'),
        (r'取得了显著的效果', '效果明显'),
        (r'为...提供了理论支撑', '支撑了...理论'),
        (r'为...指明了方向', '指明了...方向'),
        (r'推动了...的发展', '促进了...发展'),
        (r'实现了...的良好平衡', '平衡了...'),
        (r'提供了...的技术方案', '提出了...方案'),
    ]

    EN_ADVERSARIAL_PATTERNS = [
        (r'\bit is important to note that\b', 'notably'),
        (r'\bit should be noted that\b', 'notably'),
        (r'\bit is worth noting that\b', 'notably'),
        (r'\bin addition\b', 'also'),
        (r'\bfurthermore\b', 'also'),
        (r'\bmoreover\b', 'besides'),
        (r'\badditionally\b', 'plus'),
        (r'\bplays an important role\b', 'is significant'),
        (r'\bplays a crucial role\b', 'is key'),
        (r'\bplays a vital role\b', 'is vital'),
        (r'\bin terms of\b', 'regarding'),
        (r'\bwith regard to\b', 'about'),
        (r'\bwith respect to\b', 'about'),
        (r'\bvarious aspects\b', 'multiple factors'),
        (r'\bmultiple factors\b', 'several elements'),
        (r'\bit can be seen that\b', ''),
        (r'\bit has been shown that\b', ''),
        (r'\bto summarize\b', ''),
        (r'\bin conclusion\b', ''),
        (r'\boverall\b', ''),
        (r'\btaken together\b', ''),
        (r'\bin essence\b', ''),
        (r'\bnotably\b', ''),
        (r'\bsignificantly\b', ''),
        (r'\bthe present study\b', 'this study'),
        (r'\bthis paper (?:has|will have|aims to|seeks to)\b', 'this paper'),
        (r'\bleverage\b', 'use'),
        (r'\butilize\b', 'use'),
        (r'\bfoster\b', 'build'),
        (r'\bharness\b', 'use'),
        (r'\bstreamline\b', 'simplify'),
        (r'\brevolutionize\b', 'change'),
        (r'\bseamlessly\b', 'smoothly'),
        (r'\brobust\b', 'strong'),
        (r'\bseamless integration\b', 'easy integration'),
        (r'\bcomprehensive\b', 'thorough'),
        (r'\bpivotal\b', 'key'),
        (r'\bmeticulous\b', 'careful'),
        (r'\bunderscore\b', 'show'),
        (r'\bempower\b', 'enable'),
        (r'\bsynergy\b', 'teamwork'),
        (r'\bat its core\b', 'basically'),
        (r'\bin the realm of\b', 'in'),
        (r'\btapestry\b', 'mix'),
        (r'\bparadigm shift\b', 'major change'),
        (r'\bgame-changer\b', 'major change'),
    ]

    ZH_HEDGE_WORDS = [
        '似乎', '似乎在', '似乎表明', '似乎说明',
        '可能表明', '可能说明', '可能意味着',
        '或许', '或许可以', '或可', '或可为',
        '潜在地', '倾向于',
        '在一定程度上', '在某种程度上', '某种程度上',
        '有可能', '不排除', '不能完全排除',
        '可以认为', '可以推测', '可以推断',
        '初步表明', '初步显示',
    ]

    ZH_GENUINE_HEDGES = [
        '有待进一步', '尚需验证', '仍需探讨', '需进一步研究',
        '仍需深入', '有待考证', '尚待探讨'
    ]

    def __init__(self, lang: str = 'zh'):
        self.lang = lang
        self.patterns = self.ZH_ADVERSARIAL_PATTERNS if lang == 'zh' else self.EN_ADVERSARIAL_PATTERNS
        self.hedge_words = self.ZH_HEDGE_WORDS if lang == 'zh' else []
        self.genuine_hedges = self.ZH_GENUINE_HEDGES if lang == 'zh' else []

    def apply_adversarial_rewrite(self, text: str) -> Tuple[str, List[str]]:
        """
        应用对抗性改写规则
        
        Returns:
            Tuple of (rewritten_text, list_of_changes)
        """
        changes = []
        rewritten = text

        for pattern, replacement in self.patterns:
            new_text = re.sub(pattern, replacement, rewritten)
            if new_text != rewritten:
                matches = re.findall(pattern, rewritten)
                for match in matches[:3]:
                    if isinstance(match, tuple):
                        changes.append(f"'{match[0]}...' -> '{replacement.replace(r'\1', match[0]).replace(r'\2', match[1])}'")
                    else:
                        changes.append(f"'{match}' -> '{replacement}'")
                rewritten = new_text

        rewritten = self._reduce_hedging(rewritten, changes)

        rewritten = self._add_sentence_variation(rewritten, changes)

        rewritten = self._eliminate_word_repetition(rewritten, changes)

        return rewritten, changes

    def _reduce_hedging(self, text: str, changes: List[str]) -> str:
        """减少过度对冲词，保留真实学术对冲"""
        if self.lang != 'zh':
            return text

        for hedge in self.hedge_words:
            count = text.count(hedge)
            if count > 0:
                if hedge in ['似乎', '似乎表明', '可能表明']:
                    text = text.replace(hedge, '表明', 1)
                    if count > 1:
                        changes.append(f"减少对冲词: {hedge} ({count}处)")
                elif hedge in ['在一定程度上', '在某种程度上']:
                    text = text.replace(hedge, '', 1)
                    changes.append(f"删除过度对冲: {hedge}")

        return text

    def _add_sentence_variation(self, text: str, changes: List[str]) -> str:
        """
        增加句子变化 - 模拟高困惑度的人类写作风格
        
        策略:
        1. 插入短句打断均匀节奏
        2. 使用不同句式开头
        3. 添加口语化插入语
        """
        sentences = re.split(r'([。！？.!?]+)', text)
        if len(sentences) < 4:
            return text

        result = []
        for i, sent in enumerate(sentences):
            if i % 3 == 1 and len(sent.strip()) > 20:
                if sent.strip()[-1] in '。！？.!?':
                    if self.lang == 'zh':
                        prefix_options = ['其实，', '不过，', '说起来，', '有意思的是，']
                        prefix = prefix_options[i % len(prefix_options)]
                        result.append(prefix)
                        changes.append(f"添加口语化前缀: '{prefix.strip()}')")
                    else:
                        prefix_options = ['Actually, ', 'Well, ', 'Interestingly, ', 'The thing is, ']
                        prefix = prefix_options[i % len(prefix_options)]
                        result.append(prefix)
                        changes.append(f"Added colloquial prefix: '{prefix.strip()}'")
            result.append(sent)

        return ''.join(result)

    def _eliminate_word_repetition(self, text: str, changes: List[str]) -> str:
        """
        消除高频词重复 - 模拟低困惑度的反面
        
        AI倾向于重复使用相同的高频词组
        这个方法检测并替换过度重复的表达
        """
        if self.lang == 'zh':
            char_segments = re.findall(r'[\u4e00-\u9fa5]{4,}', text)
            if not char_segments:
                return text

            bigrams = []
            for seg in char_segments:
                for i in range(len(seg) - 1):
                    bigrams.append(seg[i:i+2])

            if not bigrams:
                return text

            from collections import Counter
            counter = Counter(bigrams)
            repeated = [(bg, cnt) for bg, cnt in counter.items() if cnt >= 4]

            for bigram, count in repeated[:3]:
                pattern = bigram + bigram
                if pattern in text:
                    replacements = {
                        '的': ['用于', '属于', '关于'],
                        '是': ['为', '等于', '属于'],
                        '了': ['已完成', '已完成'],
                        '和': ['与', '及', '以及'],
                    }
                    char = bigram[0]
                    if char in replacements:
                        text = text.replace(pattern, replacements[char][0], 1)
                        changes.append(f"消除重复: '{pattern}' -> '{replacements[char][0]}'")

        return text

    def generate_feedback_from_detection(self, metrics: Dict, is_en: bool) -> str:
        """根据检测指标生成针对性的反馈提示"""
        feedback = []

        if not is_en:
            over_hedge = metrics.get('over_hedging', {})
            if over_hedge.get('count', 0) >= 3:
                feedback.append('【对冲词过多】"似乎"、"可能表明"、"在一定程度上"等词是AI模仿学术谨慎的典型痕迹。请替换为更直接确定的表述，或使用"有待验证"、"尚需探讨"等真正的人类学术表达。')

            suizhe = metrics.get('suizhe_template', {})
            if suizhe.get('count', 0) >= 2:
                feedback.append('【模板句式】"随着...的..."、"基于...的..."等句式是AI论文的标志性特征。请改为更自然的表达，如"X之后，Y..."。')

            idiom = metrics.get('idiom_overuse', {})
            if idiom.get('count', 0) >= 4:
                feedback.append('【成语堆砌】AI生成文本喜欢堆砌成语。请将多余的四字词替换为平实表述。')

            ttr = metrics.get('bigram_ttr', {})
            if ttr.get('value', 1.0) < 0.65:
                feedback.append('【词汇重复】高频词重复过多。请更换近义词，打破固定的词汇组合模式。')

            clause = metrics.get('clause_chain_density', {})
            if clause.get('value', 0) > 3.2:
                feedback.append('【从句嵌套】单句内从句嵌套过多。请用句号断句，减少逗号连接的长定语。')

            opening_rep = metrics.get('sentence_opening_repetition', {})
            if opening_rep.get('count', 0) >= 3:
                feedback.append(f'【句首重复】"{opening_rep.get("top_pattern", "")}"开头出现次数过多。请变换句首表达。')

        else:
            uni = metrics.get('sentence_uniformity', {})
            if uni.get('score', 0) > 0.3:
                feedback.append('Sentence length is too uniform. Mix short sentences (5-10 words) with long complex ones (25+ words).')

            trans = metrics.get('transition_overuse', {})
            if trans.get('count', 0) > 0:
                feedback.append(f'Overused mechanical transitions ({trans.get("count")} detected). Replace "Moreover", "Furthermore" with natural connectors.')

            abs_lang = metrics.get('abstract_language', {})
            if abs_lang.get('count', 0) > 0:
                feedback.append(f'Replace abstract phrases like "various aspects", "in terms of" with specific concepts.')

            hedge = metrics.get('hedging_overuse', {})
            if hedge.get('score', 0) > 0.3:
                feedback.append('Excessive hedging ("may suggest", "could indicate"). Use more direct academic language.')

        if not feedback:
            feedback.append('Continue varying sentence structures and replacing formulaic phrases.' if is_en
                         else '继续打散句子长度，替换公式化短语。')

        return '\n'.join(feedback)


class PerplexitySurrogate:
    """
    困惑度代理检测器 - 轻量级启发式方法
    
    不运行完整的LLM困惑度计算，而是在CPU上模拟其效果:
    1. 词汇分布均匀度 (类似困惑度的词汇维度)
    2. 句子开始词的概率分布
    3. 字符级熵分析
    
    这些方法捕捉AI文本"过于平滑"的特征
    """

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def calculate_surrogate_perplexity(self, text: str) -> Dict:
        """
        计算困惑度代理分数
        
        Returns:
            Dict with 'surrogate_ppl', 'details', 'interpretation'
        """
        import math
        from collections import Counter

        if not text.strip():
            return {'surrogate_ppl': 0, 'interpretation': 'empty text'}

        metrics = {}

        vocab_entropy = self._vocabulary_entropy(text)
        metrics['vocab_entropy'] = vocab_entropy

        char_entropy = self._character_entropy(text)
        metrics['char_entropy'] = char_entropy

        word_burstiness = self._word_burstiness(text)
        metrics['word_burstiness'] = word_burstiness

        ngram_predictability = self._ngram_predictability(text)
        metrics['ngram_predictability'] = ngram_predictability

        normalized_ppl = self._normalize_surrogate_score(metrics)
        metrics['surrogate_ppl'] = normalized_ppl

        if normalized_ppl < 30:
            interpretation = 'Low perplexity surrogate - text appears too "smooth", likely AI-generated'
        elif normalized_ppl < 50:
            interpretation = 'Moderate perplexity surrogate - some AI indicators present'
        else:
            interpretation = 'Higher perplexity surrogate - text shows natural variation, likely human-written'

        metrics['interpretation'] = interpretation

        return metrics

    def _vocabulary_entropy(self, text: str) -> float:
        """词汇熵 - 衡量词汇分布的均匀程度"""
        import math
        from collections import Counter

        if self.lang == 'zh':
            words = re.findall(r'[\u4e00-\u9fa5]+', text)
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(words) < 5:
            return 0

        counter = Counter(words)
        total = len(words)

        entropy = 0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        normalized_entropy = entropy / math.log2(min(len(counter), 100)) if len(counter) > 1 else 0

        return round(normalized_entropy, 4)

    def _character_entropy(self, text: str) -> float:
        """字符级熵 - 捕捉字符分布的规律性"""
        import math
        from collections import Counter

        if self.lang == 'zh':
            chars = re.findall(r'[\u4e00-\u9fa5]', text)
        else:
            chars = re.findall(r'[a-z]', text.lower())

        if len(chars) < 20:
            return 0

        counter = Counter(chars)
        total = len(chars)

        entropy = 0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return round(entropy, 4)

    def _word_burstiness(self, text: str) -> float:
        """
        词汇突发性 - 模拟困惑度概念
        
        AI文本倾向于在局部重复相同的词（burstiness低）
        人类文本词汇分布更随机
        """
        import statistics
        from collections import Counter

        if self.lang == 'zh':
            words = re.findall(r'[\u4e00-\u9fa5]+', text)
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(words) < 10:
            return 0

        counter = Counter(words)
        counts = list(counter.values())

        if not counts:
            return 0

        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0

        if mean_count == 0:
            return 0

        cv = std_count / mean_count

        burstiness = min(1.0, cv / 2.0)

        return round(burstiness, 4)

    def _ngram_predictability(self, text: str) -> float:
        """
        N-gram可预测性 - 模拟困惑度的局部特征
        
        高可预测性 = 低困惑度 = AI特征
        """
        from collections import Counter

        if self.lang == 'zh':
            tokens = re.findall(r'[\u4e00-\u9fa5]', text)
        else:
            tokens = re.findall(r'[a-z]', text.lower())

        if len(tokens) < 10:
            return 0

        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]

        counter = Counter(bigrams)
        total_bigrams = len(bigrams)
        unique_bigrams = len(counter)

        ttr = unique_bigrams / total_bigrams if total_bigrams > 0 else 0

        high_freq_ratio = sum(1 for c in counter.values() if c >= 3) / len(counter) if counter else 0

        predictability = (1 - ttr) * high_freq_ratio

        return round(predictability, 4)

    def _normalize_surrogate_score(self, metrics: Dict) -> float:
        """
        将多个代理指标综合为困惑度代理分数

        分数范围: 0-100
        低分 = AI特征 (低困惑度/高可预测性)
        高分 = 人类特征 (高困惑度/低可预测性)
        """
        vocab_ent = metrics.get('vocab_entropy', 0.5)
        char_ent = metrics.get('char_entropy', 0)
        burst = metrics.get('word_burstiness', 0.5)
        pred = metrics.get('ngram_predictability', 0.5)

        vocab_weight = 0.25
        char_weight = 0.20
        burst_weight = 0.30
        pred_weight = 0.25

        score = (
            vocab_ent * vocab_weight * 100 +
            min(1.0, char_ent / 4.5) * char_weight * 100 +
            burst * burst_weight * 100 +
            (1 - pred) * pred_weight * 100
        )

        return round(score, 1)


def humanize_with_adversarial_rules(text: str, lang: str = 'zh', metrics: Dict = None) -> str:
    """
    使用对抗性规则对文本进行人类化处理

    这个函数封装了对抗性改写流程:
    1. 应用启发式替换规则
    2. 增加句子变化
    3. 减少词汇重复
    4. 基于检测指标调整
    """
    rewriter = AdversarialRewriter(lang=lang)

    if metrics:
        feedback = rewriter.generate_feedback_from_detection(metrics, is_en=(lang == 'en'))

    rewritten, changes = rewriter.apply_adversarial_rewrite(text)

    return rewritten
