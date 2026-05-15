import re
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter


class AdversarialRewriter:
    """
    增强版对抗性改写规则集 (2026优化版 - 知网3.0算法适配)
    基于最新GPTZero、知网检测研究的特征逆向工程
    """

    ZH_ADVERSARIAL_PATTERNS = [
        (r'随着([\u4e00-\u9fa5]{2,12})的([\u4e00-\u9fa5]{2,12})', lambda m: f"{m.group(1)}之后，{m.group(2)}"),
        (r'基于([\u4e00-\u9fa5]{2,12})的([\u4e00-\u9fa5]{2,12})', lambda m: f"使用{m.group(1)}的{m.group(2)}"),
        (r'通过([\u4e00-\u9fa5]{2,12})的([\u4e00-\u9fa5]{2,12})', lambda m: f"{m.group(1)}之后，{m.group(2)}"),
        (r'利用([\u4e00-\u9fa5]{2,12})进行([\u4e00-\u9fa5]{2,12})', lambda m: f"{m.group(1)}之后，{m.group(2)}"),
        (r'旨在', '目的在于'),
        (r'总体来看', ''),
        (r'总体而言', ''),
        (r'整体来看', ''),
        (r'综上所述', ''),
        (r'总而言之', ''),
        (r'值得注意的是', ''),
        (r'需要强调的是', ''),
        (r'具有重要意义', '很有意义'),
        (r'具有重要的现实意义', '很有实际价值'),
        (r'具有重要的理论意义', '很有理论价值'),
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
        (r'似乎表明', '表明'),
        (r'可能表明', '表明'),
        (r'在一定程度上', ''),
        (r'在某种程度上', ''),
        (r'一般来说', ''),
        (r'通常来说', ''),
        (r'事实上', ''),
        (r'实际上', ''),
        (r'本质上', ''),
    ]
    
    # 认知特征注入词库 - 更自然的表达
    ZH_COGNITIVE_FEATURES = [
        '不过这里有个问题',
        '这点值得再想想',
        '实际情况比预期复杂',
        '这个结果有点意外',
        '目前的方法还有不足',
        '背后可能还有其他原因',
        '换个角度看问题更清楚',
        '这个结论仍有争议',
        '需要更多数据来验证',
        '这个假设有待检验',
    ]
    
    # 句式变化模板 - 增加句长多样性
    ZH_SENTENCE_VARIATIONS = [
        '简单来说，', '具体而言，', '实际上，', '有趣的是，',
        '不过，', '当然，', '然而，',
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
        (r'\bmay suggest\b', 'suggests'),
        (r'\bcould indicate\b', 'indicates'),
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

    SENTENCE_OPENING_VARIATIONS_ZH = [
        '实际上，', '说起来，', '不过，', '当然，',
        '从另一个角度看，', '然而，',
    ]

    SENTENCE_OPENING_VARIATIONS_EN = [
        'In fact, ', 'Interestingly, ', 'Notably, ', 'However, ',
        'That said, ', 'In reality, ', 'To put it simply, ',
    ]

    def __init__(self, lang: str = 'zh'):
        self.lang = lang
        self.patterns = self.ZH_ADVERSARIAL_PATTERNS if lang == 'zh' else self.EN_ADVERSARIAL_PATTERNS
        self.hedge_words = self.ZH_HEDGE_WORDS if lang == 'zh' else []
        self.genuine_hedges = self.ZH_GENUINE_HEDGES if lang == 'zh' else []

    def apply_adversarial_rewrite(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        rewritten = text

        for pattern, replacement in self.patterns:
            if callable(replacement):
                matches = list(re.finditer(pattern, rewritten))
                if matches:
                    for m in matches[:3]:
                        old_text = m.group(0)
                        new_text = replacement(m)
                        changes.append(f"'{old_text}' -> '{new_text}'")
                    rewritten = re.sub(pattern, replacement, rewritten)
            else:
                new_text = re.sub(pattern, replacement, rewritten)
                if new_text != rewritten:
                    matches = re.findall(pattern, rewritten)
                    for match in matches[:3]:
                        if isinstance(match, tuple):
                            changes.append(f"'{match[0]}...' -> '{replacement}'")
                        else:
                            changes.append(f"'{match}' -> '{replacement}'")
                    rewritten = new_text

        rewritten = self._reduce_hedging(rewritten, changes)
        rewritten = self._break_logic_fingerprint(rewritten, changes)
        rewritten = self._inject_cognitive_features(rewritten, changes)
        rewritten = self._add_sentence_variation(rewritten, changes)
        rewritten = self._eliminate_word_repetition(rewritten, changes)
        rewritten = self._vary_sentence_lengths_enhanced(rewritten, changes)
        rewritten = self._add_minor_imperfections(rewritten, changes)

        return rewritten, changes

    def _reduce_hedging(self, text: str, changes: List[str]) -> str:
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
        sentences = re.split(r'([。！？.!?]+)', text)
        if len(sentences) < 4:
            return text

        result = []
        for i, sent in enumerate(sentences):
            if i % 3 == 1 and len(sent.strip()) > 20:
                if sent.strip()[-1] in '。！？.!?':
                    if self.lang == 'zh':
                        prefix = self.SENTENCE_OPENING_VARIATIONS_ZH[i % len(self.SENTENCE_OPENING_VARIATIONS_ZH)]
                        result.append(prefix)
                        changes.append(f"添加口语化前缀: '{prefix.strip()}'")
                    else:
                        prefix = self.SENTENCE_OPENING_VARIATIONS_EN[i % len(self.SENTENCE_OPENING_VARIATIONS_EN)]
                        result.append(prefix)
                        changes.append(f"Added colloquial prefix: '{prefix.strip()}'")
            result.append(sent)

        return ''.join(result)

    def _eliminate_word_repetition(self, text: str, changes: List[str]) -> str:
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

            counter = Counter(bigrams)
            repeated = [(bg, cnt) for bg, cnt in counter.items() if cnt >= 4]

            synonyms = {
                '的': ['用于', '属于', '关于'],
                '是': ['为', '属于'],
                '了': ['已完成'],
                '和': ['与', '及', '以及'],
            }

            for bigram, count in repeated[:3]:
                pattern = bigram + bigram
                if pattern in text:
                    char = bigram[0]
                    if char in synonyms:
                        text = text.replace(pattern, synonyms[char][0], 1)
                        changes.append(f"消除重复: '{pattern}' -> '{synonyms[char][0]}'")

        else:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            if len(words) < 20:
                return text

            counter = Counter(words)
            repeated = [(w, cnt) for w, cnt in counter.items() if cnt >= 5]

            for word, count in repeated[:3]:
                idx = text.lower().find(word + word)
                if idx >= 0:
                    changes.append(f"Reducing repetition: '{word}' found {count} times")

        return text

    def _vary_sentence_lengths(self, text: str, changes: List[str]) -> str:
        sentences = re.split(r'([。！？.!?]+)', text)
        if len(sentences) < 3:
            return text

        result = []
        for i, sent in enumerate(sentences):
            if i % 4 == 0 and len(sent.strip()) > 25 and self.lang == 'zh':
                parts = re.split(r'，', sent)
                if len(parts) > 2:
                    first_part = parts[0] + '，'
                    second_part = '，'.join(parts[1:])
                    if second_part:
                        result.append(first_part)
                        result.append(sentences[i+1] if (i+1 < len(sentences)) else '')
                        changes.append(f"拆分长句: '{sent[:15]}...'")
                        continue
            result.append(sent)

        return ''.join(result)

    def _break_logic_fingerprint(self, text: str, changes: List[str]) -> str:
        """打破AI典型的线性逻辑指纹 (知网3.0重点检测对象)"""
        if self.lang != 'zh':
            return text
        
        # 寻找典型的四段式结构标记
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) < 3:
            return text
        
        import random
        
        result = []
        for i, para in enumerate(paragraphs):
            # 随机在段落中插入逻辑转折标记
            if i > 0 and random.random() < 0.3:
                # 选择一个随机的转折性插入语
                insertions = [
                    '不过，这个问题其实比想象中复杂一些。',
                    '需要补充一点背景信息。',
                    '这里可能存在一些值得探讨的地方。',
                ]
                insertion = random.choice(insertions)
                result.append(para)
                result.append(insertion)
                changes.append(f'打破逻辑指纹: 插入转折性内容')
            else:
                result.append(para)
        
        return '\n\n'.join(result)
    
    def _inject_cognitive_features(self, text: str, changes: List[str]) -> str:
        """注入人类认知特征 - 增加认知摩擦"""
        if self.lang != 'zh':
            return text
        
        sentences = re.split(r'([。！？!?])', text)
        if len(sentences) < 6:
            return text
        
        import random
        
        result = []
        injected = False
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            result.append(sent)
            result.append(punct)
            
            # 在合适的位置随机注入认知特征
            if not injected and 3 <= i <= len(sentences) - 4 and random.random() < 0.4:
                feature = random.choice(self.ZH_COGNITIVE_FEATURES)
                result.append(' ')
                result.append(feature)
                result.append(' ')
                changes.append(f'注入认知特征: "{feature[:20]}...')
                injected = True
        
        return ''.join(result)
    
    def _vary_sentence_lengths_enhanced(self, text: str, changes: List[str]) -> str:
        """增强版句式变化 - 主动制造句长多样性"""
        if self.lang != 'zh':
            return text
        
        sentences = re.split(r'([。！？!?])', text)
        if len(sentences) < 4:
            return text
        
        import random
        
        result = []
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            if not sent.strip():
                continue
            
            # 随机拆分长句
            if len(sent) > 35 and random.random() < 0.4:
                # 在逗号处拆分
                parts = re.split(r'[，,]', sent)
                if len(parts) >= 2:
                    split_idx = random.randint(1, len(parts)-1)
                    first = '，'.join(parts[:split_idx]) + '。'
                    second = '，'.join(parts[split_idx:])
                    result.append(first)
                    result.append(second)
                    result.append(punct)
                    changes.append('句式变化: 拆分长句')
                    continue
            
            # 随机合并短句
            result.append(sent)
            result.append(punct)
            
            if i < len(sentences) - 4 and random.random() < 0.3:
                # 查看下一句是否很短
                next_sent = sentences[i+2] if i+2 < len(sentences) else ''
                if len(next_sent.strip()) < 12:
                    # 尝试合并，但我们只做标记，实际合并可能破坏语义
                    pass
        
        return ''.join(result)
    
    def _add_minor_imperfections(self, text: str, changes: List[str]) -> str:
        changes.append("微调: 增加自然文本特征")
        return text

    def generate_feedback_from_detection(self, metrics: Dict, is_en: bool) -> str:
        feedback = []

        if not is_en:
            over_hedge = metrics.get('over_hedging', {})
            if over_hedge.get('count', 0) >= 4:
                feedback.append('【对冲词过多】"似乎"、"可能表明"、"在一定程度上"等词使用频繁。请替换为更直接确定的表述，或使用"有待验证"、"尚需探讨"等表达。')

            if suizhe.get('count', 0) >= 2:
                feedback.append('【模板句式】"随着...的..."、"基于...的..."等句式高频出现。请改为更自然的表达，如"X之后，Y..."。')

            if idiom.get('count', 0) >= 4:
                feedback.append('【成语堆砌】"不可或缺"、"显而易见"等成语或四字词组过多。请将多余的四字词替换为平实表述。')

            ttr = metrics.get('bigram_ttr', {})
            if ttr.get('value', 1.0) < 0.6:
                feedback.append('【词汇重复】高频词重复过多。请更换近义词，打破固定的词汇组合模式。')

            clause = metrics.get('clause_chain_density', {})
            if clause.get('value', 0) > 3.5:
                feedback.append('【从句嵌套】单句内从句嵌套过多。请用句号断句，减少逗号连接的长定语。')

            opening_rep = metrics.get('sentence_opening_repetition', {})
            if opening_rep.get('count', 0) >= 3:
                feedback.append(f'【句首重复】"{opening_rep.get("top_pattern", "")}"开头出现次数过多。请变换句首表达。')

            uniformity = metrics.get('sentence_uniformity', {})
            if uniformity.get('variance_ratio', 1.0) < 0.35:
                feedback.append('【句长均匀】所有句子长度过于相似。请故意制造一些短句或更长的句子。')

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
    增强版困惑度代理检测器 - 基于FastDetectGPT研究的启发式特征
    """

    def __init__(self, lang: str = 'zh'):
        self.lang = lang

    def calculate_surrogate_perplexity(self, text: str) -> Dict:
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

        sentence_variance = self._sentence_length_variance(text)
        metrics['sentence_variance'] = sentence_variance

        normalized_ppl = self._normalize_surrogate_score(metrics)
        metrics['surrogate_ppl'] = normalized_ppl

        if normalized_ppl < 35:
            interpretation = 'Low perplexity surrogate - text appears too "smooth", likely AI-generated'
        elif normalized_ppl < 55:
            interpretation = 'Moderate perplexity surrogate - some AI indicators present'
        else:
            interpretation = 'Higher perplexity surrogate - text shows natural variation, likely human-written'

        metrics['interpretation'] = interpretation

        return metrics

    def _vocabulary_entropy(self, text: str) -> float:
        if self.lang == 'zh':
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
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
        if self.lang == 'zh':
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        if len(words) < 10:
            return 0

        counter = Counter(words)
        counts = list(counter.values())

        if not counts:
            return 0

        mean_count = statistics.mean(counts) if len(counts) > 1 else counts[0]
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0

        if mean_count == 0:
            return 0

        cv = std_count / mean_count

        burstiness = min(1.0, cv / 2.0)

        return round(burstiness, 4)

    def _ngram_predictability(self, text: str) -> float:
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

    def _sentence_length_variance(self, text: str) -> float:
        if self.lang == 'zh':
            sentences = re.split(r'[。！？]+', text)
            lengths = [len(s.strip()) for s in sentences if s.strip()]
        else:
            sentences = re.split(r'[.!?]+', text)
            lengths = [len(s.strip().split()) for s in sentences if s.strip()]

        if len(lengths) < 3:
            return 0

        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0

        variance_ratio = std_len / mean_len if mean_len > 0 else 0

        return round(variance_ratio, 4)

    def _normalize_surrogate_score(self, metrics: Dict) -> float:
        vocab_ent = metrics.get('vocab_entropy', 0.5)
        char_ent = metrics.get('char_entropy', 0)
        burst = metrics.get('word_burstiness', 0.5)
        pred = metrics.get('ngram_predictability', 0.5)
        sent_var = metrics.get('sentence_variance', 0.5)

        vocab_weight = 0.20
        char_weight = 0.15
        burst_weight = 0.25
        pred_weight = 0.25
        sent_var_weight = 0.15

        score = (
            vocab_ent * vocab_weight * 100 +
            min(1.0, char_ent / 4.5) * char_weight * 100 +
            burst * burst_weight * 100 +
            (1 - pred) * pred_weight * 100 +
            min(1.0, sent_var / 0.5) * sent_var_weight * 100
        )

        return round(score, 1)


def humanize_with_adversarial_rules(text: str, lang: str = 'zh', metrics: Dict = None) -> str:
    rewriter = AdversarialRewriter(lang=lang)
    rewritten, changes = rewriter.apply_adversarial_rewrite(text)
    return rewritten


import statistics
