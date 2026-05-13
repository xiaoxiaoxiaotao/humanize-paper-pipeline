"""
中文AI文本检测器 - 知网3.0算法适配
包含维普、知网等平台的常见检测特征
"""

import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Optional

from .base_detector import BaseDetector


class ChineseDetector(BaseDetector):
    """
    中文AI文本检测器
    集成知网3.0、万方等平台的最新检测特征
    """

    AI_TRANSITIONS = [
        '首先', '其次', '再次', '最后', '综上所述',
        '值得注意的是', '需要强调的是', '进一步而言',
        '总体来说', '一般而言', '事实上',
        '随着', '基于', '通过', '利用',
        '然而', '因此', '此外', '另外', '同时',
        '从而', '进而', '与此同时', '不仅如此',
        '进一步说', '换言之', '简而言之',
        '本文', '本研究', '本文旨在', '本文拟', '总之', '总而言之',
        '由此可见', '显然', '毫无疑问',
        '旨在', '可分为', '大致可分为', '主要包括', '具体包括',
        '催生了', '开创了', '分为', '替换为', '替换为了',
        '避免了', '此后', '在此基础上',
        '尽管', '虽然', '甚至', '首次', '逐步', '日益',
        '特别是', '尤其是', '如今', '目前',
        '极大', '极大地',
    ]

    ABSTRACT_PHRASES = [
        '具有重要意义', '发挥着重要作用', '至关重要',
        '不可或缺', '具有重要价值', '促进了发展',
        '综上所述', '基于此', '因此', '然而',
        '具有重要的现实意义', '具有重要的应用价值',
        '具有重大意义', '成为关键', '重要引擎',
        '提供了有力', '实现了良好', '推动了发展',
        '具有深远意义', '具有广阔前景',
        '提供了坚实', '奠定了坚实', '打下了坚实',
        '核心痛点', '技术方案', '强有力的支撑',
        '坚实的基础', '取得了良好的效果',
        '取得了显著的效果', '为...提供了',
        '为...指明了方向', '起到了关键作用',
        '起到了重要作用', '扮演着重要角色',
        '占据着重要地位', '具有不可替代的作用',
        '核心思想', '设计理念', '端到端',
        '吸引了大量关注', '引导了发展方向', '实现了超越',
        '带来了性能上的提升', '带来了提升',
        '不断引入', '逐步弥合',
        '成为首选方案', '首选方案',
        '日益逼近', '向着更加',
        '颠覆性的', '大量关注', '发展方向',
        '真正意义上的',
    ]

    TEMPLATE_PATTERNS = [
        r'随着[^，。]{2,40}的[^，。]{2,40}',
        r'基于[^，。]{2,40}的[^，。]{2,40}',
        r'通过[^，。]{2,40}的[^，。]{2,40}',
        r'因此[^，。]{2,40}具有重要意义',
        r'利用[^，。]{2,40}的[^，。]{2,40}',
        r'旨在[^，。]{2,40}',
        r'大致可分为[^，。]{2,40}',
        r'可分为[^，。]{2,20}：',
        r'以下[^，。]{2,30}阐述',
        r'是[^，。]{2,30}之一',
        r'为[^，。]{2,30}提供了[^，。]{2,40}',
        r'[^，。]{2,20}等工作[^，。]{2,30}',
        r'催生了[^，。]{2,40}',
        r'开创了[^，。]{2,20}先河',
        r'为了解决[^，。]{2,30}',
        r'核心[^，。]{2,10}是[^，。]{2,40}',
        r'极大地[^，。]{2,40}',
        r'带来了[^，。]{2,30}',
        r'尽管[^，。]{2,30}但[^，。]{2,30}',
        r'首次提出了[^，。]{2,40}',
        r'成为[^，。]{2,20}的[^，。]{2,20}',
        r'推动着[^，。]{2,30}',
        r'在[^，。]{2,15}上存在[^，。]{2,20}',
        r'特别是[^，。]{2,30}',
        r'日益[^，。]{2,20}',
        r'真正意义上的[^，。]{2,20}',
    ]

    HEDGE_WORDS = [
        '似乎', '似乎在', '似乎表明', '似乎说明',
        '可能表明', '可能说明', '可能意味着',
        '或许', '或许可以', '或可', '或可为',
        '潜在地', '倾向于',
        '在一定程度上', '在某种程度上', '某种程度上',
        '有可能', '不排除', '不能完全排除',
        '可以认为', '可以推测', '可以推断',
        '初步表明', '初步显示',
    ]

    CONCLUDING_FORMULAS = [
        '综上所述', '总而言之', '总之', '由此可见',
        '因此，本研究', '因此，本文', '综上',
        '通过以上分析', '基于以上分析',
        '从以上分析可以看出', '以上结果表明',
    ]

    IDIOM_PATTERNS = [
        '不可或缺', '至关重要', '举足轻重', '显而易见',
        '毋庸置疑', '不言而喻', '众所周知',
        '日新月异', '突飞猛进', '蓬勃发展',
        '相辅相成', '密不可分', '息息相关',
        '行之有效', '卓有成效', '立竿见影',
        '层出不穷', '应运而生', '势在必行',
        '先河',
    ]

    PARAGRAPH_TEMPLATE_MARKERS = [
        '随着', '近年来', '当前', '目前', '现阶段',
        '然而', '但是', '因此', '所以',
        '本文', '本研究', '本课题',
    ]

    def __init__(self):
        super().__init__(name="Chinese AI Detector")

    def detect(self, text: str) -> Tuple[int, Dict]:
        details = {'metrics': {}, 'platform': '知网3.0'}
        score = 0

        sentences = self.split_sentences(text, chinese=True)
        paragraphs = self.split_paragraphs(text)

        if len(sentences) < 2 or not text.strip():
            return 0, details

        score, details = self._analyze_sentence_length_distribution(sentences, score, details)
        score, details = self._analyze_paragraph_structure_similarity(paragraphs, sentences, score, details)
        score, details = self._analyze_info_density_distribution(sentences, score, details)
        score, details = self._analyze_transition_word_distribution(text, sentences, score, details)

        score, details = self._analyze_sentence_uniformity(sentences, score, details)
        score, details = self._analyze_transition_overuse(text, score, details)
        score, details = self._analyze_abstract_language(text, score, details)
        score, details = self._analyze_sentence_opening_repetition(sentences, score, details)
        score, details = self._analyze_templates(text, score, details)
        score, details = self._analyze_word_burstiness(text, score, details)

        score, details = self._analyze_over_hedging(text, score, details)
        score, details = self._analyze_bigram_ttr(text, score, details)
        score, details = self._analyze_clause_chain_density(text, sentences, score, details)
        score, details = self._analyze_idiom_overuse(text, score, details)
        score, details = self._analyze_punctuation_density(text, score, details)
        score, details = self._analyze_concluding_formula(text, score, details)
        score, details = self._analyze_suizhe_template(text, score, details)
        score, details = self._analyze_paragraph_template(paragraphs, score, details)
        score, details = self._analyze_definition_pattern(text, score, details)
        score, details = self._analyze_em_dash_overuse(text, score, details)
        score, details = self._analyze_citation_distribution(text, score, details)

        final_score = min(100, max(0, score))
        details['overall_score'] = final_score
        details['sentence_count'] = len(sentences)
        details['paragraph_count'] = len(paragraphs)

        return final_score, details

    def _analyze_sentence_length_distribution(self, sentences: List[str],
                                             score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        lengths = []
        for sent in sentences:
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if clean_sent:
                lengths.append(len(clean_sent))

        if len(lengths) < 3:
            return score, details

        avg_len = statistics.mean(lengths)
        std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
        cv = std_dev / avg_len if avg_len > 0 else 0

        ai_peak_count = sum(1 for l in lengths if 15 <= l <= 25)
        ai_peak_ratio = ai_peak_count / len(lengths)

        metric_score = 0
        if ai_peak_ratio > 0.7:
            metric_score += 15
        elif ai_peak_ratio > 0.55:
            metric_score += 8

        if cv < 0.25:
            metric_score += 12
        elif cv < 0.35:
            metric_score += 6

        details['metrics']['sentence_length_distribution'] = {
            'avg_length': round(avg_len, 1),
            'cv': round(cv, 3),
            'ai_peak_ratio': round(ai_peak_ratio, 2),
            'score': metric_score,
            'details': f'句长均值{avg_len:.1f}，CV={cv:.3f}，15-25字占比{ai_peak_ratio*100:.0f}%'
        }

        return score + metric_score, details

    def _analyze_paragraph_structure_similarity(self, paragraphs: List[str],
                                                sentences: List[str], score: int,
                                                details: Dict) -> Tuple[int, Dict]:
        if len(paragraphs) < 2:
            return score, details

        paragraph_features = []
        for para in paragraphs:
            para_sentences = [s.strip() for s in re.split(r'[。！？!?]+', para) if s.strip()]
            if para_sentences:
                sent_count = len(para_sentences)
                avg_len = sum(len(s) for s in para_sentences) / sent_count
                has_transition = any(t in para for t in self.AI_TRANSITIONS[:5])
                paragraph_features.append((sent_count, avg_len, has_transition))

        if len(paragraph_features) >= 2:
            similar_count = 0
            total_pairs = len(paragraph_features) * (len(paragraph_features) - 1) / 2

            for i in range(len(paragraph_features)):
                for j in range(i + 1, len(paragraph_features)):
                    sent_diff = abs(paragraph_features[i][0] - paragraph_features[j][0])
                    len_diff = abs(paragraph_features[i][1] - paragraph_features[j][1])
                    trans_same = paragraph_features[i][2] == paragraph_features[j][2]

                    if sent_diff <= 1 and len_diff < 8 and trans_same:
                        similar_count += 1

            similarity_ratio = similar_count / total_pairs if total_pairs > 0 else 0

            metric_score = 0
            if similarity_ratio >= 0.7:
                metric_score = 18
            elif similarity_ratio >= 0.55:
                metric_score = 10
            elif similarity_ratio >= 0.4:
                metric_score = 5

            details['metrics']['paragraph_structure_similarity'] = {
                'similarity_ratio': round(similarity_ratio, 2),
                'paragraph_count': len(paragraphs),
                'score': metric_score,
                'details': f'段落结构相似度{similarity_ratio*100:.0f}%'
            }

            return score + metric_score, details

        return score, details

    def _analyze_info_density_distribution(self, sentences: List[str],
                                           score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        info_densities = []
        for sent in sentences:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', sent))
            total_chars = len(sent)
            if total_chars > 0:
                info_densities.append(chinese_chars / total_chars)

        if len(info_densities) < 3:
            return score, details

        avg_density = statistics.mean(info_densities)
        std_density = statistics.stdev(info_densities) if len(info_densities) > 1 else 0

        in_ai_zone = sum(1 for d in info_densities if 0.65 <= d <= 0.75)
        ai_zone_ratio = in_ai_zone / len(info_densities)

        metric_score = 0
        if ai_zone_ratio > 0.85:
            metric_score += 15
        elif ai_zone_ratio > 0.7:
            metric_score += 8

        if std_density < 0.05:
            metric_score += 10
        elif std_density < 0.08:
            metric_score += 5

        details['metrics']['info_density_distribution'] = {
            'avg_density': round(avg_density, 2),
            'std_density': round(std_density, 3),
            'ai_zone_ratio': round(ai_zone_ratio, 2),
            'score': metric_score,
            'details': f'信息密度均值{avg_density*100:.0f}%，标准差{std_density:.3f}'
        }

        return score + metric_score, details

    def _analyze_transition_word_distribution(self, text: str, sentences: List[str],
                                              score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        transition_count = 0
        transition_positions = []

        for idx, sent in enumerate(sentences):
            for t in self.AI_TRANSITIONS:
                if t in sent:
                    transition_count += 1
                    transition_positions.append(idx)

        text_chars = len(text)
        density_per_1000 = (transition_count / text_chars) * 1000 if text_chars > 0 else 0

        uniformity_score = 0
        if len(transition_positions) >= 3:
            gaps = [transition_positions[i+1] - transition_positions[i]
                   for i in range(len(transition_positions)-1)]
            if gaps:
                avg_gap = statistics.mean(gaps)
                std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
                gap_cv = std_gap / avg_gap if avg_gap > 0 else 0
                uniformity_score = 1 - gap_cv

        metric_score = 0
        if density_per_1000 > 12:
            metric_score += 12
        elif density_per_1000 > 8:
            metric_score += 6

        if uniformity_score > 0.85:
            metric_score += 10
        elif uniformity_score > 0.7:
            metric_score += 5

        details['metrics']['transition_word_distribution'] = {
            'count': transition_count,
            'density_per_1000': round(density_per_1000, 1),
            'uniformity_score': round(uniformity_score, 2),
            'score': metric_score,
            'details': f'连接词{transition_count}个，密度{density_per_1000:.1f}/千字'
        }

        return score + metric_score, details

    def _analyze_sentence_uniformity(self, sentences: List[str], score: int,
                                     details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        variance_ratio = std_dev / avg_len if avg_len > 0 else 0

        metric_score = 0
        if avg_len < 30:
            if variance_ratio < 0.18:
                metric_score = 15
            elif variance_ratio < 0.25:
                metric_score = 10
            elif variance_ratio < 0.35:
                metric_score = 5
        else:
            if variance_ratio < 0.25:
                metric_score = 12
            elif variance_ratio < 0.35:
                metric_score = 7

        details['metrics']['sentence_uniformity'] = {
            'avg_length': round(avg_len, 1),
            'variance_ratio': round(variance_ratio, 3),
            'score': metric_score,
            'details': f'句长变异系数 {variance_ratio:.3f}'
        }

        return score + metric_score, details

    def _analyze_transition_overuse(self, text: str, score: int,
                                    details: Dict) -> Tuple[int, Dict]:
        transition_count = self.count_word_occurrences(text, self.AI_TRANSITIONS)

        metric_score = 0
        if transition_count > 8:
            metric_score = 15
        elif transition_count > 5:
            metric_score = 10
        elif transition_count > 2:
            metric_score = 5

        details['metrics']['transition_overuse'] = {
            'count': transition_count,
            'score': metric_score,
            'details': f'过渡词数量 {transition_count}'
        }

        return score + metric_score, details

    def _analyze_abstract_language(self, text: str, score: int,
                                  details: Dict) -> Tuple[int, Dict]:
        phrase_count = self.count_word_occurrences(text, self.ABSTRACT_PHRASES)

        metric_score = 0
        if phrase_count > 5:
            metric_score = 12
        elif phrase_count > 3:
            metric_score = 8

        details['metrics']['abstract_language'] = {
            'count': phrase_count,
            'score': metric_score,
            'details': f'抽象短语数量 {phrase_count}'
        }

        return score + metric_score, details

    def _analyze_sentence_opening_repetition(self, sentences: List[str], score: int,
                                            details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 4:
            return score, details

        openings = []
        for sent in sentences:
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if len(clean_sent) >= 4:
                openings.append(clean_sent[:4])

        if not openings:
            return score, details

        counter = Counter(openings)
        most_common = counter.most_common(1)
        top_pattern, top_count = most_common[0]

        metric_score = 0
        if top_count >= 3:
            metric_score = 12
        elif top_count >= 2:
            metric_score = 7

        details['metrics']['sentence_opening_repetition'] = {
            'top_pattern': top_pattern,
            'count': top_count,
            'score': metric_score,
            'details': f'最常见开头 "{top_pattern}" 出现 {top_count} 次'
        }

        return score + metric_score, details

    def _analyze_templates(self, text: str, score: int,
                          details: Dict) -> Tuple[int, Dict]:
        template_count = self.detect_repeated_patterns(text, self.TEMPLATE_PATTERNS)

        metric_score = 0
        if template_count >= 3:
            metric_score = 15
        elif template_count >= 2:
            metric_score = 10
        elif template_count >= 1:
            metric_score = 5

        details['metrics']['templates'] = {
            'count': template_count,
            'score': metric_score,
            'details': f'模板模式数量 {template_count}'
        }

        return score + metric_score, details

    def _analyze_word_burstiness(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        chars_segments = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        if len(chars_segments) <= 5:
            return score, details

        words = []
        for seg in chars_segments:
            for i in range(len(seg) - 1):
                words.append(seg[i:i+2])

        if len(words) <= 10:
            return score, details

        counter = Counter(words)
        counts = list(counter.values())

        if len(counts) <= 1:
            return score, details

        avg_count = statistics.mean(counts)
        std_count = statistics.stdev(counts)
        cv = std_count / avg_count if avg_count > 0 else 0

        metric_score = 0
        if cv < 0.6:
            metric_score = 10
        elif cv < 0.8:
            metric_score = 6

        details['metrics']['word_burstiness'] = {
            'cv': round(cv, 3),
            'score': metric_score,
            'details': f'词汇突发性 CV={cv:.3f}'
        }

        return score + metric_score, details

    def _analyze_over_hedging(self, text: str, score: int,
                             details: Dict) -> Tuple[int, Dict]:
        hedge_count = 0
        hedge_items = []
        for hedge in self.HEDGE_WORDS:
            count = text.count(hedge)
            if count > 0:
                hedge_count += count
                hedge_items.append((hedge, count))

        metric_score = 0
        if hedge_count >= 6:
            metric_score = 12
        elif hedge_count >= 4:
            metric_score = 8
        elif hedge_count >= 2:
            metric_score = 4

        details['metrics']['over_hedging'] = {
            'count': hedge_count,
            'items': hedge_items,
            'score': metric_score,
            'details': f'对冲词数量 {hedge_count}'
        }

        return score + metric_score, details

    def _analyze_bigram_ttr(self, text: str, score: int,
                           details: Dict) -> Tuple[int, Dict]:
        chars = re.findall(r'[\u4e00-\u9fa5]', text)
        if len(chars) < 10:
            return score, details

        bigrams = []
        for i in range(len(chars) - 1):
            bigrams.append(chars[i] + chars[i+1])

        if not bigrams:
            return score, details

        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        ttr = unique_bigrams / total_bigrams if total_bigrams > 0 else 1.0

        metric_score = 0
        if ttr < 0.45:
            metric_score = 12
        elif ttr < 0.55:
            metric_score = 8
        elif ttr < 0.65:
            metric_score = 4

        details['metrics']['bigram_ttr'] = {
            'value': round(ttr, 3),
            'unique': unique_bigrams,
            'total': total_bigrams,
            'score': metric_score,
            'details': f'Bigram TTR={ttr:.3f}'
        }

        return score + metric_score, details

    def _analyze_clause_chain_density(self, text: str, sentences: List[str],
                                      score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 2:
            return score, details

        total_commas = text.count('，') + text.count(',')
        total_periods = text.count('。') + text.count('.') + text.count('！') + text.count('？')

        comma_per_sentence = total_commas / len(sentences) if len(sentences) > 0 else 0

        metric_score = 0
        if comma_per_sentence > 4.0:
            metric_score = 12
        elif comma_per_sentence > 3.0:
            metric_score = 8
        elif comma_per_sentence > 2.0:
            metric_score = 4

        details['metrics']['clause_chain_density'] = {
            'value': round(comma_per_sentence, 2),
            'total_commas': total_commas,
            'score': metric_score,
            'details': f'从句链密度 {comma_per_sentence:.1f}逗号/句'
        }

        return score + metric_score, details

    def _analyze_idiom_overuse(self, text: str, score: int,
                              details: Dict) -> Tuple[int, Dict]:
        idiom_count = 0
        for idiom in self.IDIOM_PATTERNS:
            idiom_count += text.count(idiom)

        metric_score = 0
        if idiom_count >= 6:
            metric_score = 12
        elif idiom_count >= 4:
            metric_score = 8
        elif idiom_count >= 2:
            metric_score = 4

        details['metrics']['idiom_overuse'] = {
            'count': idiom_count,
            'score': metric_score,
            'details': f'成语/四字词数量 {idiom_count}'
        }

        return score + metric_score, details

    def _analyze_punctuation_density(self, text: str, score: int,
                                    details: Dict) -> Tuple[int, Dict]:
        comma_count = text.count('，') + text.count(',')
        period_count = text.count('。') + text.count('.')

        ratio = comma_count / period_count if period_count > 0 else 0

        metric_score = 0
        if ratio > 5.0:
            metric_score = 10
        elif ratio > 3.5:
            metric_score = 6
        elif ratio > 2.5:
            metric_score = 3

        details['metrics']['punctuation_density'] = {
            'comma_count': comma_count,
            'period_count': period_count,
            'comma_period_ratio': round(ratio, 2),
            'score': metric_score,
            'details': f'逗号/句号比 {ratio:.1f}'
        }

        return score + metric_score, details

    def _analyze_concluding_formula(self, text: str, score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        formula_count = 0
        for formula in self.CONCLUDING_FORMULAS:
            formula_count += text.count(formula)

        metric_score = 0
        if formula_count >= 3:
            metric_score = 10
        elif formula_count >= 2:
            metric_score = 6
        elif formula_count >= 1:
            metric_score = 3

        details['metrics']['concluding_formula'] = {
            'count': formula_count,
            'score': metric_score,
            'details': f'段末总结套话 {formula_count} 处'
        }

        return score + metric_score, details

    def _analyze_suizhe_template(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        suizhe_patterns = [
            r'随着[^，。]{2,50}的[^，。]{2,50}',
            r'基于[^，。]{2,50}的[^，。]{2,50}',
        ]

        total_count = 0
        for pattern in suizhe_patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 3:
            metric_score = 15
        elif total_count >= 2:
            metric_score = 10
        elif total_count >= 1:
            metric_score = 5

        details['metrics']['suizhe_template'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'"随着/基于...的..."模板 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_paragraph_template(self, paragraphs: List[str], score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        if len(paragraphs) < 2:
            return score, details

        marker_count = 0
        for para in paragraphs:
            for marker in self.PARAGRAPH_TEMPLATE_MARKERS:
                if marker in para:
                    marker_count += 1
                    break

        metric_score = 0
        if marker_count >= 4:
            metric_score = 12
        elif marker_count >= 3:
            metric_score = 8
        elif marker_count >= 2:
            metric_score = 4

        details['metrics']['paragraph_template'] = {
            'marker_count': marker_count,
            'paragraph_count': len(paragraphs),
            'score': metric_score,
            'details': f'段落模板标记 {marker_count}/{len(paragraphs)} 段'
        }

        return score + metric_score, details

    def _analyze_definition_pattern(self, text: str, score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        patterns = [
            r'是[\u4e00-\u9fa5]{2,20}的[\u4e00-\u9fa5]{2,20}',
            r'是指[\u4e00-\u9fa5]{2,40}的[\u4e00-\u9fa5]{2,20}',
            r'是[\u4e00-\u9fa5]{2,20}之一',
        ]
        seen_spans = set()
        count = 0
        for p in patterns:
            for m in re.finditer(p, text):
                span = (m.start(), m.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    count += 1

        metric_score = 0
        if count >= 3:
            metric_score = 14
        elif count >= 2:
            metric_score = 10
        elif count >= 1:
            metric_score = 5

        details['metrics']['definition_pattern'] = {
            'count': count,
            'score': metric_score,
            'details': f'"是...的"定义式 {count} 处'
        }

        return score + metric_score, details

    def _analyze_em_dash_overuse(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        em_dash_count = text.count('——')
        single_dash_count = text.count('—') - em_dash_count * 2

        total_dashes = em_dash_count + max(0, single_dash_count // 2)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))

        density = (total_dashes / chinese_chars) * 1000 if chinese_chars > 0 else 0

        metric_score = 0
        if total_dashes >= 3:
            metric_score = 12
        elif total_dashes >= 2:
            metric_score = 8
        elif total_dashes >= 1:
            metric_score = 4

        if density > 3.0:
            metric_score = max(metric_score, 10)

        details['metrics']['em_dash_overuse'] = {
            'count': total_dashes,
            'density_per_1000': round(density, 2),
            'score': metric_score,
            'details': f'破折号 {total_dashes} 处（密度{density:.1f}/千字）'
        }

        return score + metric_score, details

    def _analyze_citation_distribution(self, text: str, score: int,
                                      details: Dict) -> Tuple[int, Dict]:
        citations = re.findall(r'\[\d+(?:[-,]\d+)*\]', text)
        total_citations = len(citations)

        if total_citations < 2:
            return score, details

        sentences = self.split_sentences(text, chinese=True)
        end_citations = 0
        for sent in sentences:
            stripped = sent.rstrip()
            if re.search(r'\[\d+(?:[-,]\d+)*\]$', stripped):
                end_citations += 1

        end_ratio = end_citations / total_citations if total_citations > 0 else 0

        metric_score = 0
        if end_ratio > 0.8 and total_citations >= 3:
            metric_score = 10
        elif end_ratio > 0.6 and total_citations >= 2:
            metric_score = 5

        details['metrics']['citation_distribution'] = {
            'total_citations': total_citations,
            'end_citations': end_citations,
            'end_citation_ratio': round(end_ratio, 2),
            'score': metric_score,
            'details': f'引用{total_citations}处，{end_ratio*100:.0f}%在句末'
        }

        return score + metric_score, details