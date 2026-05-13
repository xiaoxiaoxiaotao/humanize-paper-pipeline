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
    
    # 中文AI过渡词
    AI_TRANSITIONS = [
        '首先', '其次', '再次', '最后', '综上所述',
        '值得注意的是', '需要强调的是', '进一步而言',
        '总体来说', '一般而言', '事实上',
        '随着', '基于', '通过', '利用'
    ]
    
    # 抽象占位符短语
    ABSTRACT_PHRASES = [
        '具有重要意义', '发挥着重要作用', '至关重要',
        '不可或缺', '具有重要价值', '促进了发展',
        '综上所述', '基于此', '因此', '然而'
    ]
    
    # 知网典型模板模式
    TEMPLATE_PATTERNS = [
        r'随着[^，。]{2,25}的[^，。]{2,25}',
        r'基于[^，。]{2,25}的[^，。]{2,25}',
        r'通过[^，。]{2,25}的[^，。]{2,25}',
        r'因此[^，。]{2,25}具有重要意义',
    ]
    
    def __init__(self):
        super().__init__(name="Chinese AI Detector")
    
    def detect(self, text: str) -> Tuple[int, Dict]:
        """
        检测中文AI生成文本
        
        Args:
            text: 要检测的文本
            
        Returns:
            (ai_score, details)
        """
        details = {'metrics': {}, 'platform': '知网3.0'}
        score = 0
        
        sentences = self.split_sentences(text, chinese=True)
        paragraphs = self.split_paragraphs(text)
        
        if len(sentences) < 2 or not text.strip():
            return 0, details
        
        # 知网3.0核心检测指标（优先）
        score, details = self._analyze_sentence_length_distribution(sentences, score, details)
        score, details = self._analyze_paragraph_structure_similarity(paragraphs, sentences, score, details)
        score, details = self._analyze_info_density_distribution(sentences, score, details)
        score, details = self._analyze_transition_word_distribution(text, sentences, score, details)
        
        # 其他中文检测指标
        score, details = self._analyze_sentence_uniformity(sentences, score, details)
        score, details = self._analyze_transitions(text, score, details)
        score, details = self._analyze_abstract_language(text, score, details)
        score, details = self._analyze_sentence_openings(sentences, score, details)
        score, details = self._analyze_templates(text, score, details)
        score, details = self._analyze_word_burstiness(text, score, details)
        
        final_score = min(100, max(0, score))
        details['overall_score'] = final_score
        details['sentence_count'] = len(sentences)
        details['paragraph_count'] = len(paragraphs)
        
        return final_score, details
    
    def _analyze_sentence_length_distribution(self, sentences: List[str], 
                                             score: int, details: Dict) -> Tuple[int, Dict]:
        """知网3.0 - 句长分布分析"""
        if len(sentences) < 3:
            return score, details
        
        # 计算每句话不含标点的长度
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
        
        # 统计落在15-25字的比例（AI典型特征）
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
        """知网3.0 - 段落结构相似性分析"""
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
        """知网3.0 - 信息密度分布分析"""
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
        
        # 统计落在65-75%区间的比例
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
        """知网3.0 - 连接词频率与分布均匀性分析"""
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
        """句子长度均匀性分析"""
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
    
    def _analyze_transitions(self, text: str, score: int, 
                            details: Dict) -> Tuple[int, Dict]:
        """过渡词过度使用分析"""
        transition_count = self.count_word_occurrences(text, self.AI_TRANSITIONS)
        
        metric_score = 0
        if transition_count > 8:
            metric_score = 15
        elif transition_count > 5:
            metric_score = 10
        elif transition_count > 2:
            metric_score = 5
        
        details['metrics']['transitions'] = {
            'count': transition_count,
            'score': metric_score,
            'details': f'过渡词数量 {transition_count}'
        }
        
        return score + metric_score, details
    
    def _analyze_abstract_language(self, text: str, score: int, 
                                  details: Dict) -> Tuple[int, Dict]:
        """抽象语言分析"""
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
    
    def _analyze_sentence_openings(self, sentences: List[str], score: int, 
                                  details: Dict) -> Tuple[int, Dict]:
        """句子开头重复分析"""
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
        
        details['metrics']['sentence_openings'] = {
            'top_pattern': top_pattern,
            'count': top_count,
            'score': metric_score,
            'details': f'最常见开头 "{top_pattern}" 出现 {top_count} 次'
        }
        
        return score + metric_score, details
    
    def _analyze_templates(self, text: str, score: int, 
                          details: Dict) -> Tuple[int, Dict]:
        """模板模式分析"""
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
        """词汇突发性分析"""
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
