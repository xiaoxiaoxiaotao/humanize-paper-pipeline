"""
英文AI文本检测器
基于常见的AI生成学术文本模式进行检测
"""

import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Optional

from .base_detector import BaseDetector


class EnglishDetector(BaseDetector):
    """
    英文AI文本检测器
    """
    
    # 常见AI过渡词
    AI_TRANSITIONS = [
        'moreover', 'furthermore', 'additionally', 'in addition',
        'it is important to note that', 'it should be noted that',
        'it is worth noting that', 'notably', 'significantly',
        'however', 'nevertheless', 'therefore', 'thus',
        'consequently', 'hence', 'accordingly', 'overall',
        'in conclusion', 'to summarize', 'to sum up',
        'taken together', 'in essence', 'firstly', 'secondly',
        'thirdly', 'lastly', 'finally'
    ]
    
    # 抽象占位符短语
    ABSTRACT_PHRASES = [
        'various aspects', 'multiple factors', 'different perspectives',
        'in terms of', 'with regard to', 'with respect to',
        'it can be seen that', 'it has been shown that',
        'plays an important role', 'plays a crucial role',
        'serves as', 'acts as', 'functions as',
        'various ways', 'multiple dimensions', 'different angles',
        'significant impact', 'profound effect', 'important implications',
        'key factors', 'critical aspects', 'essential components',
        'a variety of', 'a number of', 'a range of',
        'in the context of', 'from the perspective of',
        'in the realm of', 'in the field of',
        'has gained attention', 'has drawn interest',
        'has been widely studied', 'has been extensively researched',
        'leveraging', 'utilizing', 'using', 'employing',
        'substantial', 'considerable', 'significant'
    ]
    
    # 模糊语气词
    HEDGING_PHRASES = [
        'may suggest', 'might indicate', 'could imply',
        'would seem', 'appears to be', 'seems to',
        'to some extent', 'in some cases', 'to a certain degree',
        'tends to', 'is likely to', 'is possible that',
        'potentially', 'presumably', 'arguably'
    ]
    
    def __init__(self):
        super().__init__(name="English AI Detector")
    
    def detect(self, text: str) -> Tuple[int, Dict]:
        """
        检测英文AI生成文本
        
        Args:
            text: 要检测的文本
            
        Returns:
            (ai_score, details)
        """
        details = {'metrics': {}, 'platform': 'general'}
        score = 0
        
        sentences = self.split_sentences(text, chinese=False)
        paragraphs = self.split_paragraphs(text)
        
        if len(sentences) < 2 or not text.strip():
            return 0, details
        
        # 分析各类指标
        score, details = self._analyze_sentence_uniformity(sentences, score, details)
        score, details = self._analyze_transition_overuse(text, sentences, score, details)
        score, details = self._analyze_abstract_language(text, score, details)
        score, details = self._analyze_vocabulary_diversity(text, score, details)
        score, details = self._analyze_passive_voice(text, sentences, score, details)
        score, details = self._analyze_paragraph_patterns(paragraphs, score, details)
        score, details = self._analyze_burstiness(sentences, score, details)
        score, details = self._analyze_bigram_ttr(text, score, details)
        score, details = self._analyze_clause_chain(sentences, score, details)
        score, details = self._analyze_sentence_openings(sentences, score, details)
        score, details = self._analyze_punctuation(text, score, details)
        score, details = self._analyze_concluding_formula(text, score, details)
        score, details = self._analyze_hedging(text, score, details)
        score, details = self._analyze_word_repetition(text, score, details)
        
        final_score = min(100, max(0, score))
        details['overall_score'] = final_score
        details['sentence_count'] = len(sentences)
        details['paragraph_count'] = len(paragraphs)
        
        return final_score, details
    
    def _analyze_sentence_uniformity(self, sentences: List[str], score: int, 
                                     details: Dict) -> Tuple[int, Dict]:
        """分析句子长度均匀性"""
        if len(sentences) < 3:
            return score, details
        
        word_counts = [len(s.split()) for s in sentences]
        avg_length = statistics.mean(word_counts)
        std_dev = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        variance_ratio = std_dev / avg_length if avg_length > 0 else 0
        
        metric_score = 0
        if avg_length < 10:
            threshold_high, threshold_mod = 0.15, 0.25
        elif avg_length < 18:
            threshold_high, threshold_mod = 0.20, 0.30
        else:
            threshold_high, threshold_mod = 0.25, 0.35
        
        if variance_ratio < threshold_high:
            metric_score = 15
        elif variance_ratio < threshold_mod:
            metric_score = 10
        elif variance_ratio < 0.45:
            metric_score = 5
        
        details['metrics']['sentence_uniformity'] = {
            'avg_length': round(avg_length, 1),
            'variance_ratio': round(variance_ratio, 3),
            'score': metric_score,
            'details': f'句长变异系数 {variance_ratio:.3f}'
        }
        
        return score + metric_score, details
    
    def _analyze_transition_overuse(self, text: str, sentences: List[str], 
                                    score: int, details: Dict) -> Tuple[int, Dict]:
        """分析过渡词过度使用"""
        text_lower = text.lower()
        transition_count = self.count_word_occurrences(text_lower, self.AI_TRANSITIONS)
        
        transition_pct = (transition_count / len(sentences)) * 100 if sentences else 0
        
        metric_score = 0
        if transition_pct > 25:
            metric_score = 25
        elif transition_pct > 15:
            metric_score = 15
        elif transition_pct > 8:
            metric_score = 8
        
        details['metrics']['transition_overuse'] = {
            'count': transition_count,
            'percentage': round(transition_pct, 1),
            'score': metric_score,
            'details': f'{transition_count} 个过渡词 ({transition_pct:.1f}%)'
        }
        
        return score + metric_score, details
    
    def _analyze_abstract_language(self, text: str, score: int, 
                                  details: Dict) -> Tuple[int, Dict]:
        """分析抽象语言使用"""
        text_lower = text.lower()
        phrase_count = self.count_word_occurrences(text_lower, self.ABSTRACT_PHRASES)
        
        word_count = len(text.split())
        density = (phrase_count / word_count) * 100 if word_count > 0 else 0
        
        metric_score = 0
        if density > 2.0:
            metric_score = 20
        elif density > 1.0:
            metric_score = 12
        elif density > 0.5:
            metric_score = 5
        
        details['metrics']['abstract_language'] = {
            'count': phrase_count,
            'density': round(density, 2),
            'score': metric_score,
            'details': f'抽象短语密度 {density:.2f}/100词'
        }
        
        return score + metric_score, details
    
    def _analyze_vocabulary_diversity(self, text: str, score: int, 
                                     details: Dict) -> Tuple[int, Dict]:
        """分析词汇多样性"""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        if len(words) < 10:
            return score, details
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        metric_score = 0
        if ttr < 0.40:
            metric_score = 15
        elif ttr < 0.50:
            metric_score = 10
        elif ttr < 0.60:
            metric_score = 5
        
        details['metrics']['vocabulary_diversity'] = {
            'ttr': round(ttr, 3),
            'unique_words': len(unique_words),
            'total_words': len(words),
            'score': metric_score,
            'details': f'词汇多样性 TTR={ttr:.3f}'
        }
        
        return score + metric_score, details
    
    def _analyze_passive_voice(self, text: str, sentences: List[str], 
                              score: int, details: Dict) -> Tuple[int, Dict]:
        """分析被动语态使用"""
        text_lower = text.lower()
        passive_patterns = [
            r'\b(?:is|are|was|were|been|be|being)\s+\w+ed\b',
            r'\b(?:has|have|had)\s+been\s+\w+ed\b',
        ]
        
        passive_count = 0
        for pattern in passive_patterns:
            passive_count += len(re.findall(pattern, text_lower))
        
        passive_pct = (passive_count / len(sentences)) * 100 if sentences else 0
        
        metric_score = 0
        if passive_pct > 50:
            metric_score = 12
        elif passive_pct > 35:
            metric_score = 8
        elif passive_pct > 20:
            metric_score = 4
        
        details['metrics']['passive_voice'] = {
            'count': passive_count,
            'percentage': round(passive_pct, 1),
            'score': metric_score,
            'details': f'被动语态占比 {passive_pct:.1f}%'
        }
        
        return score + metric_score, details
    
    def _analyze_paragraph_patterns(self, paragraphs: List[str], 
                                   score: int, details: Dict) -> Tuple[int, Dict]:
        """分析段落模式"""
        if len(paragraphs) < 3:
            return score, details
        
        para_starts = []
        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            if sentences:
                para_starts.append(sentences[0].lower()[:30])
        
        similar_count = 0
        for i in range(len(para_starts)):
            for j in range(i + 1, len(para_starts)):
                if para_starts[i][:20] == para_starts[j][:20]:
                    similar_count += 1
        
        similarity_ratio = similar_count / len(paragraphs) if paragraphs else 0
        
        metric_score = 0
        if similarity_ratio > 0.3:
            metric_score = 12
        elif similarity_ratio > 0.15:
            metric_score = 7
        
        details['metrics']['paragraph_patterns'] = {
            'similar_count': similar_count,
            'similarity_ratio': round(similarity_ratio, 3),
            'score': metric_score,
            'details': f'{similar_count} 个相似段落开头'
        }
        
        return score + metric_score, details
    
    def _analyze_burstiness(self, sentences: List[str], score: int, 
                           details: Dict) -> Tuple[int, Dict]:
        """分析句子长度突发性"""
        if len(sentences) < 3:
            return score, details
        
        word_counts = [len(s.split()) for s in sentences]
        avg_len = statistics.mean(word_counts)
        std_dev = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        cv = std_dev / avg_len if avg_len > 0 else 0
        
        metric_score = 0
        if cv < 0.25:
            metric_score = 8
        elif cv < 0.35:
            metric_score = 4
        
        details['metrics']['burstiness'] = {
            'cv': round(cv, 3),
            'score': metric_score,
            'details': f'突发性 CV={cv:.3f}'
        }
        
        return score + metric_score, details
    
    def _analyze_bigram_ttr(self, text: str, score: int, 
                          details: Dict) -> Tuple[int, Dict]:
        """分析二元组类型标记比"""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if len(words) < 10:
            return score, details
        
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        ttr = len(unique_bigrams) / len(bigrams) if bigrams else 1.0
        
        metric_score = 0
        if ttr < 0.55:
            metric_score = 10
        elif ttr < 0.65:
            metric_score = 6
        
        details['metrics']['bigram_ttr'] = {
            'ttr': round(ttr, 3),
            'unique_bigrams': len(unique_bigrams),
            'total_bigrams': len(bigrams),
            'score': metric_score,
            'details': f'二元组 TTR={ttr:.3f}'
        }
        
        return score + metric_score, details
    
    def _analyze_clause_chain(self, sentences: List[str], score: int, 
                             details: Dict) -> Tuple[int, Dict]:
        """分析从句链密度"""
        if len(sentences) < 3:
            return score, details
        
        comma_counts = [s.count(',') + s.count(';') for s in sentences]
        avg_commas = statistics.mean(comma_counts) if comma_counts else 0
        
        metric_score = 0
        if avg_commas > 4.0:
            metric_score = 8
        elif avg_commas > 3.0:
            metric_score = 5
        
        details['metrics']['clause_chain'] = {
            'avg_commas': round(avg_commas, 2),
            'score': metric_score,
            'details': f'平均句长逗号数 {avg_commas:.1f}'
        }
        
        return score + metric_score, details
    
    def _analyze_sentence_openings(self, sentences: List[str], score: int, 
                                 details: Dict) -> Tuple[int, Dict]:
        """分析句子开头重复"""
        if len(sentences) < 5:
            return score, details
        
        openings = []
        for s in sentences:
            words = s.strip().split()
            if len(words) >= 2:
                openings.append(' '.join(words[:2]).lower())
        
        if not openings:
            return score, details
        
        counter = Counter(openings)
        most_common = counter.most_common(1)
        top_pattern, top_count = most_common[0]
        
        metric_score = 0
        if top_count >= 4:
            metric_score = 10
        elif top_count >= 3:
            metric_score = 6
        
        details['metrics']['sentence_openings'] = {
            'top_pattern': top_pattern,
            'count': top_count,
            'score': metric_score,
            'details': f'最常见开头 "{top_pattern}" 出现 {top_count} 次'
        }
        
        return score + metric_score, details
    
    def _analyze_punctuation(self, text: str, score: int, 
                          details: Dict) -> Tuple[int, Dict]:
        """分析标点符号模式"""
        comma_count = text.count(',')
        period_count = text.count('.') + text.count('!') + text.count('?')
        ratio = comma_count / period_count if period_count > 0 else 0
        
        metric_score = 0
        if ratio > 4.0:
            metric_score = 8
        elif ratio > 3.0:
            metric_score = 5
        
        details['metrics']['punctuation'] = {
            'comma_period_ratio': round(ratio, 2),
            'score': metric_score,
            'details': f'逗号句号比 {ratio:.2f}'
        }
        
        return score + metric_score, details
    
    def _analyze_concluding_formula(self, text: str, score: int, 
                                  details: Dict) -> Tuple[int, Dict]:
        """分析公式化结束语"""
        concluding_patterns = [
            r'in conclusion', r'to summarize', r'in summary',
            r'overall', r'taken together', r'in essence',
            r'this paper', r'the present study',
        ]
        
        count = 0
        text_lower = text.lower()
        for pattern in concluding_patterns:
            count += len(re.findall(pattern, text_lower))
        
        metric_score = 0
        if count >= 3:
            metric_score = 10
        elif count >= 2:
            metric_score = 6
        
        details['metrics']['concluding_formula'] = {
            'count': count,
            'score': metric_score,
            'details': f'公式化短语 {count} 个'
        }
        
        return score + metric_score, details
    
    def _analyze_hedging(self, text: str, score: int, 
                        details: Dict) -> Tuple[int, Dict]:
        """分析模糊语气词过度使用"""
        text_lower = text.lower()
        count = self.count_word_occurrences(text_lower, self.HEDGING_PHRASES)
        
        metric_score = 0
        if count >= 5:
            metric_score = 12
        elif count >= 3:
            metric_score = 8
        
        details['metrics']['hedging'] = {
            'count': count,
            'score': metric_score,
            'details': f'模糊语气词 {count} 个'
        }
        
        return score + metric_score, details
    
    def _analyze_word_repetition(self, text: str, score: int, 
                               details: Dict) -> Tuple[int, Dict]:
        """分析词汇重复"""
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        if len(words) < 20:
            return score, details
        
        counter = Counter(words)
        total = len(words)
        repeated = sum(c for w, c in counter.items() if c >= 3)
        repeat_ratio = repeated / total if total > 0 else 0
        
        metric_score = 0
        if repeat_ratio > 0.15:
            metric_score = 10
        elif repeat_ratio > 0.10:
            metric_score = 6
        
        details['metrics']['word_repetition'] = {
            'repeat_ratio': round(repeat_ratio, 3),
            'score': metric_score,
            'details': f'重复词汇占比 {repeat_ratio*100:.1f}%'
        }
        
        return score + metric_score, details
