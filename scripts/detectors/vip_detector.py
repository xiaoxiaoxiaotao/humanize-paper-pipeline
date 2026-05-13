"""
维普AI检测器 - 定向优化版
针对维普AIGC检测算法原理的启发式检测
"""

import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Optional

from .base_detector import BaseDetector


class VIPDetector(BaseDetector):
    """
    维普AI文本检测器
    针对维普AIGC检测算法的定向检测
    """
    
    # 维普语义指纹模式
    VIP_AI_FINGERPRINTS = [
        r'首先[^，。]{0,15}[，。]其次[^，。]{0,15}[，。](?:再次|最后|此外)',
        r'第一[^，。]{0,10}[，。]第二[^，。]{0,10}[，。]第三',
        r'一方面[^，。]{2,15}[，。]另一方面[^，。]{2,15}[，。]此外',
        r'随着[^的]+的发展[,，]',
        r'基于[^的]+的研究[,，]',
        r'在当前背景下[,，]',
        r'近年来[,，]',
        r'众所周知[,，]',
        r'综上所述[,，]',
        r'总而言之[,，]',
        r'由此可见[,，]',
        r'总之[,，]',
    ]
    
    # 维普机械模式
    VIP_MECHANICAL_PATTERNS = [
        r'(?:首先|第一)[^，。]{5,30}重要性[^，。]{2,15}[，。]',
        r'(?:其次|第二)[^，。]{5,30}挑战[^，。]{2,15}[，。]',
        r'(?:最后|第三)[^，。]{5,30}对策[^，。]{2,15}[，。]',
        r'存在[^，。]{3,15}问题[,，](?:因此|所以|为此)',
        r'面临[^，。]{3,15}困境[,，](?:本文|本研究|本文提出)',
        r'针对[^，。]{3,15}不足[,，](?:本文|本研究)',
        r'本文(?:旨在|研究|探讨|分析)[^，。]{5,30}具有(?:重要|现实|理论)意义',
        r'本研究的(?:目的|意义)是[^，。]{5,30}具有(?:重要|现实|理论)意义',
        r'国内外学者[^，。]{0,30}进行了[^，。]{0,30}研究[,，]',
        r'目前[^，。]{0,30}研究[^，。]{0,30}但[^，。]{0,30}不足',
    ]
    
    # 重复结构模式
    VIP_REPEATED_STRUCTURES = [
        r'(?:随着|基于|通过|利用)[^，。]{2,10}的[^，。]{2,10}[，，]',
        r'(?:具有|发挥|起到)[^，。]{2,10}(?:作用|意义|价值)[，，]',
        r'(?:重要|关键|核心)[的]?[^，。]{2,5}[，，]',
    ]
    
    # 指纹词替换库
    VIP_FINGERPRINT_REPLACEMENTS = {
        '首先': ['开篇来说', '最开始', '第一点', '首先需要'],
        '其次': ['再者', '第二点', '接着', '然后'],
        '再次': ['此外还有', '另外', '补充说明'],
        '最后': ['综上', '最终', '综上所述', '结尾部分'],
        '综上所述': ['综合来看', '基于以上', '从整体而言'],
        '总而言之': ['整体而言', '综合判断', '总的来说'],
        '由此可见': ['可见', '说明', '表明'],
        '值得注意的是': ['需要关注的是', '值得留意的是'],
        '显而易见': ['明显', '显然', '不言而喻'],
        '随着': ['近年来', '当前', '在这个阶段'],
        '基于': ['根据', '依据', '通过'],
        '通过': ['利用', '采用', '使用'],
    }
    
    def __init__(self):
        super().__init__(name="VIP AI Detector")
    
    def detect(self, text: str) -> Tuple[int, Dict]:
        """
        维普AI检测
        
        Args:
            text: 要检测的文本
            
        Returns:
            (ai_score, details)
        """
        details = {'metrics': {}, 'platform': '维普AIGC'}
        score = 0
        
        if not text.strip():
            return 0, details
        
        sentences = self.split_sentences(text, chinese=True)
        paragraphs = self.split_paragraphs(text)
        
        # 维普核心检测指标
        score, details = self._analyze_semantic_fingerprints(text, score, details)
        score, details = self._analyze_mechanical_patterns(text, score, details)
        score, details = self._analyze_repeated_structures(text, sentences, score, details)
        score, details = self._analyze_paragraph_structure(text, paragraphs, sentences, score, details)
        score, details = self._analyze_data_authenticity(text, score, details)
        
        final_score = min(100, max(0, score))
        details['overall_score'] = final_score
        details['sentence_count'] = len(sentences)
        details['paragraph_count'] = len(paragraphs)
        
        return final_score, details
    
    def _analyze_semantic_fingerprints(self, text: str, score: int, 
                                      details: Dict) -> Tuple[int, Dict]:
        """语义指纹检测 - 维普核心"""
        fingerprint_count = 0
        matched_patterns = []
        
        for pattern in self.VIP_AI_FINGERPRINTS:
            matches = re.findall(pattern, text)
            if matches:
                fingerprint_count += len(matches)
                matched_patterns.append(pattern[:30] + '...')
        
        metric_score = 0
        if fingerprint_count >= 5:
            metric_score = 30
        elif fingerprint_count >= 3:
            metric_score = 20
        elif fingerprint_count >= 1:
            metric_score = 10
        
        details['metrics']['semantic_fingerprint'] = {
            'count': fingerprint_count,
            'patterns': matched_patterns[:3],
            'score': metric_score,
            'details': f'检测到{fingerprint_count}个语义指纹'
        }
        
        return score + metric_score, details
    
    def _analyze_mechanical_patterns(self, text: str, score: int, 
                                    details: Dict) -> Tuple[int, Dict]:
        """机械模式检测 - 维普核心"""
        mechanical_count = 0
        
        for pattern in self.VIP_MECHANICAL_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                mechanical_count += len(matches)
        
        metric_score = 0
        if mechanical_count >= 4:
            metric_score = 35
        elif mechanical_count >= 2:
            metric_score = 20
        elif mechanical_count >= 1:
            metric_score = 10
        
        details['metrics']['mechanical_pattern'] = {
            'count': mechanical_count,
            'score': metric_score,
            'details': f'检测到{mechanical_count}个机械论证模式'
        }
        
        return score + metric_score, details
    
    def _analyze_repeated_structures(self, text: str, sentences: List[str], 
                                     score: int, details: Dict) -> Tuple[int, Dict]:
        """重复结构检测 - 维普特别敏感"""
        if len(sentences) < 3:
            return score, details
        
        sentence_starts = []
        for sent in sentences:
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if len(clean_sent) >= 6:
                sentence_starts.append(clean_sent[:6])
        
        if len(sentence_starts) < 3:
            return score, details
        
        counter = Counter(sentence_starts)
        most_common = counter.most_common(3)
        
        total_repetitions = sum(count - 1 for _, count in most_common)
        
        metric_score = 0
        if total_repetitions >= 6:
            metric_score = 25
        elif total_repetitions >= 3:
            metric_score = 15
        elif total_repetitions >= 1:
            metric_score = 10
        
        details['metrics']['repeated_structures'] = {
            'repetitions': total_repetitions,
            'top_patterns': [pattern[:8] for pattern, count in most_common[:3]],
            'score': metric_score,
            'details': f'检测到{total_repetitions}处结构重复'
        }
        
        return score + metric_score, details
    
    def _analyze_paragraph_structure(self, text: str, paragraphs: List[str], 
                                     sentences: List[str], score: int, 
                                     details: Dict) -> Tuple[int, Dict]:
        """段落结构分析 - 维普段落级检测"""
        if len(paragraphs) < 2:
            return score, details
        
        paragraph_lengths = [len(p) for p in paragraphs]
        
        if len(paragraph_lengths) > 1:
            avg_len = statistics.mean(paragraph_lengths)
            std_len = statistics.stdev(paragraph_lengths) if len(paragraph_lengths) > 1 else 0
            cv = std_len / avg_len if avg_len > 0 else 0
            
            para_sent_counts = []
            for para in paragraphs:
                sents = [s.strip() for s in re.split(r'[。！？!?]+', para) if s.strip()]
                para_sent_counts.append(len(sents))
            
            sent_count_cv = 0
            if len(para_sent_counts) > 1 and max(para_sent_counts) != min(para_sent_counts):
                avg_sents = statistics.mean(para_sent_counts)
                std_sents = statistics.stdev(para_sent_counts)
                sent_count_cv = std_sents / avg_sents if avg_sents > 0 else 0
            
            structure_uniformity = (cv + sent_count_cv) / 2
            
            metric_score = 0
            if structure_uniformity < 0.15:
                metric_score = 20
            elif structure_uniformity < 0.25:
                metric_score = 12
            elif structure_uniformity < 0.35:
                metric_score = 6
            
            details['metrics']['paragraph_structure'] = {
                'cv': round(cv, 3),
                'sent_count_cv': round(sent_count_cv, 3),
                'uniformity': round(structure_uniformity, 3),
                'score': metric_score,
                'details': f'段落结构均匀度{structure_uniformity*100:.0f}%'
            }
            
            return score + metric_score, details
        
        return score, details
    
    def _analyze_data_authenticity(self, text: str, score: int, 
                                  details: Dict) -> Tuple[int, Dict]:
        """数据真实性检测 - 维普方法"""
        suspicious_patterns = [
            r'\d{4}年[^，。]{0,20}(?:增长|提高|下降|减少|上升)[^，。]{0,20}\d+(?:\.\d+)?%',
            r'(?:高达|约为|接近|约)\d+(?:\.\d+)?%',
            r'(?:显著|明显|大幅|急剧)(?:增长|提高|下降|减少)',
            r'\d+倍[^，。]{0,10}(?:增长|提高|下降)',
        ]
        
        suspicious_count = 0
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, text)
            suspicious_count += len(matches)
        
        precise_numbers = re.findall(r'\d+\.\d{2,}', text)
        suspicious_count += len(precise_numbers)
        
        metric_score = 0
        if suspicious_count >= 5:
            metric_score = 20
        elif suspicious_count >= 3:
            metric_score = 12
        
        details['metrics']['data_authenticity'] = {
            'count': suspicious_count,
            'score': metric_score,
            'details': f'检测到{suspicious_count}个可疑数据'
        }
        
        return score + metric_score, details
