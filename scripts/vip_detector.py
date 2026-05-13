"""
维普AI检测器 - 定向优化版
基于维普AIGC检测算法原理的启发式检测

维普检测核心方法：
1. 语义指纹对比：检测AI高频句式
2. 模式识别算法：分析段落结构和论证逻辑
3. 数据真实性验证：检测虚构数据
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional


VIP_AI_FINGERPRINTS = [
    # "首先，其次，最后" 结构
    r'首先[,，](?:[^，。]{2,20}[，。])?其次[,，](?:[^，。]{2,20}[，。])?(?:再次|最后|此外)',
    r'第一[^，。]{0,10}[，。]第二[^，。]{0,10}[，。]第三',
    r'一方面[^，。]{2,15}[，。]另一方面[^，。]{2,15}[，。]此外',
    
    # 常见AI开头模式
    r'随着[^的]+的发展[,，]',
    r'基于[^的]+的研究[,，]',
    r'在当前背景下[,，]',
    r'近年来[,，]',
    r'众所周知[,，]',
    
    # AI典型结尾模式
    r'综上所述[,，]',
    r'总而言之[,，]',
    r'由此可见[,，]',
    r'总之[,，]',
    
    # 过度使用的论证词
    r'首先[^，。]{0,15}具有重要意义[^，。]{0,15}其次',
    r'第一[^，。]{0,15}发挥着重要作用[^，。]{0,15}第二',
]


VIP_MECHANICAL_PATTERNS = [
    # "提出问题→分析问题→解决问题" 机械模式
    r'(?:首先|第一)[^，。]{5,30}重要性[^，。]{2,15}[，。]',
    r'(?:其次|第二)[^，。]{5,30}挑战[^，。]{2,15}[，。]',
    r'(?:最后|第三)[^，。]{5,30}对策[^，。]{2,15}[，。]',
    
    # 机械的问题-解决方案模式
    r'存在[^，。]{3,15}问题[,，](?:因此|所以|为此)',
    r'面临[^，。]{3,15}困境[,，](?:本文|本研究|本文提出)',
    r'针对[^，。]{3,15}不足[,，](?:本文|本研究)',
    
    # 标准化的研究意义写法
    r'本文(?:旨在|研究|探讨|分析)[^，。]{5,30}具有(?:重要|现实|理论)意义',
    r'本研究的(?:目的|意义)是[^，。]{5,30}具有(?:重要|现实|理论)意义',
    
    # 机械的文献综述模式
    r'国内外学者[^，。]{0,30}进行了[^，。]{0,30}研究[,，]',
    r'目前[^，。]{0,30}研究[^，。]{0,30}但[^，。]{0,30}不足',
]


VIP_REPEATED_STRUCTURES = [
    # 连续使用相同的句式开头
    r'(?:随着|基于|通过|利用)[^，。]{2,10}的[^，。]{2,10}[,，]',
    r'(?:具有|发挥|起到)[^，。]{2,10}(?:作用|意义|价值)[，,]',
    r'(?:重要|关键|核心)[的]?[^，。]{2,5}[，,]',
]


class VIPDetector:
    """
    维普AI检测器
    针对维普AIGC检测算法的定向检测
    """
    
    def __init__(self):
        self.name = "VIP Detector"
    
    def detect(self, text: str) -> Tuple[int, Dict]:
        """
        检测AI生成内容
        
        Returns:
            Tuple of (ai_score, details)
        """
        details = {'metrics': {}, 'platform': '维普AIGC'}
        score = 0
        
        if not text.strip():
            return 0, details
        
        # 1. 语义指纹检测
        score, details = self._analyze_semantic_fingerprints(text, score, details)
        
        # 2. 机械模式检测
        score, details = self._analyze_mechanical_patterns(text, score, details)
        
        # 3. 重复句式检测 (维普特别敏感)
        score, details = self._analyze_repeated_structures(text, score, details)
        
        # 4. 段落结构分析
        score, details = self._analyze_paragraph_structure(text, score, details)
        
        # 5. 数据真实性检测
        score, details = self._analyze_data_authenticity(text, score, details)
        
        final_score = min(100, max(0, int(score)))
        details['overall_score'] = final_score
        return final_score, details
    
    def _analyze_semantic_fingerprints(self, text: str, score: int, details: Dict) -> Tuple[int, Dict]:
        """语义指纹检测 - 维普核心方法"""
        fingerprint_matches = []
        total_matches = 0
        
        for pattern in VIP_AI_FINGERPRINTS:
            matches = re.findall(pattern, text)
            if matches:
                count = len(matches) if isinstance(matches[0], str) else len(matches)
                total_matches += count
                fingerprint_matches.extend(matches if isinstance(matches[0], str) else [m[0] if isinstance(m, tuple) else m for m in matches])
        
        details['metrics']['semantic_fingerprint'] = {
            'match_count': total_matches,
            'fingerprint_count': len(fingerprint_matches),
            'details': f'检测到{total_matches}个AI语义指纹'
        }
        
        # 维普对此非常敏感
        if total_matches >= 5:
            penalty = min((total_matches - 4) * 8, 30)
            score += penalty
            details['metrics']['semantic_fingerprint']['details'] += f'，强烈惩罚 +{penalty}'
        elif total_matches >= 3:
            penalty = min((total_matches - 2) * 6, 20)
            score += penalty
            details['metrics']['semantic_fingerprint']['details'] += f'，惩罚 +{penalty}'
        elif total_matches >= 1:
            penalty = total_matches * 4
            score += penalty
            details['metrics']['semantic_fingerprint']['details'] += f'，轻微惩罚 +{penalty}'
        
        return score, details
    
    def _analyze_mechanical_patterns(self, text: str, score: int, details: Dict) -> Tuple[int, Dict]:
        """机械模式检测 - 维普核心方法"""
        mechanical_matches = []
        total_matches = 0
        
        for pattern in VIP_MECHANICAL_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                count = len(matches)
                total_matches += count
                mechanical_matches.extend(matches)
        
        details['metrics']['mechanical_pattern'] = {
            'match_count': total_matches,
            'details': f'检测到{total_matches}个机械论证模式'
        }
        
        # 维普对此特别敏感
        if total_matches >= 4:
            penalty = min((total_matches - 3) * 10, 35)
            score += penalty
            details['metrics']['mechanical_pattern']['details'] += f'，强烈惩罚 +{penalty}'
        elif total_matches >= 2:
            penalty = min((total_matches - 1) * 7, 25)
            score += penalty
            details['metrics']['mechanical_pattern']['details'] += f'，惩罚 +{penalty}'
        elif total_matches >= 1:
            penalty = 6
            score += penalty
            details['metrics']['mechanical_pattern']['details'] += f'，轻微惩罚 +{penalty}'
        
        return score, details
    
    def _analyze_repeated_structures(self, text: str, score: int, details: Dict) -> Tuple[int, Dict]:
        """重复句式检测 - 维普特别敏感"""
        sentences = [s.strip() for s in re.split(r'[。！？]+', text) if s.strip()]
        
        if len(sentences) < 3:
            return score, details
        
        # 提取句式开头
        sentence_starts = []
        for sent in sentences:
            # 提取前8个字符作为句式特征
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if len(clean_sent) >= 6:
                sentence_starts.append(clean_sent[:6])
        
        if len(sentence_starts) < 3:
            return score, details
        
        # 统计重复
        start_counter = Counter(sentence_starts)
        most_common = start_counter.most_common(3)
        
        total_repetitions = sum(count - 1 for _, count in most_common)
        
        details['metrics']['repeated_structures'] = {
            'repetition_count': total_repetitions,
            'top_patterns': [(pattern, count) for pattern, count in most_common[:3]],
            'details': f'检测到{total_repetitions}处句式重复'
        }
        
        # 维普对重复模式特别敏感
        if total_repetitions >= 6:
            penalty = min(total_repetitions * 5, 30)
            score += penalty
            details['metrics']['repeated_structures']['details'] += f'，强烈惩罚 +{penalty}'
        elif total_repetitions >= 3:
            penalty = min(total_repetitions * 4, 20)
            score += penalty
            details['metrics']['repeated_structures']['details'] += f'，惩罚 +{penalty}'
        elif total_repetitions >= 1:
            penalty = total_repetitions * 3
            score += penalty
            details['metrics']['repeated_structures']['details'] += f'，轻微惩罚 +{penalty}'
        
        return score, details
    
    def _analyze_paragraph_structure(self, text: str, score: int, details: Dict) -> Tuple[int, Dict]:
        """段落结构分析 - 维普段落级检测"""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if len(paragraphs) < 2:
            return score, details
        
        # 分析段落结构相似性
        paragraph_lengths = [len(p) for p in paragraphs]
        import statistics
        
        if len(paragraph_lengths) > 1:
            mean_len = statistics.mean(paragraph_lengths)
            std_len = statistics.stdev(paragraph_lengths) if len(paragraph_lengths) > 1 else 0
            cv = std_len / mean_len if mean_len > 0 else 0
            
            # 统计段落句数
            para_sent_counts = []
            for para in paragraphs:
                sents = [s.strip() for s in re.split(r'[。！？]+', para) if s.strip()]
                para_sent_counts.append(len(sents))
            
            sent_count_cv = 0
            if len(para_sent_counts) > 1 and max(para_sent_counts) != min(para_sent_counts):
                mean_sents = statistics.mean(para_sent_counts)
                std_sents = statistics.stdev(para_sent_counts)
                sent_count_cv = std_sents / mean_sents if mean_sents > 0 else 0
            
            # 段落长度过于均匀 + 句数过于均匀 = AI特征
            structure_uniformity = (cv + sent_count_cv) / 2
            
            details['metrics']['paragraph_structure'] = {
                'length_cv': round(cv, 3),
                'sent_count_cv': round(sent_count_cv, 3),
                'uniformity': round(structure_uniformity, 3),
                'details': f'段落结构均匀度{round(structure_uniformity*100)}%'
            }
            
            # 维普阈值
            if structure_uniformity < 0.15:
                penalty = 20
                score += penalty
                details['metrics']['paragraph_structure']['details'] += '，段落结构过于机械，惩罚 +20'
            elif structure_uniformity < 0.25:
                penalty = 12
                score += penalty
                details['metrics']['paragraph_structure']['details'] += '，段落结构较机械，惩罚 +12'
            elif structure_uniformity < 0.35:
                penalty = 6
                score += penalty
                details['metrics']['paragraph_structure']['details'] += '，段落结构偏机械，轻微惩罚 +6'
        
        return score, details
    
    def _analyze_data_authenticity(self, text: str, score: int, details: Dict) -> Tuple[int, Dict]:
        """数据真实性检测 - 维普方法"""
        # 检测可疑的数据模式
        suspicious_patterns = [
            r'\d{4}年[^，。]{0,20}(?:增长|提高|下降|减少|上升)[^，。]{0,20}\d+(?:\.\d+)?%',
            r'(?:高达|约为|接近|约)\d+(?:\.\d+)?%',
            r'(?:显著|明显|大幅|急剧)(?:增长|提高|下降|减少)',
            r'\d+倍[^，。]{0,10}(?:增长|提高|下降)',
        ]
        
        suspicious_matches = []
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, text)
            suspicious_matches.extend(matches)
        
        # 检测过于精确的数据
        precise_numbers = re.findall(r'\d+\.\d{2,}', text)
        suspicious_matches.extend(precise_numbers)
        
        details['metrics']['data_authenticity'] = {
            'suspicious_count': len(suspicious_matches),
            'examples': suspicious_matches[:5],
            'details': f'检测到{len(suspicious_matches)}个可疑数据'
        }
        
        if len(suspicious_matches) >= 5:
            penalty = min(len(suspicious_matches) * 4, 25)
            score += penalty
            details['metrics']['data_authenticity']['details'] += f'，可疑数据过多，惩罚 +{penalty}'
        elif len(suspicious_matches) >= 3:
            penalty = min(len(suspicious_matches) * 3, 15)
            score += penalty
            details['metrics']['data_authenticity']['details'] += f'，存在可疑数据，轻微惩罚 +{penalty}'
        
        return score, details


def detect_vip_ai(text: str) -> Tuple[int, Dict]:
    """维普AI检测入口函数"""
    detector = VIPDetector()
    return detector.detect(text)
