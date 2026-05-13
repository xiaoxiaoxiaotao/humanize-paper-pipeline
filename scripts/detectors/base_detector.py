"""
基类检测器 - 提供通用功能和接口
"""

import re
import statistics
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Tuple, Optional


class BaseDetector(ABC):
    """
    检测器基类，所有具体检测器都继承自这个基类
    提供通用的文本处理和检测接口
    """

    def __init__(self, name: str = "Base Detector"):
        self.name = name

    @abstractmethod
    def detect(self, text: str) -> Tuple[int, Dict]:
        """
        检测文本是否为AI生成
        
        Args:
            text: 要检测的文本
            
        Returns:
            Tuple of (ai_score, details)
            ai_score: 0-100的分数，越高越可能是AI生成
            details: 详细的检测结果
        """
        pass

    def split_sentences(self, text: str, chinese: bool = False) -> List[str]:
        """
        将文本分割为句子
        
        Args:
            text: 要分割的文本
            chinese: 是否为中文文本
            
        Returns:
            句子列表
        """
        if chinese:
            sentences = re.split(r'[。！？!?]+', text)
        else:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        return [s.strip() for s in sentences if s.strip()]

    def split_paragraphs(self, text: str) -> List[str]:
        """
        将文本分割为段落
        
        Args:
            text: 要分割的文本
            
        Returns:
            段落列表
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]
        return paragraphs

    def calculate_sentence_length_stats(self, sentences: List[str]) -> Dict:
        """
        计算句子长度的统计特征
        
        Args:
            sentences: 句子列表
            
        Returns:
            统计特征字典
        """
        if len(sentences) < 2:
            return {
                'avg_length': 0,
                'std_dev': 0,
                'variance_ratio': 0
            }
        
        lengths = [len(s) for s in sentences]
        avg_length = statistics.mean(lengths)
        std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
        variance_ratio = std_dev / avg_length if avg_length > 0 else 0
        
        return {
            'avg_length': round(avg_length, 2),
            'std_dev': round(std_dev, 2),
            'variance_ratio': round(variance_ratio, 3)
        }

    def detect_repeated_patterns(self, text: str, patterns: List[str]) -> int:
        """
        检测文本中重复出现的模式数量
        
        Args:
            text: 要检测的文本
            patterns: 模式列表
            
        Returns:
            匹配数量
        """
        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            total_matches += len(matches)
        return total_matches

    def count_word_occurrences(self, text: str, words: List[str]) -> int:
        """
        统计特定词在文本中的出现次数
        
        Args:
            text: 要检测的文本
            words: 词列表
            
        Returns:
            总出现次数
        """
        count = 0
        for word in words:
            count += text.count(word)
        return count
