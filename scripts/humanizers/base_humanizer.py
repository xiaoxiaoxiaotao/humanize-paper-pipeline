"""
人类化器基类 - 提供通用的文本人类化功能
"""

import re
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


class BaseHumanizer(ABC):
    """
    人类化器基类
    所有具体人类化器都继承自这个基类
    """
    
    def __init__(self, name: str = "Base Humanizer"):
        self.name = name
    
    @abstractmethod
    def humanize(self, text: str) -> Tuple[str, List[str]]:
        """
        对文本进行人类化处理
        
        Args:
            text: 要人类化的文本
            
        Returns:
            (humanized_text, list_of_changes)
        """
        pass
    
    def _random_insert(self, text: str, insertions: List[str], 
                     chance: float = 0.3) -> Tuple[str, List[str]]:
        """
        随机插入文本片段
        
        Args:
            text: 原文本
            insertions: 可插入的文本片段列表
            chance: 插入概率
            
        Returns:
            (modified_text, changes)
        """
        paragraphs = re.split(r'(\n\s*\n)', text)
        result = []
        changes = []
        
        for i, para in enumerate(paragraphs):
            result.append(para)
            
            if para.strip() and random.random() < chance:
                insertion = random.choice(insertions)
                result.append(" " + insertion + " ")
                changes.append(f"插入内容: {insertion[:20]}...")
        
        return ''.join(result), changes
    
    def _replace_patterns(self, text: str, 
                        replacements: Dict[str, List[str]]) -> Tuple[str, List[str]]:
        """
        随机替换文本中的模式
        
        Args:
            text: 原文本
            replacements: 替换字典 {old_word: [new_words]}
            
        Returns:
            (modified_text, changes)
        """
        changes = []
        result = text
        
        for old_word, new_words in replacements.items():
            if old_word in text:
                new_word = random.choice(new_words)
                # 只替换部分出现的，保留一些原始特征
                parts = text.split(old_word)
                new_parts = [parts[0]]
                for i in range(1, len(parts)):
                    if random.random() < 0.6:
                        new_parts.append(new_word)
                    else:
                        new_parts.append(old_word)
                    new_parts.append(parts[i])
                
                result = ''.join(new_parts)
                changes.append(f"替换 {old_word} -> {new_word}")
        
        return result, changes
    
    def _split_sentences(self, text: str, chinese: bool = True) -> List[str]:
        """
        分割文本为句子
        
        Args:
            text: 要分割的文本
            chinese: 是否为中文
            
        Returns:
            句子列表
        """
        if chinese:
            return re.split(r'([。！？!?])', text)
        else:
            return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    def _vary_sentence_lengths(self, text: str, 
                            change_chance: float = 0.3) -> Tuple[str, List[str]]:
        """
        变化句子长度，增加文本的自然度
        
        Args:
            text: 原文本
            change_chance: 变化概率
            
        Returns:
            (modified_text, changes)
        """
        # 这里可以在子类中实现具体的变化策略
        return text, []
