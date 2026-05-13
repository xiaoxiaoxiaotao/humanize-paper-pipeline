"""
中文人类化器 - 知网3.0算法适配
用于将AI生成的中文文本人类化，降低知网检测概率
"""

import re
import random
from typing import Dict, List, Tuple, Optional

from .base_humanizer import BaseHumanizer


class ChineseHumanizer(BaseHumanizer):
    """
    中文人类化器
    针对知网3.0等中文AI检测平台的人类化优化
    """
    
    # 对抗性模式替换库
    ADVERSARIAL_PATTERNS = {
        '随着': ['近年来', '当前', '在这个阶段'],
        '基于': ['根据', '依据', '通过'],
        '通过': ['利用', '采用', '使用'],
        '首先': ['开篇来说', '最开始', '第一点', '首先需要'],
        '其次': ['再者', '第二点', '接着', '然后'],
        '再次': ['此外还有', '另外', '补充说明'],
        '最后': ['综上', '最终', '结尾部分'],
        '综上所述': ['综合来看', '基于以上', '从整体而言'],
        '总而言之': ['整体而言', '综合判断', '总的来说'],
        '由此可见': ['可见', '说明', '表明'],
        '值得注意的是': ['需要关注的是', '值得留意的是'],
        '显而易见': ['明显', '显然', '不言而喻'],
        '具有重要意义': ['很有意义', '值得注意'],
        '发挥重要作用': ['起到作用', '很重要'],
        '不可或缺': ['必要', '需要'],
        '至关重要': ['关键', '重要'],
        '举足轻重': ['重要'],
        '在一定程度上': [''],
        '事实上': [''],
        '实际上': [''],
        '本质上': [''],
    }
    
    # 认知特征注入库 - 模仿人类思维过程
    COGNITIVE_FEATURES = [
        '然而，这个理论在实际应用中似乎存在局限性',
        '起初我们认为如此，但后来发现情况有所不同',
        '这里存在一个容易被忽视的问题',
        '有趣的是，实际效果与预期并不完全一致',
        '需要指出的是，这个结论可能存在争议',
        '基于我们的观察，这个假设需要进一步验证',
        '值得思考的是，为什么会出现这样的差异',
        '这促使我们重新思考原有的理论框架',
        '从另一个角度来看，这个问题更加复杂',
        '坦白说，这个结果有点出乎意料',
        '我们需要承认，目前的方法还不够完善',
        '这个现象背后可能还有其他因素在起作用',
    ]
    
    # 机械模式打破器
    MECHANICAL_BREAKERS = [
        '但这里有个问题：',
        '不过我们需要思考的是：',
        '有意思的是，实际情况可能更复杂。',
        '说起来，这里有个细节需要注意。',
        '从实际经验来看，情况可能不太一样。',
        '坦白说，这个问题没那么简单。',
        '不过，这个观点或许需要修正。',
    ]
    
    def __init__(self):
        super().__init__(name="Chinese Humanizer")
    
    def humanize(self, text: str) -> Tuple[str, List[str]]:
        """
        对中文文本进行人类化处理
        
        Args:
            text: 要人类化的文本
            
        Returns:
            (humanized_text, list_of_changes)
        """
        changes = []
        result = text
        
        # 1. 打破语义指纹模式
        result, new_changes = self._break_semantic_fingerprints(result)
        changes.extend(new_changes)
        
        # 2. 打破机械论证模式
        result, new_changes = self._break_mechanical_patterns(result)
        changes.extend(new_changes)
        
        # 3. 注入认知特征
        result, new_changes = self._inject_cognitive_features(result)
        changes.extend(new_changes)
        
        # 4. 变化句式结构
        result, new_changes = self._vary_sentence_structures(result)
        changes.extend(new_changes)
        
        # 5. 变化句子长度
        result, new_changes = self._vary_sentence_lengths(result)
        changes.extend(new_changes)
        
        return result, changes
    
    def _break_semantic_fingerprints(self, text: str) -> Tuple[str, List[str]]:
        """打破知网等平台关注的语义指纹"""
        return self._replace_patterns(text, self.ADVERSARIAL_PATTERNS)
    
    def _break_mechanical_patterns(self, text: str) -> Tuple[str, List[str]]:
        """打破机械的论证模式"""
        return self._random_insert(text, self.MECHANICAL_BREAKERS, 0.35)
    
    def _inject_cognitive_features(self, text: str) -> Tuple[str, List[str]]:
        """注入人类认知特征"""
        return self._random_insert(text, self.COGNITIVE_FEATURES, 0.25)
    
    def _vary_sentence_structures(self, text: str) -> Tuple[str, List[str]]:
        """变化句式结构，避免重复"""
        sentences = self._split_sentences(text, chinese=True)
        result = []
        changes = []
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            if len(sent.strip()) > 20 and random.random() < 0.25:
                # 随机添加一些变化前缀
                prefixes = ['简单来说，', '具体而言，', '实际上，', '不过，']
                prefix = random.choice(prefixes)
                result.append(prefix + sent)
                changes.append(f"添加句式前缀: {prefix}")
            else:
                result.append(sent)
            
            result.append(punct)
        
        return ''.join(result), changes
    
    def _vary_sentence_lengths(self, text: str) -> Tuple[str, List[str]]:
        """变化句子长度，打破均匀分布"""
        sentences = self._split_sentences(text, chinese=True)
        result = []
        changes = []
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            # 对于特别长的句子，考虑拆分
            if len(sent) > 40 and random.random() < 0.3:
                # 找到逗号位置，尝试拆分
                comma_pos = sent.find('，', len(sent)//2)
                if comma_pos != -1:
                    part1 = sent[:comma_pos+1]
                    part2 = sent[comma_pos+1:]
                    result.append(part1)
                    result.append('。')
                    result.append(part2)
                    changes.append("拆分长句")
                else:
                    result.append(sent)
                    result.append(punct)
            else:
                result.append(sent)
                result.append(punct)
        
        return ''.join(result), changes
