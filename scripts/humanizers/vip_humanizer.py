"""
维普人类化器 - 定向优化版
针对维普AIGC检测算法的对抗性人类化
"""

import re
import random
from typing import Dict, List, Tuple, Optional

from .base_humanizer import BaseHumanizer


class VIPHumanizer(BaseHumanizer):
    """
    维普人类化器
    针对维普AIGC检测算法的定向人类化优化
    """
    
    # 维普指纹词替换库
    VIP_FINGERPRINT_REPLACEMENTS = {
        '首先': ['开篇来说', '最开始', '第一点', '首先需要'],
        '其次': ['再者', '第二点', '接着', '然后'],
        '再次': ['此外还有', '另外', '补充说明'],
        '最后': ['综上', '最终', '结尾部分'],
        '综上所述': ['综合来看', '基于以上', '从整体而言'],
        '总而言之': ['整体而言', '综合判断', '总的来说'],
        '由此可见': ['可见', '说明', '表明'],
        '值得注意的是': ['需要关注的是', '值得留意的是'],
        '显而易见': ['明显', '显然', '不言而喻'],
        '随着': ['近年来', '当前', '在这个阶段'],
        '基于': ['根据', '依据', '通过'],
        '通过': ['利用', '采用', '使用'],
        '存在问题': ['这里有个问题', '遇到些问题', '有些不足'],
        '面临困境': ['遇到困难', '有些挑战', '有点麻烦'],
        '针对不足': ['考虑到不足', '针对缺点', '针对问题'],
        '具有重要意义': ['很有价值', '值得关注', '挺重要的'],
        '发挥重要作用': ['起到作用', '很重要', '很关键'],
    }
    
    # 维普机械模式打破器
    VIP_MECHANICAL_BREAKERS = [
        '但这里有个问题：',
        '不过我们需要思考的是：',
        '有意思的是，实际情况可能更复杂。',
        '说起来，这里有个细节需要注意。',
        '从实际经验看，情况可能不太一样。',
        '坦白说，这个问题没那么简单。',
        '不过，这个观点或许需要修正。',
        '需要指出的是，这个结论可能有争议。',
    ]
    
    # 维普认知特征注入
    VIP_COGNITIVE_INJECTIONS = [
        '说起来，我之前遇到过类似的情况。',
        '根据我的观察，这个问题其实更复杂一些。',
        '不过，需要考虑实际情况可能有偏差。',
        '有个疑问：这个结论在实际中是否完全适用？',
        '基于经验来看，可能还需要考虑其他因素。',
        '不过我的理解可能有限，欢迎指正。',
        '有意思的是，实操中可能会遇到各种意外。',
        '从经验判断，这里可能需要进一步验证。',
    ]
    
    def __init__(self):
        super().__init__(name="VIP Humanizer")
    
    def humanize(self, text: str) -> Tuple[str, List[str]]:
        """
        对文本进行维普定向人类化处理
        
        Args:
            text: 要人类化的文本
            
        Returns:
            (humanized_text, list_of_changes)
        """
        changes = []
        result = text
        
        # 1. 打破维普语义指纹
        result, new_changes = self._break_semantic_fingerprints(result)
        changes.extend(new_changes)
        
        # 2. 打破维普机械模式
        result, new_changes = self._break_mechanical_patterns(result)
        changes.extend(new_changes)
        
        # 3. 注入维普认知特征
        result, new_changes = self._inject_cognitive_features(result)
        changes.extend(new_changes)
        
        # 4. 变化句式结构
        result, new_changes = self._vary_sentence_structures(result)
        changes.extend(new_changes)
        
        # 5. 处理可疑数据
        result, new_changes = self._process_suspicious_data(result)
        changes.extend(new_changes)
        
        return result, changes
    
    def _break_semantic_fingerprints(self, text: str) -> Tuple[str, List[str]]:
        """打破维普关注的语义指纹"""
        return self._replace_patterns(text, self.VIP_FINGERPRINT_REPLACEMENTS)
    
    def _break_mechanical_patterns(self, text: str) -> Tuple[str, List[str]]:
        """打破维普检测的机械模式"""
        return self._random_insert(text, self.VIP_MECHANICAL_BREAKERS, 0.35)
    
    def _inject_cognitive_features(self, text: str) -> Tuple[str, List[str]]:
        """注入维普定向的认知特征"""
        return self._random_insert(text, self.VIP_COGNITIVE_INJECTIONS, 0.25)
    
    def _vary_sentence_structures(self, text: str) -> Tuple[str, List[str]]:
        """变化句式结构，避免重复"""
        sentences = self._split_sentences(text, chinese=True)
        result = []
        changes = []
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            if len(sent.strip()) > 20 and random.random() < 0.3:
                # 随机添加一些变化前缀
                prefixes = ['不过，', '实际上，', '说起来，', '坦白说，']
                prefix = random.choice(prefixes)
                result.append(prefix + sent)
                changes.append(f"添加前缀: {prefix}")
            else:
                result.append(sent)
            
            result.append(punct)
        
        return ''.join(result), changes
    
    def _process_suspicious_data(self, text: str) -> Tuple[str, List[str]]:
        """处理维普检测的可疑数据"""
        changes = []
        result = text
        
        # 降低数据精度，处理过于精确的数字
        precise_number_pattern = r'(\d+)\.\d{2,}'
        
        def reduce_precision(match):
            num_part = match.group(1)
            changes.append(f"降低数据精度: {match.group(0)} -> {num_part}")
            return num_part
        
        result = re.sub(precise_number_pattern, reduce_precision, result)
        
        # 替换夸张的倍数表达
        exaggerated_patterns = [
            (r'(\d+)倍(?:增长|提高)', '大幅增长'),
            (r'(?:高达|约为|接近|约)(\d+)%', '显著增长'),
        ]
        
        for pattern, replacement in exaggerated_patterns:
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result)
                changes.append(f"替换夸张表达: {pattern[:30]}...")
        
        return result, changes
