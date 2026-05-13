"""
维普AI文本人类化器 - 定向优化版
针对维普AIGC检测的对抗性改写

维普检测重点：
1. 语义指纹：AI高频句式
2. 机械模式："首先...其次...最后"等结构
3. 重复句式：相同句式开头
4. 段落结构：过于均匀的结构
5. 数据真实性：虚构数据
"""

import re
import random
from typing import Dict, List, Tuple


class VIPHumanizer:
    """
    维普AI文本人类化器
    针对维普AIGC检测算法的定向改写
    """
    
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
    
    VIP_MECHANICAL_BREAKERS = [
        '但这里有个问题：',
        '不过我们需要思考的是：',
        '有意思的是，实际情况可能更复杂。',
        '说起来，这里有个细节需要注意。',
        '从实际经验看，情况可能不太一样。',
        '坦白说，这个问题没那么简单。',
        '然而事实并非完全如此。',
        '需要指出的是，上述分析可能存在偏差。',
    ]
    
    COGNITIVE_INJECTIONS = [
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
        self.name = "VIP Humanizer"
    
    def humanize(self, text: str) -> Tuple[str, List[str]]:
        """
        对文本进行人类化改写
        
        Returns:
            Tuple of (humanized_text, list_of_changes)
        """
        changes = []
        result = text
        
        # 1. 打破语义指纹
        result = self._break_semantic_fingerprints(result, changes)
        
        # 2. 打破机械模式
        result = self._break_mechanical_patterns(result, changes)
        
        # 3. 注入认知特征
        result = self._inject_cognitive_features(result, changes)
        
        # 4. 变化句式结构
        result = self._vary_sentence_structures(result, changes)
        
        # 5. 处理可疑数据
        result = self._process_suspicious_data(result, changes)
        
        return result, changes
    
    def _break_semantic_fingerprints(self, text: str, changes: List[str]) -> str:
        """打破维普关注的语义指纹"""
        for old, alternatives in self.VIP_FINGERPRINT_REPLACEMENTS.items():
            if old in text:
                # 随机选择一个替代词
                new_word = random.choice(alternatives)
                # 只替换前几次出现
                count = text.count(old)
                if count > 0:
                    # 替换部分出现
                    parts = text.split(old)
                    new_parts = [parts[0]]
                    for i, part in enumerate(parts[1:]):
                        if i < min(count // 2, 2):
                            new_parts.append(new_word)
                        else:
                            new_parts.append(old)
                        new_parts.append(part)
                    text = ''.join(new_parts)
                    changes.append(f'替换"{old}"为更自然的表达 ({count}处)')
        
        return text
    
    def _break_mechanical_patterns(self, text: str, changes: List[str]) -> str:
        """打破机械的论证模式"""
        sentences = re.split(r'([。！？])', text)
        
        if len(sentences) < 6:
            return text
        
        result = []
        has_inserted = False
        
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            
            result.append(sent)
            result.append(punct)
            
            # 在句子的合适位置插入认知特征
            if not has_inserted and len(sent) > 20 and random.random() < 0.3:
                insertion = random.choice(self.VIP_MECHANICAL_BREAKERS)
                result.append(insertion)
                changes.append(f'打破机械模式: 插入"{insertion[:15]}..."')
                has_inserted = True
        
        return ''.join(result)
    
    def _inject_cognitive_features(self, text: str, changes: List[str]) -> str:
        """注入认知特征 - 模仿人类思维过程"""
        # 在段落开头注入
        paragraphs = re.split(r'\n\s*\n', text)
        
        if len(paragraphs) < 2:
            return text
        
        result = []
        injected = False
        
        for i, para in enumerate(paragraphs):
            if not injected and i > 0 and random.random() < 0.5:
                injection = random.choice(self.COGNITIVE_INJECTIONS)
                result.append(injection)
                result.append(para)
                changes.append(f'注入认知特征: "{injection[:15]}..."')
                injected = True
            else:
                result.append(para)
            
            if i < len(paragraphs) - 1:
                result.append('\n\n')
        
        return ''.join(result)
    
    def _vary_sentence_structures(self, text: str, changes: List[str]) -> str:
        """变化句式结构 - 打破重复模式"""
        sentences = re.split(r'([。！？])', text)
        
        if len(sentences) < 4:
            return text
        
        # 分析句子开头模式
        sentence_patterns = []
        for i in range(0, len(sentences), 2):
            sent = sentences[i].strip()
            if sent and len(sent) >= 4:
                # 提取句式开头特征
                clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
                if len(clean_sent) >= 6:
                    pattern = clean_sent[:6]
                    sentence_patterns.append((i, pattern, sent))
        
        # 找出重复的句式开头
        pattern_counts = {}
        for idx, pattern, sent in sentence_patterns:
            if pattern not in pattern_counts:
                pattern_counts[pattern] = []
            pattern_counts[pattern].append((idx, sent))
        
        # 替换重复句式
        modified = False
        for pattern, occurrences in pattern_counts.items():
            if len(occurrences) >= 3:
                # 保留第一个，其余替换句式
                for occ_idx, (sent_idx, sent) in enumerate(occurrences[1:], 1):
                    # 在句子前添加随机前缀
                    prefixes = ['然而，', '不过，', '实际上，', '说起来，', '坦白说，']
                    prefix = random.choice(prefixes)
                    sentences[sent_idx] = prefix + sentences[sent_idx]
                    changes.append(f'变化句式: 替换"{pattern}"开头的句子')
                    modified = True
        
        return ''.join(sentences)
    
    def _process_suspicious_data(self, text: str, changes: List[str]) -> str:
        """处理可疑数据 - 降低虚构数据特征"""
        # 检测过于精确的百分比
        suspicious_patterns = [
            (r'\d+\.\d{2,}%', lambda m: m.group().replace('%', '').split('.')[0] + '%'),
        ]
        
        for pattern, replacement in suspicious_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # 降低精确度
                text = re.sub(pattern, replacement, text)
                changes.append(f'调整数据精确度: {len(matches)}处')
        
        # 检测可疑的倍数表达
        suspicious_multiples = [
            (r'(\d+)倍(?:增长|提高|上升)', r'大幅提升'),
            (r'(?:高达|约为|接近|约)(\d+)%', r'显著增长'),
        ]
        
        for pattern, replacement in suspicious_multiples:
            matches = re.findall(pattern, text)
            if matches:
                text = re.sub(pattern, replacement, text)
                changes.append(f'替换夸张数据表达: {len(matches)}处')
        
        return text


def humanize_for_vip(text: str) -> Tuple[str, List[str]]:
    """维普人类化入口函数"""
    humanizer = VIPHumanizer()
    return humanizer.humanize(text)
