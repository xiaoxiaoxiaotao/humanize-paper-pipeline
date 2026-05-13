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
        '扮演着': ['充当了', '作为', '承担'],
        '扮演了': ['充当了', '作为', '承担了'],
        '直接决定了': ['在很大程度上影响着', '与...密切相关', '制约着'],
        '决定了': ['影响着', '关系到', '制约着'],
        '是...的前提': ['为...奠定了基础', '对...至关重要'],
        '是...的基础': ['支撑着', '为...提供了条件'],
        '是...的关键': ['在很大程度上取决于', '对...有决定性影响'],
        '强大且高效': ['性能出色', '表现优异', '兼具性能优势'],
        '高效且': ['性能好且', '速度快且'],
        '富含语义': ['语义丰富', '包含大量语义'],
        '高维的': ['高维度'],
        '首要步骤': ['第一步', '最先要做的事情', '基础环节'],
        '首要任务': ['核心任务', '重点'],
        '送入': ['输入', '传入', '交给'],
        '输入到': ['传入', '交给'],
        '当前': ['现阶段', '目前来看', '就目前而言'],
        '无论': ['不管', '不论', '任'],
        '因此': ['所以', '于是', '基于此', '正因如此'],
        '以获取': ['从而得到', '进而获得', '用来提取'],
        '以提升': ['从而提高', '进而改善'],
        '不仅是': ['不只是', '不光是', '不单单是'],
        '不仅': ['不只', '不光', '不单'],
        '其质量也': ['质量也', '它的质量也', '这些特征图的质量'],
        '直接输入': ['输入来源', '基础'],
        '检测系统的上限': ['检测系统的性能边界', '检测效果的上限'],
        '构建高性能': ['搭建高性能', '设计高性能', '开发高性能'],
        '是构建': ['是搭建', '是设计', '是开发'],
    }

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

    MECHANICAL_BREAKERS = [
        '但这里有个问题：',
        '不过我们需要思考的是：',
        '有意思的是，实际情况可能更复杂。',
        '说起来，这里有个细节需要注意。',
        '从实际经验来看，情况可能不太一样。',
        '坦白说，这个问题没那么简单。',
        '不过，这个观点或许需要修正。',
    ]

    SENTENCE_PREFIX_VARIATIONS = [
        '简单来说，',
        '具体而言，',
        '实际上，',
        '不过，',
        '从另一个角度看，',
        '值得注意的是，',
        '坦白讲，',
        '客观来说，',
    ]

    ABSOLUTE_LANGUAGE_SOFTENERS = [
        ('直接决定了', '在很大程度上影响着'),
        ('直接决定', '在很大程度上影响'),
        ('决定了', '影响着'),
        ('是...的前提', '为...创造了条件'),
        ('是...的基础', '为...提供了支撑'),
        ('是...的关键', '对...有重要影响'),
        ('毫无疑问', '总体来看'),
        ('显然', '不难发现'),
        ('必然', '往往'),
        ('彻底', '较大程度上'),
        ('完全', '很大程度上'),
    ]

    ROLE_PLAYING_ALTERNATIVES = [
        ('扮演着', '充当了'),
        ('扮演了', '承担了'),
        ('充当着', '作为'),
        ('起着', '发挥着'),
        ('起到', '发挥'),
    ]

    def __init__(self):
        super().__init__(name="Chinese Humanizer")

    def humanize(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        result, new_changes = self._break_semantic_fingerprints(result)
        changes.extend(new_changes)

        result, new_changes = self._break_mechanical_patterns(result)
        changes.extend(new_changes)

        result, new_changes = self._inject_cognitive_features(result)
        changes.extend(new_changes)

        result, new_changes = self._vary_sentence_structures(result)
        changes.extend(new_changes)

        result, new_changes = self._vary_sentence_lengths(result)
        changes.extend(new_changes)

        result, new_changes = self._soften_absolute_language(result)
        changes.extend(new_changes)

        result, new_changes = self._break_role_playing(result)
        changes.extend(new_changes)

        result, new_changes = self._break_correlative_conjunctions(result)
        changes.extend(new_changes)

        result, new_changes = self._break_premise_conclusion(result)
        changes.extend(new_changes)

        return result, changes

    def _break_semantic_fingerprints(self, text: str) -> Tuple[str, List[str]]:
        return self._replace_patterns(text, self.ADVERSARIAL_PATTERNS)

    def _break_mechanical_patterns(self, text: str) -> Tuple[str, List[str]]:
        return self._random_insert(text, self.MECHANICAL_BREAKERS, 0.35)

    def _inject_cognitive_features(self, text: str) -> Tuple[str, List[str]]:
        return self._random_insert(text, self.COGNITIVE_FEATURES, 0.25)

    def _vary_sentence_structures(self, text: str) -> Tuple[str, List[str]]:
        sentences = self._split_sentences(text, chinese=True)
        result = []
        changes = []

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''

            if len(sent.strip()) > 20 and random.random() < 0.25:
                prefix = random.choice(self.SENTENCE_PREFIX_VARIATIONS)
                result.append(prefix + sent)
                changes.append(f"添加句式前缀: {prefix}")
            else:
                result.append(sent)

            result.append(punct)

        return ''.join(result), changes

    def _vary_sentence_lengths(self, text: str) -> Tuple[str, List[str]]:
        sentences = self._split_sentences(text, chinese=True)
        result = []
        changes = []

        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else ''

            if len(sent) > 40 and random.random() < 0.3:
                candidates = []
                for j, ch in enumerate(sent):
                    if ch == '，' and j >= len(sent) // 2:
                        before = sent[:j]
                        after = sent[j+1:]
                        open_parens = before.count('（') + before.count('(') + before.count('[')
                        close_parens = before.count('）') + before.count(')') + before.count(']')
                        if open_parens == close_parens:
                            candidates.append(j)

                if candidates:
                    comma_pos = candidates[len(candidates) // 2]
                    part1 = sent[:comma_pos]
                    part2 = sent[comma_pos+1:]
                    result.append(part1)
                    result.append('。')
                    result.append(part2)
                    result.append(punct)
                    changes.append("拆分长句")
                else:
                    result.append(sent)
                    result.append(punct)
            else:
                result.append(sent)
                result.append(punct)

        return ''.join(result), changes

    def _soften_absolute_language(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        for old, new in self.ABSOLUTE_LANGUAGE_SOFTENERS:
            if old in result and random.random() < 0.7:
                result = result.replace(old, new, 1)
                changes.append(f"软化绝对化语言: {old} -> {new}")

        return result, changes

    def _break_role_playing(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        for old, new in self.ROLE_PLAYING_ALTERNATIVES:
            if old in result and random.random() < 0.6:
                result = result.replace(old, new, 1)
                changes.append(f"替换角色扮演表达: {old} -> {new}")

        return result, changes

    def _break_correlative_conjunctions(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        patterns = [
            (r'不仅是[^，。]{2,20}也是', lambda m: m.group(0).replace('不仅是', '不只是').replace('也是', '同样是')),
            (r'不仅是[^，。]{2,20}还', lambda m: m.group(0).replace('不仅是', '不光是').replace('还', '而且')),
            (r'无论是[^，。]{2,20}还是', lambda m: m.group(0).replace('无论是', '不管').replace('还是', '还是')),
            (r'无论[^，。]{2,20}其', lambda m: m.group(0).replace('无论', '不论')),
        ]

        for pattern, repl_func in patterns:
            if random.random() < 0.5:
                new_result, count = re.subn(pattern, repl_func, result, count=1)
                if count > 0:
                    result = new_result
                    changes.append("打破关联词结构")

        return result, changes

    def _break_premise_conclusion(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        patterns = [
            (r'因此[，,][^，。]{2,30}是[^，。]{2,30}的前提',
             lambda m: m.group(0).replace('因此', '正因如此').replace('是', '可以说是').replace('的前提', '的重要条件')),
            (r'因此[，,][^，。]{2,30}选择',
             lambda m: m.group(0).replace('因此', '基于上述原因').replace('选择', '倾向于选择')),
            (r'因此[，,][^，。]{2,30}需要',
             lambda m: m.group(0).replace('因此', '由此来看').replace('需要', '有必要')),
        ]

        for pattern, repl_func in patterns:
            if random.random() < 0.6:
                new_result, count = re.subn(pattern, repl_func, result, count=1)
                if count > 0:
                    result = new_result
                    changes.append("打破前提-结论模式")

        return result, changes