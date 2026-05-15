import re
import random
from typing import Dict, List, Tuple, Optional

from .base_humanizer import BaseHumanizer


class ChineseHumanizer(BaseHumanizer):

    AI_TEMPLATE_REPLACEMENTS = {
        '综上所述': '所以',
        '总而言之': '所以',
        '由此可见': '可见',
        '值得注意的是': '要注意',
        '需要强调的是': '要强调',
        '需要指出的是': '要指出',
        '在一定程度上': '部分',
        '在某种程度上': '部分',
        '事实上，': '其实，',
        '实际上，': '其实，',
        '本质上，': '本质上，',
        '显而易见': '明显',
        '毫无疑问': '确实',
        '不言而喻': '自然',
        '毋庸置疑': '确实',
        '众所周知': '大家知道',
        '总体来说': '整体看',
        '总体而言': '整体看',
        '一般而言': '通常',
        '简而言之': '简单说',
        '进一步而言': '进一步',
        '进一步说': '进一步',
        '换言之': '换句话说',
        '具体而言，': '具体来说，',
        '也就是说，': '即，',
        '从整体而言': '整体看',
        '从整体来看': '整体看',
        '从总体来看': '总体看',
        '从总体而言': '总体看',
        '可以说': '可以说',
        '不可否认': '确实',
        '毋庸置疑地': '确实',
        '毋庸置疑的': '确实的',
        '众所周知地': '大家知道',
        '众所周知的是': '大家知道的是',
        '随着': '',
        '基于': '使用',
        '旨在': '为了',
        '本文旨在': '本文为了',
        '本文将重点研究': '本文研究',
        '本文拟': '本文',
        '发挥着重要作用': '很重要',
        '发挥着关键作用': '很关键',
        '具有重要意义': '很重要',
        '具有重要价值': '很有价值',
        '具有重要的现实意义': '有实际用途',
        '具有重要的应用价值': '有应用价值',
        '具有重大意义': '很重要',
        '具有深远意义': '影响深远',
        '具有广阔前景': '前景广阔',
        '不可或缺': '必要',
        '至关重要': '关键',
        '举足轻重': '重要',
        '扮演着': '作为',
        '扮演了': '作为',
        '充当着': '作为',
        '充当了': '作为',
        '首要步骤': '第一步',
        '首要任务': '核心任务',
        '首要目标': '核心目标',
        '直接决定了': '影响着',
        '直接决定': '影响',
        '核心痛点': '主要问题',
        '有效解决了': '解决了',
        '极大地': '大幅',
        '催生了': '带来了',
        '开创了': '提出了',
        '推动了发展': '促进了发展',
        '提供了有力': '提供了',
        '奠定了坚实': '奠定了',
        '打下了坚实': '打下了',
        '提供了坚实': '提供了',
        '起到了关键作用': '起到了作用',
        '起到了重要作用': '起到了作用',
        '扮演着重要角色': '是重要角色',
        '占据着重要地位': '地位重要',
        '具有不可替代的作用': '作用重要',
        '成为关键': '很关键',
        '重要引擎': '重要动力',
        '强有力支撑': '支撑',
        '强有力的支撑': '支撑',
        '坚实基础': '基础',
        '真正意义上的': '真正的',
        '创新性地': '创新地',
        '显著提升了': '提升了',
        '有效提升了': '提升了',
        '实现了显著': '实现了',
        '日益逼近': '逐渐接近',
        '日益增长': '逐渐增长',
        '蓬勃发展': '发展',
        '突飞猛进': '快速发展',
        '日新月异': '快速变化',
        '势在必行': '必须',
        '应运而生': '出现',
        '层出不穷': '不断出现',
        '相辅相成': '互相配合',
        '密不可分': '紧密相关',
        '息息相关': '相关',
        '行之有效': '有效',
        '卓有成效': '有效',
        '立竿见影': '立即见效',
        '强大且高效': '高效',
        '高效且': '高效',
        '有效且': '有效',
        '富含语义': '语义丰富',
        '高维的': '高维度',
        '送入': '输入',
        '输入到': '传入',
    }

    AI_ENUM_REPLACEMENTS = [
        (r'首先[，,]', '一方面，'),
        (r'其次[，,]', '另一方面，'),
        (r'再次[，,]', '另外，'),
        (r'最后[，,]', '最终，'),
        (r'第一[，,]', '其一，'),
        (r'第二[，,]', '其二，'),
        (r'第三[，,]', '其三，'),
    ]

    SUIZHE_PATTERN = re.compile(r'随着([^，。的了]+?的[^，。]*)[，,]')
    JIYU_PATTERN = re.compile(r'基于([^，。的了]+?的[^，。]*)')
    TONGGUO_PATTERN = re.compile(r'通过([^，。的了]+?的[^，。]*)')

    def __init__(self):
        super().__init__(name="Chinese Humanizer")

    def humanize(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        result, new_changes = self._replace_ai_templates(result)
        changes.extend(new_changes)

        result, new_changes = self._fix_suizhe_pattern(result)
        changes.extend(new_changes)

        result, new_changes = self._fix_jiyu_pattern(result)
        changes.extend(new_changes)

        result, new_changes = self._fix_tongguo_pattern(result)
        changes.extend(new_changes)

        result, new_changes = self._replace_enum_words(result)
        changes.extend(new_changes)

        result, new_changes = self._replace_em_dashes(result)
        changes.extend(new_changes)

        result, new_changes = self._clean_extra_punctuation(result)
        changes.extend(new_changes)

        return result, changes

    def _replace_ai_templates(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text
        for old, new in self.AI_TEMPLATE_REPLACEMENTS.items():
            if old in result:
                result = result.replace(old, new)
                changes.append(f"替换AI模板: {old} -> {new or '(删除)'}")
        return result, changes

    def _fix_suizhe_pattern(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        def replace_suizhe(m):
            content = m.group(1)
            changes.append(f"修复'随着'模板: 随着{content}， -> {content}，")
            return f"{content}，"

        result = self.SUIZHE_PATTERN.sub(replace_suizhe, result)
        return result, changes

    def _fix_jiyu_pattern(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        def replace_jiyu(m):
            content = m.group(1)
            if content.endswith('的'):
                new_content = content[:-1]
            else:
                new_content = content
            changes.append(f"修复'基于'模板: 基于{content} -> 使用{new_content}")
            return f"使用{new_content}"

        result = self.JIYU_PATTERN.sub(replace_jiyu, result)
        return result, changes

    def _fix_tongguo_pattern(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text

        def replace_tongguo(m):
            content = m.group(1)
            if content.endswith('的'):
                new_content = content[:-1]
            else:
                new_content = content
            changes.append(f"修复'通过'模板: 通过{content} -> 利用{new_content}")
            return f"利用{new_content}"

        result = self.TONGGUO_PATTERN.sub(replace_tongguo, result)
        return result, changes

    def _replace_enum_words(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text
        for pattern, replacement in self.AI_ENUM_REPLACEMENTS:
            new_result, count = re.subn(pattern, replacement, result)
            if count > 0:
                result = new_result
                changes.append(f"替换编号词: {pattern} -> {replacement}")
        return result, changes

    def _replace_em_dashes(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text
        if '——' in result:
            result = result.replace('——', '，即')
            changes.append("替换破折号为'，即'")
        return result, changes

    def _clean_extra_punctuation(self, text: str) -> Tuple[str, List[str]]:
        changes = []
        result = text
        result = re.sub(r'^[，,]', '', result)
        result = re.sub(r'([。！？])\s*[，,]', r'\1', result)
        result = re.sub(r'[，,]{2,}', '，', result)
        result = re.sub(r'。\s*。', '。', result)
        return result, changes
