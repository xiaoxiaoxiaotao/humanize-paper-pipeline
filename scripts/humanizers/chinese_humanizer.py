import re
import random
from typing import Dict, List, Tuple, Optional

from .base_humanizer import BaseHumanizer


class ChineseHumanizer(BaseHumanizer):

    AI_TEMPLATE_REPLACEMENTS = {
        # 总结类 - 替换为更自然的表达
        '综上所述': '所以',
        '总而言之': '因此',
        '由此可见': '可见',
        '总之': '因此',
        
        # 强调类 - 直接删除或简化为自然表达
        '值得注意的是': '',
        '需要强调的是': '',
        '需要指出的是': '',
        '值得注意的是，': '',
        '需要强调的是，': '',
        '需要指出的是，': '',
        
        # 程度修饰 - 简化为直接表述
        '在一定程度上': '',
        '在某种程度上': '',
        '事实上，': '',
        '实际上，': '',
        '显而易见': '',
        '毫无疑问': '',
        '不言而喻': '',
        '毋庸置疑': '',
        '众所周知': '',
        
        # 过渡词 - 删除大部分，让逻辑自然推进
        '总体来说': '',
        '总体而言': '',
        '一般而言': '',
        '简而言之': '',
        '进一步而言': '',
        '进一步说': '',
        '换言之': '',
        '具体而言，': '',
        '也就是说，': '',
        '从整体而言': '',
        '从整体来看': '',
        '从总体来看': '',
        '从总体而言': '',
        '不可否认': '',
        
        # 开头模板 - 删除"随着""基于"等
        '随着': '',
        '基于': '用',
        '旨在': '为了',
        '本文旨在': '本文',
        '本文将重点研究': '本文研究',
        '本文拟': '本文',
        '本文首先': '',
        '本文其次': '',
        '本文最后': '',
        
        # 重要作用类 - 替换为具体描述
        '发挥着重要作用': '有助于',
        '发挥着关键作用': '对...关键',
        '发挥着不可忽视的作用': '有助于',
        '具有重要意义': '对...重要',
        '具有重要价值': '有...价值',
        '具有重要的现实意义': '实际意义是',
        '具有重要的应用价值': '应用价值是',
        '具有重大意义': '意义重大',
        '具有深远意义': '影响深远',
        '具有广阔前景': '前景好',
        '具有很大的价值': '有价值',
        '具有重要的作用': '有作用',
        '具有显著的优势': '优势是',
        '具有明显的特征': '特征是',
        
        # 强调性形容词 - 删除或简化
        '不可或缺': '必要',
        '至关重要': '关键',
        '举足轻重': '重要',
        '首要步骤': '第一步',
        '首要任务': '核心任务',
        '首要目标': '核心目标',
        
        # 副词修饰 - 直接删除
        '极大地': '',
        '显著地': '',
        '有效地': '',
        '成功地': '',
        '创新性地': '',
        '充分地': '',
        '深入地': '',
        '全面地': '',
        '系统地': '',
        
        # 动词短语 - 简化
        '极大地促进了': '促进了',
        '显著地改善了': '改善了',
        '有效地提高了': '提高了',
        '有效地解决了': '解决了',
        '显著提升了': '提升了',
        '有效提升了': '提升了',
        '成功实现了': '实现了',
        '创新性地提出了': '提出了',
        '催生了': '带来',
        '开创了': '提出',
        '推动了发展': '促进发展',
        '提供了有力': '提供',
        '奠定了坚实': '奠定',
        '打下了坚实': '打下',
        '提供了坚实': '提供',
        
        # 角色类 - 替换为更自然的表达
        '起到了关键作用': '起关键作用',
        '起到了重要作用': '起重要作用',
        '扮演着重要角色': '是重要角色',
        '占据着重要地位': '地位重要',
        '具有不可替代的作用': '作用重要',
        '扮演着': '是',
        '扮演了': '是',
        '充当着': '是',
        '充当了': '是',
        
        # 比喻性表达 - 替换为直接表述
        '成为关键': '是关键',
        '重要引擎': '重要动力',
        '强有力支撑': '支撑',
        '强有力的支撑': '支撑',
        '坚实基础': '基础',
        '真正意义上的': '',
        
        # 成语/四字词 - 替换为平实表达
        '日益逼近': '接近',
        '日益增长': '增长',
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
        
        # 形容词组合 - 简化
        '强大且高效': '高效',
        '高效且': '高效',
        '有效且': '有效',
        '富含语义': '语义丰富',
        '高维的': '高维',
        
        # 技术术语 - 保持专业但简化表达
        '送入': '输入',
        '输入到': '传入',
        '被广泛应用于': '广泛用于',
        '被广泛使用于': '广泛用于',
        '被广泛采用': '广泛采用',
        '被普遍接受': '普遍接受',
        '被证实': '证实',
        '被验证': '验证',
        '被认为是': '是',
        '可以被认为是': '是',
        '可以被视为': '是',
        '被定义为': '定义为',
        '被称为': '称为',
        
        # 目的状语 - 简化
        '为了能够': '为了',
        '为了可以': '为了',
        '基于以下原因': '因为',
        '基于上述原因': '因此',
        '在...背景下': '在...下',
        '在当前背景下': '当前',
        '在现有条件下': '现有条件下',
        
        # 分析/研究表达 - 简化
        '通过对...的分析': '分析',
        '通过对...的研究': '研究',
        '对...进行分析': '分析',
        '对...进行研究': '研究',
        '从...角度': '从...看',
        '从...方面': '从...看',
        '在...方面': '在...上',
        '在...领域': '在...中',
        '针对...问题': '对...问题',
        '关于...的研究': '...研究',
        '有关...的': '...的',
        
        # 动词替换 - 使用更自然的表达
        '涉及到': '涉及',
        '关联到': '关联',
        '依赖于': '依赖',
        '取决于': '取决于',
        '归因于': '因为',
        '归结为': '归为',
        '表现为': '表现为',
        '体现为': '体现',
        '呈现为': '呈现',
        '转化为': '转为',
        '演变为': '演变',
        '发展为': '发展',
        '扩展为': '扩展',
        '延伸为': '延伸',
        '简化为': '简化',
        '优化为': '优化',
        '改进为': '改进',
        '提升为': '提升',
        '增强为': '增强',
        '削弱为': '削弱',
        '降低为': '降低',
        '减少为': '减少',
        '增加为': '增加',
        '扩大为': '扩大',
        '缩小为': '缩小',
        
        # 学术套话 - 简化
        '逐级抽象': '逐层抽象',
        '制约着': '限制',
        '性能上限': '性能上限',
        '在很大程度上影响着': '影响',
        '在很大程度上': '',
        '构成...的基本运算': '是...的基本运算',
        '可以看作': '可以是',
        '可以视为': '可以是',
        '逐步生成': '生成',
        '逐步构建': '构建',
        '逐步形成': '形成',
        '高维且': '高维',
        '语义信息丰富的': '语义丰富',
        '语义信息丰富': '语义丰富',
        
        # 技术词汇 - 保持专业
        '表征能力': '表示能力',
        '骨干网络': '主干网络',
        '特征提取能力': '提取能力',
        '后续层': '后续层',
        '输入规格': '输入尺寸',
        '计算量以及': '计算量和',
        '局部区域': '局部',
        '加权和': '加权和',
        '扫描整个': '扫描',
        '滑动': '滑动',
        '基于此': '因此',
        
        # 效果描述 - 简化
        '实现了良好': '实现',
        '取得了良好的效果': '有效果',
        '取得了显著的效果': '效果显著',
        '为...提供了': '提供',
        '为...指明了方向': '指明方向',
        '核心思想': '核心思路',
        '设计理念': '设计思路',
        '端到端': '端到端',
        '吸引了大量关注': '受关注',
        '引导了发展方向': '引导方向',
        '实现了超越': '超越',
        '带来了性能上的提升': '提升性能',
        '带来了提升': '提升',
        '不断引入': '引入',
        '逐步弥合': '弥合',
        '成为首选方案': '成为首选',
        '首选方案': '首选',
        '向着更加': '向着',
        '颠覆性的': '颠覆性',
        '大量关注': '关注',
        '发展方向': '方向',
        '协同优化': '协同优化',
        '有效的算法支持': '算法支持',
        '有效的支持': '支持',
        '基础框架': '框架',
        '轻量计算量': '轻量计算',
        '轻量级': '轻量',
        '兼顾性能': '兼顾性能',
        '频域特征': '频域特征',
        '特征建模': '特征建模',
        '特征增强': '特征增强',
        '决定了...上限': '决定上限',
        '决定了...性能': '决定性能',
        '决定了...质量': '决定质量',
        '富含': '含',
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
