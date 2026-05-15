"""
中文AI文本检测器 - 知网3.0算法适配
包含维普、知网等平台的常见检测特征
"""

import re
import statistics
from collections import Counter
from typing import Dict, List, Tuple, Optional

from .base_detector import BaseDetector


class ChineseDetector(BaseDetector):
    """
    中文AI文本检测器
    集成知网3.0、万方等平台的最新检测特征
    """

    AI_TRANSITIONS = [
        '首先', '其次', '再次', '最后', '综上所述',
        '值得注意的是', '需要强调的是', '进一步而言',
        '总体来说', '一般而言', '事实上',
        '随着', '基于', '通过', '利用',
        '然而', '因此', '此外', '另外', '同时',
        '从而', '进而', '与此同时', '不仅如此',
        '进一步说', '换言之', '简而言之',
        '本文', '本研究', '本文旨在', '本文拟', '总之', '总而言之',
        '由此可见', '显然', '毫无疑问',
        '旨在', '可分为', '大致可分为', '主要包括', '具体包括',
        '催生了', '开创了', '分为', '替换为', '替换为了',
        '避免了', '此后', '在此基础上',
        '尽管', '虽然', '甚至', '首次', '逐步', '日益',
        '特别是', '尤其是', '如今', '目前',
        '极大', '极大地',
        '创新性地', '该改进', '深入分析',
        '显著提升了', '有效提升了', '实现了显著',
        '当前', '现阶段', '无论', '不论',
    ]

    AI_HIGH_FREQ_VERBS = [
        '融合', '优化', '采用', '实现',
        '提升', '增强', '引入', '构建',
    ]

    AI_HIGH_FREQ_NOUNS = [
        '效率', '鲁棒性', '范式', '痛点',
        '表征能力', '骨干网络', '性能上限',
    ]

    ABSTRACT_PHRASES = [
        '具有重要意义', '发挥着重要作用', '至关重要',
        '不可或缺', '具有重要价值', '促进了发展',
        '综上所述', '基于此',
        '具有重要的现实意义', '具有重要的应用价值',
        '具有重大意义', '成为关键', '重要引擎',
        '提供了有力', '实现了良好', '推动了发展',
        '具有深远意义', '具有广阔前景',
        '提供了坚实', '奠定了坚实', '打下了坚实',
        '核心痛点', '强有力的支撑',
        '取得了良好的效果',
        '取得了显著的效果',
        '起到了关键作用',
        '起到了重要作用', '扮演着重要角色',
        '占据着重要地位', '具有不可替代的作用',
        '核心思想', '设计理念',
        '吸引了大量关注', '引导了发展方向', '实现了超越',
        '带来了性能上的提升', '带来了提升',
        '不断引入', '逐步弥合',
        '成为首选方案', '首选方案',
        '日益逼近', '向着更加',
        '颠覆性的', '大量关注', '发展方向',
        '真正意义上的',
        '扮演着', '扮演了', '充当着', '充当了',
        '逐级抽象', '制约着', '性能上限',
        '在很大程度上影响着', '在很大程度上',
        '可以看作', '可以视为',
        '逐步生成', '逐步构建', '逐步形成',
        '高维且', '语义信息丰富', '表征能力',
        '骨干网络', '特征提取能力',
        '直接决定了', '直接决定',
        '强大且高效', '高效且', '有效且',
        '富含语义', '高维的',
        '首要步骤', '首要任务', '首要目标',
        '送入',
        '迫切需求', '持续演进', '研究进展',
        '代表性成果', '各有优势与不足',
        '驱动了', '催生了', '开创了',
        '具体挑战包括', '主要包括',
        '奠定了基础', '奠定了理论基础',
        '技术范式', '技术方案',
        '有效支持', '有效保障',
        '显著提升', '有效提升', '大幅提升',
        '实时推理', '自动告警',
        '计算开销', '检测能力',
        '局部细节', '细粒度',
        '不同尺度', '多尺度特征',
        '高分辨率特征', '分辨率特征',
        '奠定了理论基础', '提供了技术基础',
        '智能化升级', '安全管控水平',
        '实际应用需求', '推理延迟低',
        '被广泛应用', '自动化检测',
        '理论实践兼备',
    ]

    TEMPLATE_PATTERNS = [
        r'随着[^，。]{2,40}的[^，。]{2,40}',
        r'基于[^，。]{2,40}的[^，。]{2,40}',
        r'通过[^，。]{2,40}的[^，。]{2,40}',
        r'因此[^，。]{2,40}具有重要意义',
        r'利用[^，。]{2,40}的[^，。]{2,40}',
        r'旨在[^，。]{2,40}',
        r'大致可分为[^，。]{2,40}',
        r'可分为[^，。]{2,20}：',
        r'以下[^，。]{2,30}阐述',
        r'是[^，。]{2,30}之一',
        r'为[^，。]{2,30}提供了[^，。]{2,40}',
        r'[^，。]{2,20}等工作[^，。]{2,30}',
        r'催生了[^，。]{2,40}',
        r'开创了[^，。]{2,20}先河',
        r'为了解决[^，。]{2,30}',
        r'核心[^，。]{2,10}是[^，。]{2,40}',
        r'极大地[^，。]{2,40}',
        r'带来了[^，。]{2,30}',
        r'尽管[^，。]{2,30}但[^，。]{2,30}',
        r'首次提出了[^，。]{2,40}',
        r'成为[^，。]{2,20}的[^，。]{2,20}',
        r'推动着[^，。]{2,30}',
        r'在[^，。]{2,15}上存在[^，。]{2,20}',
        r'特别是[^，。]{2,30}',
        r'日益[^，。]{2,20}',
        r'真正意义上的[^，。]{2,20}',
        r'第[^，。]{0,5}章旨在[^，。]{2,40}',
        r'兼顾[^，。]{2,20}和[^，。]{2,30}',
        r'针对[^，。]{2,30}等[^，。]{2,10}',
        r'以[^，。]{2,30}为基础[^，。]{2,30}',
        r'深入分析[^，。]{2,40}',
        r'创新性地[^，。]{2,40}',
        r'设计了[^，。]{2,30}',
        r'进一步构建[^，。]{2,30}',
        r'替代原[^，。]{2,20}',
        r'在[^，。]{2,20}的前提下[^，。]{2,30}',
        r'显著提升了[^，。]{2,40}',
        r'为[^，。]{2,30}提供了[^，。]{2,30}支持',
        r'实现[^，。]{2,20}协同[^，。]{2,20}',
        r'以获取[^，。]{2,40}',
        r'以提升[^，。]{2,40}',
        r'以达到[^，。]{2,40}',
        r'将[^，。]{2,30}送入[^，。]{2,30}',
        r'将[^，。]{2,30}输入[^，。]{2,30}',
        r'无论是[^，。]{2,30}还是[^，。]{2,30}',
        r'不仅是[^，。]{2,20}也是[^，。]{2,20}',
        r'不仅是[^，。]{2,20}其[^，。]{2,20}也',
        r'不仅是[^，。]{2,20}还[^，。]{2,20}',
        r'是[^，。]{2,30}的前提',
        r'是[^，。]{2,30}的基础',
        r'是[^，。]{2,30}的关键',
        r'逐级抽象[^，。]{2,40}',
        r'制约着[^，。]{2,40}',
        r'在很大程度上影响着[^，。]{2,40}',
        r'在很大程度上[^，。]{2,40}',
        r'构成[^，。]{2,20}的基本运算',
        r'可以看作[^，。]{2,40}',
        r'可以视为[^，。]{2,40}',
        r'逐步生成[^，。]{2,40}',
        r'逐步构建[^，。]{2,40}',
        r'逐步形成[^，。]{2,40}',
        r'高维且[^，。]{2,40}',
        r'语义信息丰富[^，。]{2,40}',
        r'表征能力[^，。]{2,40}',
        r'骨干网络[^，。]{2,40}',
        r'特征提取能力[^，。]{2,40}',
        r'后续层[^，。]{2,40}',
        r'输入规格[^，。]{2,40}',
        r'局部区域[^，。]{2,40}',
        r'扫描整个[^，。]{2,40}',
        r'直接决定了[^，。]{2,40}',
        r'决定了[^，。]{2,20}的[^，。]{2,20}',
        r'扮演着[^，。]{2,20}的角色',
        r'扮演了[^，。]{2,20}的角色',
        r'充当着[^，。]{2,20}的角色',
        r'无论[^，。]{2,30}其[^，。]{2,30}',
        r'当前[，,][^，。]{2,40}无论',
        r'这个过程[^，。]{2,40}',
        r'因此[，,][^，。]{2,20}是[^，。]{2,30}的前提',
        r'因此研发[^，。]{2,40}',
        r'驱动了[^，。]{2,40}',
        r'通过[^，。]{2,30}实现了[^，。]{2,40}',
        r'能够[^，。]{2,10}实现[^，。]{2,40}',
        r'进一步[^，。]{2,40}',
        r'以满足[^，。]{2,40}',
        r'进而提升[^，。]{2,40}',
        r'从而实现[^，。]{2,40}',
        r'具体挑战包括[^，。]{2,40}',
        r'代表性成果[^，。]{2,40}',
        r'各有优势与不足',
        r'技术范式[^，。]{2,40}',
        r'有效支持[^，。]{2,40}',
        r'有效保障[^，。]{2,40}',
    ]

    HEDGE_WORDS = [
        '似乎', '似乎在', '似乎表明', '似乎说明',
        '可能表明', '可能说明', '可能意味着',
        '或许', '或许可以', '或可', '或可为',
        '潜在地', '倾向于',
        '在一定程度上', '在某种程度上', '某种程度上',
        '有可能', '不排除', '不能完全排除',
        '可以认为', '可以推测', '可以推断',
        '初步表明', '初步显示',
    ]

    CONCLUDING_FORMULAS = [
        '综上所述', '总而言之', '总之', '由此可见',
        '因此，本研究', '因此，本文', '综上',
        '通过以上分析', '基于以上分析',
        '从以上分析可以看出', '以上结果表明',
        '因此，选择', '因此，采用', '因此，提出',
        '因此，构建', '因此，设计',
    ]

    IDIOM_PATTERNS = [
        '不可或缺', '至关重要', '举足轻重', '显而易见',
        '毋庸置疑', '不言而喻', '众所周知',
        '日新月异', '突飞猛进', '蓬勃发展',
        '相辅相成', '密不可分', '息息相关',
        '行之有效', '卓有成效', '立竿见影',
        '层出不穷', '应运而生', '势在必行',
        '先河',
    ]

    PARAGRAPH_TEMPLATE_MARKERS = [
        '随着', '近年来', '当前', '目前', '现阶段',
        '然而', '但是', '因此', '所以',
        '本文', '本研究', '本课题',
    ]

    ABSOLUTE_LANGUAGE = [
        '直接决定了', '直接决定', '决定了',
        '是...的前提', '是...的基础', '是...的关键',
        '毫无疑问', '显然', '显而易见',
        '必然', '必定', '一定',
        '彻底', '完全', '根本上',
        '决定了...上限', '决定了...性能',
        '是...的保证', '是...的核心',
    ]

    CORRELATIVE_CONJUNCTIONS = [
        (r'不仅[^，。]{1,30}也[^，。]{1,30}', '不仅...也...'),
        (r'不仅[^，。]{1,30}还[^，。]{1,30}', '不仅...还...'),
        (r'不仅[^，。]{1,30}而且[^，。]{1,30}', '不仅...而且...'),
        (r'无论是[^，。]{1,30}还是[^，。]{1,30}', '无论是...还是...'),
        (r'不论[^，。]{1,30}都[^，。]{1,30}', '不论...都...'),
        (r'无论[^，。]{1,30}都[^，。]{1,30}', '无论...都...'),
        (r'无论[^，。]{1,30}其[^，。]{1,30}', '无论...其...'),
        (r'既[^，。]{1,15}又[^，。]{1,15}', '既...又...'),
        (r'既[^，。]{1,15}也[^，。]{1,15}', '既...也...'),
        (r'一方面[^，。]{1,30}另一方面', '一方面...另一方面...'),
        (r'不是[^，。]{1,20}而是[^，。]{1,20}', '不是...而是...'),
        (r'不仅[^，。]{1,30}其[^，。]{1,15}也', '不仅...其...也'),
    ]

    AI_ADJECTIVE_PAIRS = [
        r'[^\u4e00-\u9fa5]{0,5}(?:强大|高效|有效|优秀|出色|卓越|先进|创新)[^\u4e00-\u9fa5]{0,5}(?:且|而|又)[^\u4e00-\u9fa5]{0,5}(?:高效|强大|有效|优秀|出色|卓越|先进|创新|精准|稳定|可靠)',
        r'(?:高维|深层|深度|多维|高阶)[^\u4e00-\u9fa5]{0,5}[、，,][^\u4e00-\u9fa5]{0,5}(?:富含|丰富|富集|稠密|密集)',
        r'(?:富含|具有|拥有|包含)[^，。]{1,15}(?:语义|特征|信息|知识)[^\u4e00-\u9fa5]{0,5}(?:表示|表达|表征|特征)',
    ]

    PURPOSE_CLAUSE_PATTERNS = [
        r'以[^，。]{2,40}',
        r'从而[^，。]{2,40}',
        r'进而[^，。]{2,40}',
        r'用于[^，。]{2,40}',
        r'以便[^，。]{2,40}',
        r'旨在[^，。]{2,40}',
    ]

    # AI技术文本特有的检测模式 - 关注结构和表达方式而非专有名词
    TECHNICAL_PATTERNS = [
        # 被动语态模式
        r'[被为][\u4e00-\u9fa5]{1,10}[所的]',
        # 算法组成描述
        r'(?:由|包括|包含)[^，。]{2,40}(?:组成|构成|构成的)',
        # 功能职责描述
        r'(?:负责|用于|用来|旨在)[^，。]{2,40}',
        # 方法手段描述
        r'(?:通过|利用|基于)[^，。]{2,40}(?:方式|方法|算法|网络)',
        # 解决问题模板
        r'(?:解决|克服|突破)[^，。]{2,30}(?:瓶颈|问题|挑战)',
        # 性能提升描述
        r'(?:提升|提高|优化)[^，。]{2,30}(?:性能|效率|精度)',
        # 技术对比模板
        r'(?:相比|相较于|不同于)[^，。]{2,30}(?:传统|以往|现有的)',
        # 损失函数描述
        r'(?:损失函数|Loss)[^，。]{2,30}(?:交叉熵|Softmax|Smooth|L1|L2)',
        # 网络结构描述
        r'(?:网络|模型)[^，。]{2,30}(?:由.*组成|包括.*层)',
        # 端到端训练描述
        r'(?:端到端|端-端)[^，。]{0,20}(?:训练|学习|优化)',
        # 共享特征描述
        r'(?:共享|共用)[^，。]{2,20}(?:特征|权重|参数)',
        # 并行结构描述
        r'(?:并行|同时)[^，。]{2,20}(?:进行|执行|处理)',
        # 协同工作描述
        r'(?:协同|联合)[^，。]{2,20}(?:工作|训练|优化)',
    ]

    CHAIN_DESCRIPTION_PATTERNS = [
        # 链式描述："它...，它..."
        r'它[^，。]{2,30}[，,]它[^，。]{2,30}',
        # 连续被动
        r'被[^，。]{2,20}[，,][被为][^，。]{2,20}',
        # 重复的连接词
        r'(?:通过|利用|基于)[^，。]{2,30}[，,](?:通过|利用|基于)',
    ]

    VIP_SEMANTIC_FINGERPRINTS = [
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

    VIP_DATA_PATTERNS = [
        r'\d{4}年[^，。]{0,20}(?:增长|提高|下降|减少|上升)[^，。]{0,20}\d+(?:\.\d+)?%',
        r'(?:高达|约为|接近|约)\d+(?:\.\d+)?%',
        r'(?:显著|明显|大幅|急剧)(?:增长|提高|下降|减少)',
        r'\d+倍[^，。]{0,10}(?:增长|提高|下降)',
    ]

    def __init__(self):
        super().__init__(name="Chinese AI Detector")

    def detect(self, text: str) -> Tuple[int, Dict]:
        details = {'metrics': {}, 'platform': '知网3.0'}
        score = 0

        sentences = self.split_sentences(text, chinese=True)
        paragraphs = self.split_paragraphs(text)

        if not text.strip():
            return 0, details

        score, details = self._analyze_abstract_language(text, score, details)
        score, details = self._analyze_templates(text, score, details)
        score, details = self._analyze_ai_verb_density(text, score, details)
        score, details = self._analyze_over_hedging(text, score, details)
        score, details = self._analyze_absolute_language(text, score, details)
        score, details = self._analyze_verbose_expressions(text, score, details)
        score, details = self._analyze_concluding_formula(text, score, details)
        score, details = self._analyze_idiom_overuse(text, score, details)
        score, details = self._analyze_technical_patterns(text, score, details)
        score, details = self._analyze_role_playing_pattern(text, score, details)
        score, details = self._analyze_premise_conclusion_pattern(text, score, details)
        score, details = self._analyze_suizhe_template(text, score, details)
        score, details = self._analyze_definition_pattern(text, score, details)
        score, details = self._analyze_ai_adjectives(text, score, details)
        score, details = self._analyze_purpose_clauses(text, score, details)
        score, details = self._analyze_em_dash_overuse(text, score, details)

        if len(sentences) >= 2:
            score, details = self._analyze_sentence_length_distribution(sentences, score, details)
            score, details = self._analyze_paragraph_structure_similarity(paragraphs, sentences, score, details)
            score, details = self._analyze_info_density_distribution(sentences, score, details)
            score, details = self._analyze_transition_word_distribution(text, sentences, score, details)
            score, details = self._analyze_sentence_uniformity(sentences, score, details)
            score, details = self._analyze_transition_overuse(text, score, details)
            score, details = self._analyze_sentence_opening_repetition(sentences, score, details)
            score, details = self._analyze_word_burstiness(text, score, details)
            score, details = self._analyze_bigram_ttr(text, score, details)
            score, details = self._analyze_clause_chain_density(text, sentences, score, details)
            score, details = self._analyze_punctuation_density(text, score, details)
            score, details = self._analyze_paragraph_template(paragraphs, score, details)
            score, details = self._analyze_citation_distribution(text, score, details)
            score, details = self._analyze_correlative_conjunctions(text, score, details)
            score, details = self._analyze_sentence_structure_pattern(sentences, text, score, details)
            score, details = self._analyze_passive_voice_density(text, sentences, score, details)
            score, details = self._analyze_sentence_complexity(text, sentences, score, details)
            score, details = self._analyze_comma_density(text, score, details)
            score, details = self._analyze_chain_description(text, score, details)
            score, details = self._analyze_vip_semantic_fingerprints(text, score, details)
            score, details = self._analyze_vip_mechanical_patterns(text, score, details)
            score, details = self._analyze_vip_data_authenticity(text, score, details)

        final_score = min(100, max(0, score))
        details['overall_score'] = final_score
        details['sentence_count'] = len(sentences)
        details['paragraph_count'] = len(paragraphs)

        return final_score, details

    def _analyze_sentence_length_distribution(self, sentences: List[str],
                                             score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        lengths = []
        for sent in sentences:
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if clean_sent:
                lengths.append(len(clean_sent))

        if len(lengths) < 3:
            return score, details

        avg_len = statistics.mean(lengths)
        std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0
        cv = std_dev / avg_len if avg_len > 0 else 0

        ai_peak_count = sum(1 for l in lengths if 15 <= l <= 25)
        ai_peak_ratio = ai_peak_count / len(lengths)

        metric_score = 0
        if len(lengths) >= 5:
            if ai_peak_ratio > 0.7:
                metric_score += 4
            elif ai_peak_ratio > 0.55:
                metric_score += 2

            if cv < 0.25:
                metric_score += 3
            elif cv < 0.35:
                metric_score += 1
        else:
            if ai_peak_ratio > 0.85:
                metric_score += 2
            elif ai_peak_ratio > 0.75:
                metric_score += 1

            if cv < 0.18:
                metric_score += 2
            elif cv < 0.28:
                metric_score += 1

        details['metrics']['sentence_length_distribution'] = {
            'avg_length': round(avg_len, 1),
            'cv': round(cv, 3),
            'ai_peak_ratio': round(ai_peak_ratio, 2),
            'score': metric_score,
            'details': f'句长均值{avg_len:.1f}，CV={cv:.3f}，15-25字占比{ai_peak_ratio*100:.0f}%'
        }

        return score + metric_score, details

    def _analyze_ai_verb_density(self, text: str, score: int,
                                 details: Dict) -> Tuple[int, Dict]:
        chinese_chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        if len(chinese_chars) < 50:
            return score, details

        verb_types = 0
        verb_count = 0
        found_verbs = []
        for verb in self.AI_HIGH_FREQ_VERBS:
            count = text.count(verb)
            if count > 0:
                verb_types += 1
                verb_count += count
                found_verbs.append(f'{verb}({count})')

        noun_types = 0
        noun_count = 0
        found_nouns = []
        for noun in self.AI_HIGH_FREQ_NOUNS:
            count = text.count(noun)
            if count > 0:
                noun_types += 1
                noun_count += count
                found_nouns.append(f'{noun}({count})')

        total_ai_words = verb_count + noun_count
        total_types = verb_types + noun_types
        density = total_ai_words / (len(chinese_chars) / 100)

        metric_score = 0
        if total_types >= 5:
            metric_score = 15
        elif total_types >= 4:
            metric_score = 10
        elif total_types >= 3:
            metric_score = 6

        if density > 5:
            metric_score += 6
        elif density > 3:
            metric_score += 3

        metric_score = min(20, metric_score)

        details['metrics']['ai_verb_density'] = {
            'verb_count': verb_count,
            'noun_count': noun_count,
            'total_types': total_types,
            'density': round(density, 2),
            'score': metric_score,
            'details': f'AI高频词密度{density:.1f}/100字，动词[{", ".join(found_verbs[:5])}]，名词[{", ".join(found_nouns[:5])}]'
        }

        return score + metric_score, details

    def _analyze_comma_density(self, text: str, score: int,
                               details: Dict) -> Tuple[int, Dict]:
        chinese_chars = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        if len(chinese_chars) < 50:
            return score, details

        comma_count = text.count('，') + text.count(',')
        density = comma_count / (len(chinese_chars) / 100)

        metric_score = 0
        if density > 6.0:
            metric_score = 5
        elif density > 5.0:
            metric_score = 3

        details['metrics']['comma_density'] = {
            'comma_count': comma_count,
            'density': round(density, 2),
            'score': metric_score,
            'details': f'逗号密度{density:.1f}/100字'
        }

        return score + metric_score, details

    def _analyze_paragraph_structure_similarity(self, paragraphs: List[str],
                                                sentences: List[str], score: int,
                                                details: Dict) -> Tuple[int, Dict]:
        if len(paragraphs) < 2:
            return score, details

        paragraph_features = []
        for para in paragraphs:
            para_sentences = [s.strip() for s in re.split(r'[。！？!?]+', para) if s.strip()]
            if para_sentences:
                sent_count = len(para_sentences)
                avg_len = sum(len(s) for s in para_sentences) / sent_count
                has_transition = any(t in para for t in self.AI_TRANSITIONS[:5])
                paragraph_features.append((sent_count, avg_len, has_transition))

        if len(paragraph_features) >= 2:
            similar_count = 0
            total_pairs = len(paragraph_features) * (len(paragraph_features) - 1) / 2

            for i in range(len(paragraph_features)):
                for j in range(i + 1, len(paragraph_features)):
                    sent_diff = abs(paragraph_features[i][0] - paragraph_features[j][0])
                    len_diff = abs(paragraph_features[i][1] - paragraph_features[j][1])
                    trans_same = paragraph_features[i][2] == paragraph_features[j][2]

                    if sent_diff <= 1 and len_diff < 8 and trans_same:
                        similar_count += 1

            similarity_ratio = similar_count / total_pairs if total_pairs > 0 else 0

            metric_score = 0
            if similarity_ratio >= 0.7:
                metric_score = 18
            elif similarity_ratio >= 0.55:
                metric_score = 10
            elif similarity_ratio >= 0.4:
                metric_score = 5

            details['metrics']['paragraph_structure_similarity'] = {
                'similarity_ratio': round(similarity_ratio, 2),
                'paragraph_count': len(paragraphs),
                'score': metric_score,
                'details': f'段落结构相似度{similarity_ratio*100:.0f}%'
            }

            return score + metric_score, details

        return score, details

    def _analyze_info_density_distribution(self, sentences: List[str],
                                           score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        info_densities = []
        for sent in sentences:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', sent))
            total_chars = len(sent)
            if total_chars > 0:
                info_densities.append(chinese_chars / total_chars)

        if len(info_densities) < 3:
            return score, details

        avg_density = statistics.mean(info_densities)
        std_density = statistics.stdev(info_densities) if len(info_densities) > 1 else 0

        in_ai_zone = sum(1 for d in info_densities if 0.65 <= d <= 0.75)
        ai_zone_ratio = in_ai_zone / len(info_densities)

        metric_score = 0
        if ai_zone_ratio > 0.85:
            metric_score += 8
        elif ai_zone_ratio > 0.7:
            metric_score += 4

        if std_density < 0.05:
            metric_score += 5
        elif std_density < 0.08:
            metric_score += 2

        details['metrics']['info_density_distribution'] = {
            'avg_density': round(avg_density, 2),
            'std_density': round(std_density, 3),
            'ai_zone_ratio': round(ai_zone_ratio, 2),
            'score': metric_score,
            'details': f'信息密度均值{avg_density*100:.0f}%，标准差{std_density:.3f}'
        }

        return score + metric_score, details

    def _analyze_transition_word_distribution(self, text: str, sentences: List[str],
                                              score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        transition_count = 0
        transition_positions = []

        for idx, sent in enumerate(sentences):
            for t in self.AI_TRANSITIONS:
                if t in sent:
                    transition_count += 1
                    transition_positions.append(idx)

        text_chars = len(text)
        density_per_1000 = (transition_count / text_chars) * 1000 if text_chars > 0 else 0

        uniformity_score = 0
        if len(transition_positions) >= 3:
            gaps = [transition_positions[i+1] - transition_positions[i]
                   for i in range(len(transition_positions)-1)]
            if gaps:
                avg_gap = statistics.mean(gaps)
                std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
                gap_cv = std_gap / avg_gap if avg_gap > 0 else 0
                uniformity_score = 1 - gap_cv

        metric_score = 0
        if density_per_1000 > 15:
            metric_score += 6
        elif density_per_1000 > 10:
            metric_score += 3

        if uniformity_score > 0.9:
            metric_score += 5
        elif uniformity_score > 0.8:
            metric_score += 2

        details['metrics']['transition_word_distribution'] = {
            'count': transition_count,
            'density_per_1000': round(density_per_1000, 1),
            'uniformity_score': round(uniformity_score, 2),
            'score': metric_score,
            'details': f'连接词{transition_count}个，密度{density_per_1000:.1f}/千字'
        }

        return score + metric_score, details

    def _analyze_sentence_uniformity(self, sentences: List[str], score: int,
                                     details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        variance_ratio = std_dev / avg_len if avg_len > 0 else 0

        metric_score = 0
        if len(sentences) >= 5:
            if avg_len < 30:
                if variance_ratio < 0.18:
                    metric_score = 8
                elif variance_ratio < 0.25:
                    metric_score = 5
                elif variance_ratio < 0.35:
                    metric_score = 2
            else:
                if variance_ratio < 0.25:
                    metric_score = 6
                elif variance_ratio < 0.35:
                    metric_score = 3
        else:
            if avg_len < 30:
                if variance_ratio < 0.15:
                    metric_score = 5
                elif variance_ratio < 0.20:
                    metric_score = 3
            else:
                if variance_ratio < 0.20:
                    metric_score = 4
                elif variance_ratio < 0.30:
                    metric_score = 2

        details['metrics']['sentence_uniformity'] = {
            'avg_length': round(avg_len, 1),
            'variance_ratio': round(variance_ratio, 3),
            'score': metric_score,
            'details': f'句长变异系数 {variance_ratio:.3f}'
        }

        return score + metric_score, details

    def _analyze_transition_overuse(self, text: str, score: int,
                                    details: Dict) -> Tuple[int, Dict]:
        transition_count = self.count_word_occurrences(text, self.AI_TRANSITIONS)

        metric_score = 0
        if transition_count > 8:
            metric_score = 15
        elif transition_count > 5:
            metric_score = 10
        elif transition_count > 2:
            metric_score = 5

        details['metrics']['transition_overuse'] = {
            'count': transition_count,
            'score': metric_score,
            'details': f'过渡词数量 {transition_count}'
        }

        return score + metric_score, details

    def _analyze_abstract_language(self, text: str, score: int,
                                  details: Dict) -> Tuple[int, Dict]:
        phrase_count = self.count_word_occurrences(text, self.ABSTRACT_PHRASES)

        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        density = phrase_count / (chinese_chars / 100) if chinese_chars > 50 else 0

        metric_score = 0
        if phrase_count > 5:
            metric_score = 15
        elif phrase_count > 3:
            metric_score = 10
        elif phrase_count > 1:
            metric_score = 5

        if density > 3:
            metric_score += 5
        elif density > 2:
            metric_score += 3

        metric_score = min(20, metric_score)

        details['metrics']['abstract_language'] = {
            'count': phrase_count,
            'density': round(density, 2),
            'score': metric_score,
            'details': f'抽象短语数量 {phrase_count}，密度{density:.1f}/100字'
        }

        return score + metric_score, details

    def _analyze_sentence_opening_repetition(self, sentences: List[str], score: int,
                                            details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 4:
            return score, details

        openings = []
        for sent in sentences:
            clean_sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
            if len(clean_sent) >= 4:
                openings.append(clean_sent[:4])

        if not openings:
            return score, details

        counter = Counter(openings)
        most_common = counter.most_common(1)
        top_pattern, top_count = most_common[0]

        metric_score = 0
        if top_count >= 3:
            metric_score = 12
        elif top_count >= 2:
            metric_score = 7

        details['metrics']['sentence_opening_repetition'] = {
            'top_pattern': top_pattern,
            'count': top_count,
            'score': metric_score,
            'details': f'最常见开头 "{top_pattern}" 出现 {top_count} 次'
        }

        return score + metric_score, details

    def _analyze_templates(self, text: str, score: int,
                          details: Dict) -> Tuple[int, Dict]:
        template_count = self.detect_repeated_patterns(text, self.TEMPLATE_PATTERNS)

        metric_score = 0
        if template_count >= 4:
            metric_score = 18
        elif template_count >= 3:
            metric_score = 12
        elif template_count >= 2:
            metric_score = 8
        elif template_count >= 1:
            metric_score = 4

        details['metrics']['templates'] = {
            'count': template_count,
            'score': metric_score,
            'details': f'模板模式数量 {template_count}'
        }

        return score + metric_score, details

    def _analyze_word_burstiness(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        chars_segments = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        if len(chars_segments) <= 5:
            return score, details

        words = []
        for seg in chars_segments:
            for i in range(len(seg) - 1):
                words.append(seg[i:i+2])

        if len(words) <= 10:
            return score, details

        counter = Counter(words)
        counts = list(counter.values())

        if len(counts) <= 1:
            return score, details

        avg_count = statistics.mean(counts)
        std_count = statistics.stdev(counts)
        cv = std_count / avg_count if avg_count > 0 else 0

        metric_score = 0
        if len(text) >= 100:
            if cv < 0.6:
                metric_score = 5
            elif cv < 0.8:
                metric_score = 3
        else:
            if cv < 0.5:
                metric_score = 4
            elif cv < 0.7:
                metric_score = 2

        details['metrics']['word_burstiness'] = {
            'cv': round(cv, 3),
            'score': metric_score,
            'details': f'词汇突发性 CV={cv:.3f}'
        }

        return score + metric_score, details

    def _analyze_over_hedging(self, text: str, score: int,
                             details: Dict) -> Tuple[int, Dict]:
        hedge_count = 0
        hedge_items = []
        for hedge in self.HEDGE_WORDS:
            count = text.count(hedge)
            if count > 0:
                hedge_count += count
                hedge_items.append((hedge, count))

        metric_score = 0
        if hedge_count >= 6:
            metric_score = 12
        elif hedge_count >= 4:
            metric_score = 8
        elif hedge_count >= 2:
            metric_score = 4

        details['metrics']['over_hedging'] = {
            'count': hedge_count,
            'items': hedge_items,
            'score': metric_score,
            'details': f'对冲词数量 {hedge_count}'
        }

        return score + metric_score, details

    def _analyze_bigram_ttr(self, text: str, score: int,
                           details: Dict) -> Tuple[int, Dict]:
        chars = re.findall(r'[\u4e00-\u9fa5]', text)
        if len(chars) < 10:
            return score, details

        bigrams = []
        for i in range(len(chars) - 1):
            bigrams.append(chars[i] + chars[i+1])

        if not bigrams:
            return score, details

        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        ttr = unique_bigrams / total_bigrams if total_bigrams > 0 else 1.0

        metric_score = 0
        if ttr < 0.45:
            metric_score = 12
        elif ttr < 0.55:
            metric_score = 8
        elif ttr < 0.65:
            metric_score = 4

        details['metrics']['bigram_ttr'] = {
            'value': round(ttr, 3),
            'unique': unique_bigrams,
            'total': total_bigrams,
            'score': metric_score,
            'details': f'Bigram TTR={ttr:.3f}'
        }

        return score + metric_score, details

    def _analyze_clause_chain_density(self, text: str, sentences: List[str],
                                      score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 2:
            return score, details

        total_commas = text.count('，') + text.count(',')
        total_periods = text.count('。') + text.count('.') + text.count('！') + text.count('？')

        comma_per_sentence = total_commas / len(sentences) if len(sentences) > 0 else 0

        metric_score = 0
        if comma_per_sentence > 5.0:
            metric_score = 6
        elif comma_per_sentence > 4.0:
            metric_score = 3

        details['metrics']['clause_chain_density'] = {
            'value': round(comma_per_sentence, 2),
            'total_commas': total_commas,
            'score': metric_score,
            'details': f'从句链密度 {comma_per_sentence:.1f}逗号/句'
        }

        return score + metric_score, details

    def _analyze_idiom_overuse(self, text: str, score: int,
                              details: Dict) -> Tuple[int, Dict]:
        idiom_count = 0
        for idiom in self.IDIOM_PATTERNS:
            idiom_count += text.count(idiom)

        metric_score = 0
        if idiom_count >= 6:
            metric_score = 12
        elif idiom_count >= 4:
            metric_score = 8
        elif idiom_count >= 2:
            metric_score = 4

        details['metrics']['idiom_overuse'] = {
            'count': idiom_count,
            'score': metric_score,
            'details': f'成语/四字词数量 {idiom_count}'
        }

        return score + metric_score, details

    def _analyze_punctuation_density(self, text: str, score: int,
                                    details: Dict) -> Tuple[int, Dict]:
        comma_count = text.count('，') + text.count(',')
        period_count = text.count('。') + text.count('.')

        ratio = comma_count / period_count if period_count > 0 else 0

        metric_score = 0
        if ratio > 5.0:
            metric_score = 10
        elif ratio > 3.5:
            metric_score = 6
        elif ratio > 2.5:
            metric_score = 3

        details['metrics']['punctuation_density'] = {
            'comma_count': comma_count,
            'period_count': period_count,
            'comma_period_ratio': round(ratio, 2),
            'score': metric_score,
            'details': f'逗号/句号比 {ratio:.1f}'
        }

        return score + metric_score, details

    def _analyze_concluding_formula(self, text: str, score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        formula_count = 0
        for formula in self.CONCLUDING_FORMULAS:
            formula_count += text.count(formula)

        metric_score = 0
        if formula_count >= 3:
            metric_score = 10
        elif formula_count >= 2:
            metric_score = 6
        elif formula_count >= 1:
            metric_score = 3

        details['metrics']['concluding_formula'] = {
            'count': formula_count,
            'score': metric_score,
            'details': f'段末总结套话 {formula_count} 处'
        }

        return score + metric_score, details

    def _analyze_suizhe_template(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        suizhe_patterns = [
            r'随着[^，。]{2,30}的[^，。]{2,30}[,，][^，。]{0,10}(?:发展|进步|提升|增长|兴起)',
            r'基于[^，。]{2,30}的[^，。]{2,30}[,，][^，。]{0,10}(?:提出|设计|构建|实现)',
        ]

        total_count = 0
        for pattern in suizhe_patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 3:
            metric_score = 8
        elif total_count >= 2:
            metric_score = 5
        elif total_count >= 1:
            metric_score = 2

        details['metrics']['suizhe_template'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'"随着/基于...的..."模板 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_paragraph_template(self, paragraphs: List[str], score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        if len(paragraphs) < 2:
            return score, details

        marker_count = 0
        for para in paragraphs:
            for marker in self.PARAGRAPH_TEMPLATE_MARKERS:
                if marker in para:
                    marker_count += 1
                    break

        metric_score = 0
        if marker_count >= 4:
            metric_score = 12
        elif marker_count >= 3:
            metric_score = 8
        elif marker_count >= 2:
            metric_score = 4

        details['metrics']['paragraph_template'] = {
            'marker_count': marker_count,
            'paragraph_count': len(paragraphs),
            'score': metric_score,
            'details': f'段落模板标记 {marker_count}/{len(paragraphs)} 段'
        }

        return score + metric_score, details

    def _analyze_definition_pattern(self, text: str, score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        patterns = [
            r'是指[\u4e00-\u9fa5]{2,40}的[\u4e00-\u9fa5]{2,20}',
            r'被定义为[\u4e00-\u9fa5]{2,30}',
            r'定义为[\u4e00-\u9fa5]{2,30}',
        ]
        seen_spans = set()
        count = 0
        for p in patterns:
            for m in re.finditer(p, text):
                span = (m.start(), m.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    count += 1

        metric_score = 0
        if count >= 4:
            metric_score = 10
        elif count >= 3:
            metric_score = 6
        elif count >= 2:
            metric_score = 3
        elif count >= 1:
            metric_score = 1

        details['metrics']['definition_pattern'] = {
            'count': count,
            'score': metric_score,
            'details': f'定义式表达 {count} 处'
        }

        return score + metric_score, details

    def _analyze_em_dash_overuse(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        em_dash_count = text.count('——')
        single_dash_count = text.count('—') - em_dash_count * 2

        total_dashes = em_dash_count + max(0, single_dash_count // 2)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))

        density = (total_dashes / chinese_chars) * 1000 if chinese_chars > 0 else 0

        metric_score = 0
        if total_dashes >= 4:
            metric_score = 8
        elif total_dashes >= 3:
            metric_score = 4

        if density > 5.0:
            metric_score = max(metric_score, 6)

        details['metrics']['em_dash_overuse'] = {
            'count': total_dashes,
            'density_per_1000': round(density, 2),
            'score': metric_score,
            'details': f'破折号 {total_dashes} 处（密度{density:.1f}/千字）'
        }

        return score + metric_score, details

    def _analyze_citation_distribution(self, text: str, score: int,
                                      details: Dict) -> Tuple[int, Dict]:
        citations = re.findall(r'\[\d+(?:[-,]\d+)*\]', text)
        total_citations = len(citations)

        if total_citations < 2:
            return score, details

        sentences = self.split_sentences(text, chinese=True)
        end_citations = 0
        for sent in sentences:
            stripped = sent.rstrip()
            if re.search(r'\[\d+(?:[-,]\d+)*\]$', stripped):
                end_citations += 1

        end_ratio = end_citations / total_citations if total_citations > 0 else 0

        metric_score = 0
        # 提高阈值，减少对引用在句末的惩罚
        if end_ratio > 0.9 and total_citations >= 4:
            metric_score = 8
        elif end_ratio > 0.8 and total_citations >= 3:
            metric_score = 4

        details['metrics']['citation_distribution'] = {
            'total_citations': total_citations,
            'end_citations': end_citations,
            'end_citation_ratio': round(end_ratio, 2),
            'score': metric_score,
            'details': f'引用{total_citations}处，{end_ratio*100:.0f}%在句末'
        }

        return score + metric_score, details

    def _analyze_absolute_language(self, text: str, score: int,
                                  details: Dict) -> Tuple[int, Dict]:
        patterns = [
            r'直接决定了[^，。]{2,40}',
            r'直接决定[^，。]{2,40}',
            r'决定了[^，。]{2,20}的[^，。]{2,20}',
            r'是[^，。]{2,30}的前提',
            r'是[^，。]{2,30}的基础',
            r'是[^，。]{2,30}的关键',
            r'是[^，。]{2,30}的保证',
            r'是[^，。]{2,30}的核心',
            r'毫无疑问[^，。]{2,40}',
            r'显然[^，。]{2,40}',
            r'必然[^，。]{2,40}',
            r'一定[^，。]{2,40}',
            r'绝对[^，。]{2,40}',
            r'肯定[^，。]{2,40}',
            r'必须[^，。]{2,40}',
            r'只能[^，。]{2,40}',
            r'唯一[^，。]{2,40}',
            r'不可能[^，。]{2,40}',
            r'不可能不[^，。]{2,40}',
            r'总是[^，。]{2,40}',
            r'永远[^，。]{2,40}',
            r'从不[^，。]{2,40}',
            r'完全[^，。]{2,40}',
            r'彻底[^，。]{2,40}',
            r'全部[^，。]{2,40}',
            r'任何[^，。]{2,40}都[^，。]{2,40}',
            r'所有[^，。]{2,40}都[^，。]{2,40}',
            r'每个[^，。]{2,40}都[^，。]{2,40}',
            r'凡是[^，。]{2,40}都[^，。]{2,40}',
            r'无论[^，。]{2,40}都[^，。]{2,40}',
            r'不管[^，。]{2,40}都[^，。]{2,40}',
            r'非[^，。]{2,20}不可',
            r'非[^，。]{2,20}莫属',
            r'非[^，。]{2,20}不能',
            r'只有[^，。]{2,30}才能[^，。]{2,40}',
            r'只要[^，。]{2,30}就[^，。]{2,40}',
            r'一旦[^，。]{2,30}就[^，。]{2,40}',
            r'势必[^，。]{2,40}',
            r'注定[^，。]{2,40}',
            r'必然导致[^，。]{2,40}',
            r'不可避免[^，。]{2,40}',
            r'无可否认[^，。]{2,40}',
            r'不可否认[^，。]{2,40}',
            r'显而易见[^，。]{2,40}',
            r'不言而喻[^，。]{2,40}',
            r'有目共睹[^，。]{2,40}',
            r'众所周知[^，。]{2,40}',
            r'众所周知，[^，。]{2,40}',
            r'众所周知的是[^，。]{2,40}',
        ]

        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 5:
            metric_score = 20
        elif total_count >= 3:
            metric_score = 14
        elif total_count >= 2:
            metric_score = 10
        elif total_count >= 1:
            metric_score = 5

        details['metrics']['absolute_language'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'绝对化语言 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_correlative_conjunctions(self, text: str, score: int,
                                         details: Dict) -> Tuple[int, Dict]:
        total_count = 0
        found = []
        for pattern, name in self.CORRELATIVE_CONJUNCTIONS:
            matches = re.findall(pattern, text)
            if matches:
                total_count += len(matches)
                found.append((name, len(matches)))

        metric_score = 0
        if total_count >= 3:
            metric_score = 12
        elif total_count >= 2:
            metric_score = 8
        elif total_count >= 1:
            metric_score = 4

        details['metrics']['correlative_conjunctions'] = {
            'count': total_count,
            'items': found,
            'score': metric_score,
            'details': f'关联词结构 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_ai_adjectives(self, text: str, score: int,
                              details: Dict) -> Tuple[int, Dict]:
        total_count = 0
        for pattern in self.AI_ADJECTIVE_PAIRS:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 2:
            metric_score = 10
        elif total_count >= 1:
            metric_score = 5

        details['metrics']['ai_adjectives'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'AI修饰词组合 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_purpose_clauses(self, text: str, score: int,
                                details: Dict) -> Tuple[int, Dict]:
        total_count = 0
        for pattern in self.PURPOSE_CLAUSE_PATTERNS:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 6:
            metric_score = 18
        elif total_count >= 4:
            metric_score = 14
        elif total_count >= 3:
            metric_score = 10
        elif total_count >= 2:
            metric_score = 6
        elif total_count >= 1:
            metric_score = 3

        details['metrics']['purpose_clauses'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'目的/结果从句 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_sentence_structure_pattern(self, sentences: List[str],
                                           text: str, score: int,
                                           details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 3:
            return score, details

        structure_patterns = []
        for sent in sentences:
            has_comma = '，' in sent or ',' in sent
            has_citation = bool(re.search(r'\[\d+(?:[-,]\d+)*\]', sent))
            starts_with_transition = any(sent.startswith(t) for t in self.AI_TRANSITIONS)
            has_purpose = bool(re.search(r'以[^，。]{2,}', sent))
            has_definition = bool(re.search(r'是[^，。]{2,20}的[^，。]{2,20}', sent))

            pattern_key = (
                'Y' if has_comma else 'N',
                'Y' if has_citation else 'N',
                'Y' if starts_with_transition else 'N',
                'Y' if has_purpose else 'N',
                'Y' if has_definition else 'N',
            )
            structure_patterns.append(pattern_key)

        if len(structure_patterns) >= 3:
            counter = Counter(structure_patterns)
            most_common_pattern, count = counter.most_common(1)[0]
            pattern_ratio = count / len(structure_patterns)

            # 短文本（<5句）降低敏感度
            sentence_count = len(structure_patterns)
            metric_score = 0
            if sentence_count >= 5:
                if pattern_ratio > 0.75:
                    metric_score = 6
                elif pattern_ratio > 0.6:
                    metric_score = 3
                elif pattern_ratio > 0.5:
                    metric_score = 1
            else:
                if pattern_ratio > 0.85:
                    metric_score = 4
                elif pattern_ratio > 0.75:
                    metric_score = 2

            details['metrics']['sentence_structure_pattern'] = {
                'dominant_ratio': round(pattern_ratio, 2),
                'score': metric_score,
                'details': f'句式结构重复率 {pattern_ratio*100:.0f}%'
            }

            return score + metric_score, details

        return score, details

    def _analyze_role_playing_pattern(self, text: str, score: int,
                                     details: Dict) -> Tuple[int, Dict]:
        patterns = [
            r'扮演[^，。]{0,10}的角色',
            r'充当[^，。]{0,10}的角色',
            r'作为[^，。]{0,10}的角色',
            r'扮演[^，。]{0,10}角色',
            r'充当[^，。]{0,10}角色',
            r'起着[^，。]{0,10}作用',
            r'起到[^，。]{0,10}作用',
            r'发挥着[^，。]{0,10}作用',
        ]

        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 2:
            metric_score = 10
        elif total_count >= 1:
            metric_score = 5

        details['metrics']['role_playing_pattern'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'"扮演/充当/作为...角色" {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_premise_conclusion_pattern(self, text: str, score: int,
                                           details: Dict) -> Tuple[int, Dict]:
        patterns = [
            r'因此[，,][^，。]{2,30}是[^，。]{2,30}的前提',
            r'因此[，,][^，。]{2,30}是[^，。]{2,30}的基础',
            r'因此[，,][^，。]{2,30}是[^，。]{2,30}的关键',
            r'因此[，,][^，。]{2,30}选择[^，。]{2,30}',
            r'因此[，,][^，。]{2,30}采用[^，。]{2,30}',
            r'因此[，,][^，。]{2,30}提出[^，。]{2,30}',
            r'因此[，,][^，。]{2,30}构建[^，。]{2,30}',
            r'因此[，,][^，。]{2,30}设计[^，。]{2,30}',
            r'因此[，,][^，。]{2,30}需要[^，。]{2,30}',
        ]

        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 2:
            metric_score = 12
        elif total_count >= 1:
            metric_score = 6

        details['metrics']['premise_conclusion_pattern'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'"因此...是...前提/基础/关键" {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_vip_semantic_fingerprints(self, text: str, score: int,
                                          details: Dict) -> Tuple[int, Dict]:
        fingerprint_count = 0
        matched_patterns = []

        for pattern in self.VIP_SEMANTIC_FINGERPRINTS:
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

        details['metrics']['vip_semantic_fingerprint'] = {
            'count': fingerprint_count,
            'patterns': matched_patterns[:3],
            'score': metric_score,
            'details': f'维普语义指纹 {fingerprint_count} 处'
        }

        return score + metric_score, details

    def _analyze_vip_mechanical_patterns(self, text: str, score: int,
                                        details: Dict) -> Tuple[int, Dict]:
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

        details['metrics']['vip_mechanical_pattern'] = {
            'count': mechanical_count,
            'score': metric_score,
            'details': f'维普机械论证模式 {mechanical_count} 处'
        }

        return score + metric_score, details

    def _analyze_vip_data_authenticity(self, text: str, score: int,
                                      details: Dict) -> Tuple[int, Dict]:
        suspicious_count = 0
        for pattern in self.VIP_DATA_PATTERNS:
            matches = re.findall(pattern, text)
            suspicious_count += len(matches)

        metric_score = 0
        if suspicious_count >= 5:
            metric_score = 8
        elif suspicious_count >= 3:
            metric_score = 4

        details['metrics']['vip_data_authenticity'] = {
            'count': suspicious_count,
            'score': metric_score,
            'details': f'维普数据真实性 {suspicious_count} 处可疑数据'
        }

        return score + metric_score, details

    def _analyze_technical_patterns(self, text: str, score: int,
                                    details: Dict) -> Tuple[int, Dict]:
        total_count = 0
        matched_patterns = []

        for pattern in self.TECHNICAL_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                total_count += len(matches)
                matched_patterns.append(pattern)

        metric_score = 0
        if total_count >= 8:
            metric_score = 12
        elif total_count >= 6:
            metric_score = 8
        elif total_count >= 4:
            metric_score = 5
        elif total_count >= 2:
            metric_score = 3
        elif total_count >= 1:
            metric_score = 1

        details['metrics']['technical_patterns'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'技术文本模式 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_chain_description(self, text: str, score: int,
                                   details: Dict) -> Tuple[int, Dict]:
        total_count = 0
        for pattern in self.CHAIN_DESCRIPTION_PATTERNS:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 3:
            metric_score = 20
        elif total_count >= 2:
            metric_score = 15
        elif total_count >= 1:
            metric_score = 8

        details['metrics']['chain_description'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'链式描述模式 {total_count} 处'
        }

        return score + metric_score, details

    def _analyze_passive_voice_density(self, text: str, sentences: List[str],
                                       score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 2:
            return score, details

        passive_patterns = [
            r'被[\u4e00-\u9fa5]{1,15}[所的]',
            r'为[\u4e00-\u9fa5]{1,15}[所的]',
            r'予以[\u4e00-\u9fa5]{1,10}',
            r'得以[\u4e00-\u9fa5]{1,10}',
        ]

        passive_count = 0
        for pattern in passive_patterns:
            matches = re.findall(pattern, text)
            passive_count += len(matches)

        passive_ratio = passive_count / len(sentences) if len(sentences) > 0 else 0

        metric_score = 0
        if passive_ratio > 0.5:
            metric_score = 4
        elif passive_ratio > 0.35:
            metric_score = 2

        details['metrics']['passive_voice_density'] = {
            'count': passive_count,
            'ratio': round(passive_ratio, 2),
            'score': metric_score,
            'details': f'被动语态密度 {passive_count}处/{len(sentences)}句'
        }

        return score + metric_score, details

    def _analyze_sentence_complexity(self, text: str, sentences: List[str],
                                     score: int, details: Dict) -> Tuple[int, Dict]:
        if len(sentences) < 2:
            return score, details

        complex_count = 0
        for sent in sentences:
            if len(sent) > 80:
                comma_count = sent.count('，') + sent.count(',')
                if comma_count >= 4:
                    complex_count += 1

        complex_ratio = complex_count / len(sentences) if len(sentences) > 0 else 0

        metric_score = 0
        if complex_ratio > 0.6:
            metric_score = 8
        elif complex_ratio > 0.4:
            metric_score = 5
        elif complex_ratio > 0.25:
            metric_score = 3

        details['metrics']['sentence_complexity'] = {
            'complex_count': complex_count,
            'ratio': round(complex_ratio, 2),
            'score': metric_score,
            'details': f'复杂长句比例 {complex_ratio*100:.0f}%'
        }

        return score + metric_score, details

    def _analyze_verbose_expressions(self, text: str, score: int,
                                     details: Dict) -> Tuple[int, Dict]:
        verbose_patterns = [
            r'在[^，。]{2,20}过程中[^，。]{0,20}',
            r'在[^，。]{2,20}方面[^，。]{0,20}',
            r'在[^，。]{2,20}领域[^，。]{0,20}',
            r'在[^，。]{2,20}背景下[^，。]{0,20}',
            r'在[^，。]{2,20}基础上[^，。]{0,20}',
            r'在[^，。]{2,20}前提下[^，。]{0,20}',
            r'在[^，。]{2,20}条件下[^，。]{0,20}',
            r'在[^，。]{2,20}情况下[^，。]{0,20}',
            r'在[^，。]{2,20}环境中[^，。]{0,20}',
            r'在[^，。]{2,20}框架下[^，。]{0,20}',
            r'从[^，。]{2,20}角度[^，。]{0,20}来看',
            r'从[^，。]{2,20}层面[^，。]{0,20}来看',
            r'从[^，。]{2,20}维度[^，。]{0,20}来看',
            r'就[^，。]{2,20}而言',
            r'就[^，。]{2,20}来说',
            r'就[^，。]{2,20}来讲',
            r'对于[^，。]{2,20}来说',
            r'对于[^，。]{2,20}而言',
            r'关于[^，。]{2,20}的问题',
            r'关于[^，。]{2,20}的研究',
            r'关于[^，。]{2,20}的分析',
            r'针对[^，。]{2,20}的问题',
            r'针对[^，。]{2,20}的研究',
            r'针对[^，。]{2,20}的分析',
            r'通过[^，。]{2,30}的方式',
            r'通过[^，。]{2,30}的方法',
            r'通过[^，。]{2,30}的手段',
            r'通过[^，。]{2,30}的途径',
            r'利用[^，。]{2,30}的方式',
            r'利用[^，。]{2,30}的方法',
            r'利用[^，。]{2,30}的手段',
            r'采用[^，。]{2,30}的方式',
            r'采用[^，。]{2,30}的方法',
            r'采用[^，。]{2,30}的手段',
            r'使用[^，。]{2,30}的方式',
            r'使用[^，。]{2,30}的方法',
            r'使用[^，。]{2,30}的手段',
            r'进行[^，。]{2,30}的工作',
            r'进行[^，。]{2,30}的研究',
            r'进行[^，。]{2,30}的分析',
            r'开展[^，。]{2,30}的工作',
            r'开展[^，。]{2,30}的研究',
            r'开展[^，。]{2,30}的分析',
            r'实现[^，。]{2,30}的目标',
            r'实现[^，。]{2,30}的目的',
            r'实现[^，。]{2,30}的任务',
            r'完成[^，。]{2,30}的目标',
            r'完成[^，。]{2,30}的任务',
            r'达到[^，。]{2,30}的目标',
            r'达到[^，。]{2,30}的目的',
            r'需要[^，。]{2,30}的[^，。]{2,30}',
            r'应该[^，。]{2,30}的[^，。]{2,30}',
            r'必须[^，。]{2,30}的[^，。]{2,30}',
            r'能够[^，。]{2,30}的[^，。]{2,30}',
            r'可以[^，。]{2,30}的[^，。]{2,30}',
            r'对[^，。]{2,30}进行[^，。]{2,30}',
            r'将[^，。]{2,30}进行[^，。]{2,30}',
            r'把[^，。]{2,30}进行[^，。]{2,30}',
            r'为[^，。]{2,30}提供[^，。]{2,30}',
            r'为[^，。]{2,30}带来[^，。]{2,30}',
            r'给[^，。]{2,30}提供[^，。]{2,30}',
            r'给[^，。]{2,30}带来[^，。]{2,30}',
            r'使得[^，。]{2,30}能够[^，。]{2,30}',
            r'使得[^，。]{2,30}可以[^，。]{2,30}',
            r'让[^，。]{2,30}能够[^，。]{2,30}',
            r'让[^，。]{2,30}可以[^，。]{2,30}',
            r'有助于[^，。]{2,30}的[^，。]{2,30}',
            r'有利于[^，。]{2,30}的[^，。]{2,30}',
            r'有益于[^，。]{2,30}的[^，。]{2,30}',
            r'在[^，。]{2,30}中[^，。]{2,30}发挥着[^，。]{2,30}',
            r'在[^，。]{2,30}中[^，。]{2,30}起到了[^，。]{2,30}',
            r'在[^，。]{2,30}中[^，。]{2,30}扮演着[^，。]{2,30}',
        ]

        total_count = 0
        for pattern in verbose_patterns:
            matches = re.findall(pattern, text)
            total_count += len(matches)

        metric_score = 0
        if total_count >= 8:
            metric_score = 20
        elif total_count >= 5:
            metric_score = 15
        elif total_count >= 3:
            metric_score = 10
        elif total_count >= 2:
            metric_score = 6
        elif total_count >= 1:
            metric_score = 3

        details['metrics']['verbose_expressions'] = {
            'count': total_count,
            'score': metric_score,
            'details': f'冗长表达 {total_count} 处'
        }

        return score + metric_score, details