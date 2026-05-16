import streamlit as st
import openai
import re
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from detection_pipeline import DetectionPipeline
    from formatter import strip_latex, format_clean_chinese, format_formulas
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False

st.set_page_config(
    page_title="Humanize Academic Paper",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


def calculate_ai_rate(text, lang):
    """
    使用统一检测管道计算AI分数

    Args:
        text: 待检测文本
        lang: 语言 ('English' or 'Chinese')

    Returns:
        Tuple of (score, details)
    """
    if not text or not text.strip():
        return 0, None

    text_len = len(text)
    if text_len < 10:
        return 0, None

    clean_text = strip_latex(text)
    clean_len = len(clean_text.strip())
    if clean_len < 10:
        st.warning("文本过短或主要为LaTeX公式，AI检测结果仅供参考")
        return 0, None

    latex_ratio = 1.0 - (clean_len / max(text_len, 1))
    if latex_ratio > 0.7:
        st.info(f"检测到文本中约 {latex_ratio*100:.0f}% 为LaTeX公式，已自动剥离后分析")

    if not HAS_PIPELINE:
        st.warning("AI检测模块未加载，使用基础启发式回退检测")
        return _fallback_ai_detection(clean_text, lang), None

    target_lang = 'en' if lang == "English" else 'zh'

    try:
        pipeline = DetectionPipeline(lang=target_lang)
        score, details = pipeline.detect(clean_text, lang=target_lang)
        return score, details
    except Exception as e:
        st.error(f"检测出错: {str(e)}")
        return _fallback_ai_detection(clean_text, lang), None


def _fallback_ai_detection(text, lang):
    """基础回退检测：当高级检测器不可用时的简单启发式"""
    score = 0
    if lang == "English":
        words = text.split()
        if len(words) < 5:
            return 0
        ai_transitions = ['moreover', 'furthermore', 'additionally', 'in addition',
                          'it is important to note', 'it should be noted', 'it is worth noting']
        for t in ai_transitions:
            score += text.lower().count(t) * 8
        abstract_phrases = ['various aspects', 'multiple factors', 'in terms of',
                            'plays an important role', 'plays a crucial role']
        for p in abstract_phrases:
            score += text.lower().count(p) * 6
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
                cv = (variance ** 0.5) / avg_len if avg_len > 0 else 0
                if cv < 0.25:
                    score += 20
                elif cv < 0.35:
                    score += 10
    else:
        if len(text) < 5:
            return 0
        ai_patterns = ["随着", "基于", "总体来看", "综上所述", "此外", "值得注意的是",
                       "具有重要意义", "发挥着重要作用", "首先", "其次", "本文旨在"]
        for p in ai_patterns:
            score += text.count(p) * 8
        sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
        if len(sentences) >= 3:
            lengths = [len(s) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            cv = (variance ** 0.5) / avg_len if avg_len > 0 else 0
            if cv < 0.25:
                score += 20
            elif cv < 0.40:
                score += 10
    return score


# ============================================================
# 每项检测指标的独立阈值配置
# ============================================================

CHINESE_METRIC_THRESHOLDS = {
    'sentence_length_distribution': {'max_score': 8, 'description': '句长分布'},
    'paragraph_structure_similarity': {'max_score': 6, 'description': '段落结构相似度'},
    'info_density_distribution': {'max_score': 6, 'description': '信息密度分布'},
    'transition_word_distribution': {'max_score': 8, 'description': '连接词分布'},
    'sentence_uniformity': {'max_score': 8, 'description': '句式统一性'},
    'transition_overuse': {'max_score': 8, 'max_count': 5, 'description': '过渡词滥用'},
    'abstract_language': {'max_score': 6, 'max_count': 3, 'description': '抽象语言'},
    'sentence_opening_repetition': {'max_score': 6, 'description': '句首重复'},
    'templates': {'max_score': 6, 'max_count': 1, 'description': '模板句式'},
    'word_burstiness': {'max_score': 6, 'description': '词汇突发性'},
    'over_hedging': {'max_score': 6, 'max_count': 2, 'description': '过度对冲'},
    'bigram_ttr': {'max_score': 6, 'description': '二元组多样性'},
    'clause_chain_density': {'max_score': 6, 'description': '从句链密度'},
    'idiom_overuse': {'max_score': 6, 'max_count': 3, 'description': '成语堆砌'},
    'punctuation_density': {'max_score': 5, 'description': '标点密度'},
    'concluding_formula': {'max_score': 5, 'max_count': 1, 'description': '总结套话'},
    'suizhe_template': {'max_score': 6, 'max_count': 1, 'description': '"随着/基于"模板'},
    'paragraph_template': {'max_score': 6, 'description': '段落模板化'},
    'definition_pattern': {'max_score': 6, 'max_count': 1, 'description': '"是...的"定义式'},
    'em_dash_overuse': {'max_score': 4, 'max_count': 1, 'description': '破折号滥用'},
    'citation_distribution': {'max_score': 8, 'description': '引用分布'},
    'absolute_language': {'max_score': 10, 'max_count': 2, 'description': '绝对化语言'},
    'verbose_expressions': {'max_score': 10, 'max_count': 3, 'description': '冗长表达'},
    'ai_completion_pattern': {'max_score': 8, 'max_count': 3, 'description': 'AI完成句式'},
    'ai_formal_connectives': {'max_score': 6, 'max_count': 2, 'description': 'AI正式连接词'},
    'ai_abstract_suffix': {'max_score': 8, 'max_count': 4, 'description': 'AI抽象后缀'},
    'ai_causal_chain': {'max_score': 5, 'max_count': 2, 'description': 'AI因果链句式'},
}

ENGLISH_METRIC_THRESHOLDS = {
    'sentence_uniformity': {'max_score': 8, 'description': 'Sentence uniformity'},
    'transition_overuse': {'max_score': 10, 'description': 'Transition overuse'},
    'abstract_language': {'max_score': 8, 'description': 'Abstract language'},
    'vocabulary_diversity': {'max_score': 8, 'description': 'Vocabulary diversity'},
    'passive_voice': {'max_score': 8, 'description': 'Passive voice'},
    'paragraph_patterns': {'max_score': 6, 'description': 'Paragraph patterns'},
    'burstiness': {'max_score': 4, 'description': 'Burstiness'},
    'bigram_ttr': {'max_score': 6, 'description': 'Bigram TTR'},
    'clause_chain': {'max_score': 5, 'description': 'Clause chain density'},
    'sentence_openings': {'max_score': 6, 'description': 'Sentence openings'},
    'punctuation': {'max_score': 5, 'description': 'Punctuation pattern'},
    'concluding_formula': {'max_score': 6, 'description': 'Concluding formula'},
    'hedging': {'max_score': 8, 'description': 'Hedging language'},
    'word_repetition': {'max_score': 6, 'description': 'Word repetition'},
}


def check_metric_pass(metric_data: dict, threshold: dict) -> bool:
    """检查单个检测指标是否通过阈值"""
    score = metric_data.get('score', 0)
    if score > threshold['max_score']:
        return False
    if 'max_count' in threshold:
        count = metric_data.get('count', 0)
        if count > threshold['max_count']:
            return False
    return True


def get_failing_metrics(ai_details: dict, thresholds: dict) -> list:
    """获取所有未通过阈值的检测指标列表"""
    failing = []
    if not ai_details or 'metrics' not in ai_details:
        return failing
    metrics = ai_details['metrics']
    for metric_key, threshold_info in thresholds.items():
        metric_data = metrics.get(metric_key, {})
        if not check_metric_pass(metric_data, threshold_info):
            score = metric_data.get('score', 0)
            details_str = metric_data.get('details', '')
            failing.append({
                'key': metric_key,
                'description': threshold_info['description'],
                'current_score': score,
                'max_score': threshold_info['max_score'],
                'details': details_str,
                'data': metric_data,
            })
    return failing


def get_passing_metrics(ai_details: dict, thresholds: dict) -> list:
    """获取所有已通过阈值的检测指标列表"""
    passing = []
    if not ai_details or 'metrics' not in ai_details:
        return passing
    metrics = ai_details['metrics']
    for metric_key, threshold_info in thresholds.items():
        metric_data = metrics.get(metric_key, {})
        if check_metric_pass(metric_data, threshold_info):
            passing.append({
                'key': metric_key,
                'description': threshold_info['description'],
                'data': metric_data,
            })
    return passing


def calculate_quality_metrics(text: str, lang: str) -> dict:
    """
    计算文本质量指标，用于跟踪质量变化
    
    Returns:
        dict with quality metrics
    """
    if lang == "English":
        words = re.findall(r'\b[a-z]+\b', text.lower())
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        word_count = len(words)
        sent_count = len(sentences) if sentences else 1
        unique_words = len(set(words))
        avg_sent_len = word_count / sent_count if sent_count > 0 else 0
        ttr = unique_words / word_count if word_count > 0 else 0
        return {
            'word_count': word_count,
            'sentence_count': sent_count,
            'avg_sentence_length': round(avg_sent_len, 1),
            'ttr': round(ttr, 3),
            'unique_words': unique_words,
        }
    else:
        chars = re.findall(r'[\u4e00-\u9fa5]', text)
        total_chars = len(chars)
        unique_chars = len(set(chars))
        sentences = [s.strip() for s in re.split(r'[。！？!?]+', text) if s.strip()]
        sent_count = len(sentences) if sentences else 1
        avg_sent_len = total_chars / sent_count if sent_count > 0 else 0
        char_diversity = unique_chars / total_chars if total_chars > 0 else 0
        total_len = len(text)
        return {
            'char_count': total_chars,
            'sentence_count': sent_count,
            'avg_sentence_length': round(avg_sent_len, 1),
            'char_diversity': round(char_diversity, 4),
            'unique_chars': unique_chars,
            'total_length': total_len,
        }


def check_quality_degradation(original: dict, current: dict) -> str:
    """
    检查文本质量是否显著下降
    
    Returns:
        warning message if degraded, empty string otherwise
    """
    warnings = []

    if 'total_length' in original and 'total_length' in current:
        length_ratio = current['total_length'] / original['total_length'] if original['total_length'] > 0 else 1.0
        if length_ratio < 0.65:
            warnings.append(f"文本长度大幅缩减（原{original['total_length']}字→现{current['total_length']}字），需保留完整信息量")
        elif length_ratio < 0.8:
            warnings.append(f"文本长度有所缩减（原{original['total_length']}字→现{current['total_length']}字），注意不要遗漏实质性内容")

    if 'char_diversity' in original and 'char_diversity' in current:
        diversity_drop = original['char_diversity'] - current['char_diversity']
        if diversity_drop > 0.02:
            warnings.append(f"用字多样性下降（{original['char_diversity']:.4f}→{current['char_diversity']:.4f}），避免过度简化词汇")

    if 'avg_sentence_length' in original and 'avg_sentence_length' in current:
        orig_avg = original['avg_sentence_length']
        curr_avg = current['avg_sentence_length']
        if curr_avg < orig_avg * 0.6:
            warnings.append(f"平均句长从{orig_avg}字降至{curr_avg}字，句式过于碎片化，需保持适当的长句")

    if 'ttr' in original and 'ttr' in current:
        ttr_drop = original['ttr'] - current['ttr']
        if ttr_drop > 0.08:
            warnings.append(f"词汇多样性下降（TTR: {original['ttr']:.3f}→{current['ttr']:.3f}），注意丰富用词")

    return "；".join(warnings)


def get_quality_preservation_prompt(quality_warning: str, is_en: bool) -> str:
    """根据具体的质量下降问题生成针对性的提示"""
    if is_en:
        base_prompt = (
            "[Quality Preservation Alert]: The rewriting has degraded text quality. "
            "Please address the following specific issues while maintaining the AI pattern fixes:\n\n"
        )
        
        specific_prompts = []
        if "length" in quality_warning.lower() or "缩" in quality_warning:
            specific_prompts.append(
                "• Text Length Issue: The rewritten text is significantly shorter than the original. "
                "This suggests content deletion. Solution: Restore the deleted substantive content. "
                "Every technical detail, argument, and conclusion from the original must be preserved."
            )
        if "diversity" in quality_warning.lower() or "多样性" in quality_warning:
            specific_prompts.append(
                "• Vocabulary Diversity Issue: The rewritten text uses a more limited vocabulary. "
                "Solution: Use more varied vocabulary. Replace repeated words with synonyms. "
                "Introduce domain-specific terminology that was present in the original."
            )
        if "sentence" in quality_warning.lower() or "句" in quality_warning:
            specific_prompts.append(
                "• Sentence Length Issue: The rewritten sentences are too short and fragmented. "
                "Solution: Maintain a mix of sentence lengths. Keep some longer, complex sentences "
                "(30-50 words) that contain detailed technical information. Don't break everything into short, simple sentences."
            )
        
        if not specific_prompts:
            specific_prompts.append(
                "• General Quality Issue: Restore the original information content while maintaining the AI pattern fixes. "
                "Ensure the text length is comparable to the original and vocabulary diversity is preserved."
            )
        
        return base_prompt + "\n".join(specific_prompts)
    
    base_prompt = (
        "【质量保持警告】: 上一轮改写导致文本质量下降。请在修复AI痕迹的同时，针对性地解决以下问题：\n\n"
    )
    
    specific_prompts = []
    if "长度" in quality_warning or "缩" in quality_warning:
        specific_prompts.append(
            "1. 【文本长度问题】: 改写后文本长度显著缩短，说明有内容被删除。\n"
            "   解决方案：恢复被删减的实质性内容。原文中的每个技术细节、论点、结论都必须保留，不能遗漏。"
        )
    if "多样性" in quality_warning:
        specific_prompts.append(
            "2. 【词汇多样性问题】: 改写后使用的词汇种类减少，表达变得单调。\n"
            "   解决方案：丰富用词，使用更多样化的词汇表达。用同义词替换重复的词语，引入原文中出现的专业术语。"
        )
    if "句长" in quality_warning or "碎片化" in quality_warning:
        specific_prompts.append(
            "3. 【句长问题】: 改写后句子过短、过于碎片化，失去了原文的复杂性和深度。\n"
            "   解决方案：保持句长变化，保留一些包含详细技术信息的长句（30-50字）。不要把所有句子都切成短句，这样会丢失学术文本的严谨性。"
        )
    
    if not specific_prompts:
        specific_prompts.append(
            "1. 恢复被删减的实质性内容（论点、数据、方法、结论）\n"
            "2. 保持原文长度，不得大幅缩写\n"
            "3. 保持用词多样性，不要过度简化\n"
            "4. 保持适当的句长变化，不要全部切成短句\n"
            "5. 确保学术信息的完整性和准确性"
        )
    
    return base_prompt + "\n".join(specific_prompts)


# ============================================================
# 中文检测指标 -> 可解释反馈 映射表
# ============================================================

CHINESE_METRIC_FEEDBACK = {
    'sentence_length_distribution': (
        "【句长分布过于集中】检测到大量句子长度集中在15-25字区间，且变异系数偏低。"
"\n"
        "修复方案：刻意制造句长落差。插入若干极短句（5-10字，如'这一结果令人意外。'"
        "'原因何在？'），同时保留或合并若干长句（40-60字）。让最短句与最长句之间"
        "形成明显落差。"
    ),
    'paragraph_structure_similarity': (
        "【段落结构高度相似】各段落在句子数量、平均长度、过渡词使用模式上过于一致。"
"\n"
        "修复方案：打乱段落节奏。有的段落用3句短促有力的句子，有的段落用8句层层递进。"
        "不要每个段落都用'背景→分析→结论'三板斧。"
    ),
    'info_density_distribution': (
        "【信息密度过于均匀】每句话的信息密度（汉字占比）稳定在65%-75%区间，"
"\n"
        "修复方案：制造信息密度起伏。有些句子密集堆叠术语和数字（信息密度>80%），"
        "有些句子穿插口语化过渡或评论（信息密度<60%）。"
    ),
    'transition_word_distribution': (
        "【连接词分布异常】连接词使用密度过高或分布过于均匀。"
"\n"
        "修复方案：大幅删减显式连接词。用内容本身的逻辑顺序替代'因此'、'然而'、"
        "'此外'。删掉一半以上的连接词，让上下文通过语义自然衔接。"
    ),
    'sentence_uniformity': (
        "【句式过于统一】句子长度的变异系数过低，所有句子长度相近，"
        "读起来节奏单一。\n"
        "修复方案：刻意打破均匀节奏。在连续几个中等长度句子之后，"
        "插入一个短句（如'这很关键。'）或一个复杂长句（含多个分句和插入语）。"
    ),
    'transition_overuse': (
        "【过渡词过度使用】检测到过多机械逻辑连接词"
        "（如'首先、其次、最后、综上所述'等）。\n"
        "修复方案：删除大部分显式过渡词。用段落间的自然逻辑推进代替编号式结构。"
        "如果需要强调逻辑关系，使用更自然的表达方式。"
    ),
    'abstract_language': (
        "【空泛套话过多】检测到大量抽象空泛的学术套话"
        "（如'具有重要意义'、'发挥着重要作用'、'不可或缺'等）。\n"
        "修复方案：将每个空泛表述替换为具体描述或更简洁的说法。"
        "'具有重要意义'→'重要'；'发挥着重要作用'→'很重要'；"
        "'不可或缺'→'必要'；'取得了良好的效果'→'取得了效果'；"
        "'在一定程度上'→'部分'；'从...角度来看'→'从...看'；"
        "'就...而言'→'对...'；'通过...的方式'→'通过...'；"
        "'在...过程中'→'...时'；'为...提供了'→'给...提供了'；"
        "'对...进行了'→'对...做了'。"
    ),
    'sentence_opening_repetition': (
        "【句首模式重复】多个句子的开头几字相同，"
"\n"
        "修复方案：改写句首，避免连续两句用相同词开头。"
        "可以使用主语提前、状语前置、疑问句、倒装等不同句式变换开头。"
    ),
    'templates': (
        "【模板句式过多】检测到AI高频模板句式，"
        "如'随着...的...'、'基于...的...'等。\n"
        "修复方案：将这些模板拆解为自然表达。"
        "'随着X的发展'→'X发展之后'或'在X发展的背景下'。"
        "'基于X的Y'→'使用X进行Y'或'通过X实现Y'。"
        "'对X进行了Y'→'对X做了Y'。"
        "'为X提供了Y'→'给X提供了Y'。"
        "'在X过程中'→'X时'。"
        "'通过X的方式'→'通过X'。"
    ),
    'word_burstiness': (
        "【词汇突发性过低】各词汇的出现频率过于均匀，"
"\n"
        "修复方案：增加词汇使用的不均匀性。在相关段落中重复使用核心术语，"
        "同时引入一些低频但精准的专业词汇。"
    ),
    'over_hedging': (
        "【过度使用对冲词】检测到大量不确定性修饰语"
        "（如'似乎'、'可能表明'、'在一定程度上'等）。\n"
        "修复方案：将有数据支持的陈述改为确定性表述。"
        "有数据支撑时直接说'结果表明'而非'结果似乎表明'。"
        "仅在真正不确定的地方保留对冲，删减率应超过70%。"
    ),
    'bigram_ttr': (
        "【二元组重复率过高】相邻汉字组合（bigram）的类型不够丰富，"
        "\n"
        "修复方案：引入更多样化的词汇搭配。"
        "替换高频出现的固定搭配，使用近义词、同义表达变换说法。"
    ),
    'clause_chain_density': (
        "【从句链过密】单个句子包含过多逗号分隔的分句，"
        "\n"
        "修复方案：找到语义断点，把逗号换成句号，拆成2-3个独立短句。"
        "不要机械地在每个逗号处断开，而是找到语义完整的位置。"
        "例：'A，B，C，D'→'A。B，C。D。'（在语义断点处断开）"
    ),
    'idiom_overuse': (
        "【成语/四字词堆砌】检测到过多成语或四字格词组"
        "（如'不可或缺'、'显而易见'、'日新月异'等）。\n"
        "修复方案：将多余的成语替换为平实具体的表述。"
        "保留少量确实精准的成语，删减率应超过50%。"
    ),
    'punctuation_density': (
        "【逗号/句号比过高】逗号数量远超句号，"
        "\n"
        "修复方案：增加句号使用频率。将长句中语义完整的部分独立成句。"
        "目标是将逗号/句号比控制在2.5以下。"
    ),
    'concluding_formula': (
        "【段末总结套话】检测到公式化的段落结尾"
        "（如'综上所述'、'总而言之'、'由此可见'等）。"
        "\n"
        "修复方案：删除所有公式化结尾词。让段落自然收束——"
        "最后一句直接陈述结论或引出下文，不要加'总结帽子'。"
    ),
    'suizhe_template': (
        "【知网级AI特征】检测到'随着...的...'或'基于...的...'模板句式。"
        "\n"
        "修复方案：立即替换这些句式。"
        "'随着X的Y'→'X的Y，使得...'或'在X Y的背景下'。"
        "'基于X的Y'→'使用X的Y'或'采用X进行Y'。"
    ),
    'paragraph_template': (
        "【段落结构模板化】段落开头模式过于固定，"
        "呈现出'背景→问题→意义→方案'的机械四段式结构。\n"
        "修复方案：打乱段落结构。有的段落直接抛出问题，"
        "有的段落先给结论再解释原因。避免每个段落都按照统一模板展开。"
    ),
    'definition_pattern': (
        "【知网级AI特征】检测到'是...的...'定义式句式"
        "\n"
        "修复方案：将定义式改写为主动句或描述句。"
        "'X是Y的重要基础'→'X为Y奠定了基础'或'Y依赖于X'。"
        "避免连续使用'是...的'结构。"
    ),
    'em_dash_overuse': (
        "【破折号滥用】检测到文本中使用了破折号（——）来进行解释说明。\n"
        "修复方案：将所有破折号替换为逗号、冒号或拆分为独立句子。"
        "例如'X实现了Y——Z'应改为'X实现了Y，即Z'或'X实现了Y。Z正是这一目标的具体体现。'"
        "注意：中文破折号占用两个汉字宽度（——），在学术文本中极其罕见，必须全部清除。"
    ),
    'citation_distribution': (
        "【引用集中在句末】超过80%的引文出现在句子末尾，"
        "\n"
        "修复方案：将部分引用移到句中或句首。"
        "如'[1]的研究表明...'或'根据[2]的方法，我们...'。"
        "不要所有引用都放在句末括号里。"
    ),
    'absolute_language': (
        "【绝对化语言过多】检测到过多绝对化表述"
        "（如'毫无疑问'、'显然'、'必然'、'一定'、'必须'、'只能'、'唯一'等）。"
        "\n"
        "修复方案：将绝对化表述改为更谨慎、更客观的表述。"
        "'毫无疑问'→'有充分证据表明'；'显然'→'可以看出'；"
        "'必然'→'往往会'；'一定'→'通常'；'必须'→'需要'。"
        "学术写作应该保持适度的谨慎和客观性。"
    ),
    'verbose_expressions': (
        "【冗长表达过多】检测到大量冗长、啰嗦的表达方式"
        "（如'在...过程中'、'从...角度来看'、'就...而言'、'通过...的方式'等）。"
        "\n"
        "修复方案：将冗长表达简化为更直接、更简洁的表述。"
        "'在X过程中'→'X时'；'从X角度来看'→'从X看'；"
        "'就X而言'→'对X'；'通过X的方式'→'通过X'；"
        "'对X进行了Y'→'对X做了Y'；'为X提供了Y'→'给X提供了Y'；"
        "'在很大程度上'→'很大程度上'；'进一步完成了'→'还做了'。"
    ),
    'ai_completion_pattern': (
        "【AI完成句式过多】检测到大量'动词+了'的AI典型完成句式"
        "（如'提升了'、'实现了'、'解决了'、'降低了'等）。\n"
        "修复方案：替换或简化这些句式。"
        "'提升了'→'提高了'或'提升'；'实现了'→'做到了'或'实现'；"
        "'解决了'→'处理了'或'解决'；'降低了'→'减少了'或'降低'。"
        "也可以去掉'了'，改用更简洁的表述。"
    ),
    'ai_formal_connectives': (
        "【AI正式连接词过多】检测到AI常用的正式连接词"
        "（如'即'、'，且'、'不仅...而且'、'使得'、'为X提供Y'、'意味着'、'体现出'、'相较于'等）。\n"
        "修复方案：替换为更自然的表达。"
        "'即'→'也就是'或'即'删除；'，且'→'，并'或拆成两句；"
        "'不仅...而且'→'既...也'或拆成两句；'使得'→'让'或'使'；"
        "'为X提供Y'→'给X提供Y'；'将X送入'→'把X输入'；"
        "'意味着'→'说明'；'体现出'→'体现'；'相较于'→'比'；"
        "'从X来看'→'从X看'。"
    ),
    'ai_abstract_suffix': (
        "【AI抽象后缀过多】检测到大量X性/X化/X率等抽象名词"
        "（如'鲁棒性'、'泛化性'、'优化化'、'准确率'等）。\n"
        "修复方案：替换为更具体的表述。"
        "'鲁棒性'→'稳定性'；'泛化性'→'适应能力'；"
        "'高效性'→'效率高'；'轻量化'→'轻量'或'参数少'；"
        "'准确率'→'精度'或'正确率'。减少抽象后缀的堆砌。"
    ),
    'ai_causal_chain': (
        "【AI因果链句式过多】检测到大量'通过X实现Y'和'X与Y的Z'等AI典型因果链句式。\n"
        "修复方案：拆分或简化因果链。"
        "'通过X实现Y'→'用X做到Y'或拆成两句：'利用X。这样做到Y。'；"
        "'通过X提升Y'→'用X提高Y'；'通过X优化Y'→'用X改进Y'；"
        "'X与Y的Z'→'X和Y的Z'或拆成'X的Z和Y的Z'。"
    ),
}

ENGLISH_METRIC_FEEDBACK = {
    'sentence_uniformity': (
        "[Sentence Uniformity]: Sentence lengths are too uniform. "
        "Fix: Mix very short sentences (5-10 words) with long complex ones (30+ words). "
        "Create a natural rhythm with varied sentence lengths."
    ),
    'transition_overuse': (
        "[Transition Overuse]: Too many mechanical transition words detected "
        "(e.g., 'moreover', 'furthermore', 'additionally', 'in conclusion'). "
        "Fix: Remove most explicit transitions. Use implicit logical flow instead. "
        "Keep only those that are absolutely necessary for clarity."
    ),
    'abstract_language': (
        "[Abstract Language]: Excessive use of abstract placeholder phrases "
        "(e.g., 'various aspects', 'plays an important role', 'in terms of'). "
        "Fix: Replace each abstract phrase with specific, concrete content. "
        "Instead of 'plays an important role', say what it actually does."
    ),
    'vocabulary_diversity': (
        "[Low Vocabulary Diversity]: Type-Token Ratio is too low, "
        "indicating repetitive word usage patterns typical of AI. "
        "Fix: Introduce more varied vocabulary. Use synonyms and restructure "
        "sentences to avoid repeating the same words frequently."
    ),
    'passive_voice': (
        "[Passive Voice Overuse]: Excessive passive voice constructions. "
        "Fix: Convert some passive constructions to active voice. "
        "Use 'We observed that' instead of 'It was observed that'."
    ),
    'paragraph_patterns': (
        "[Paragraph Pattern Repetition]: Multiple paragraphs have similar "
        "opening patterns or structures. "
        "Fix: Vary how each paragraph starts. Some can start with a question, "
        "some with a bold claim, others with a specific example."
    ),
    'burstiness': (
        "[Low Burstiness]: Sentence length variation is insufficient. "
        "Fix: Create more dramatic contrasts between sentence lengths. "
        "Follow a long, detailed sentence with a short, punchy one."
    ),
    'bigram_ttr': (
        "[Low Bigram Diversity]: Adjacent word pairs are too repetitive. "
        "Fix: Vary word collocations and use more diverse phrasing patterns. "
        "Avoid repeatedly using the same adjective-noun combinations."
    ),
    'clause_chain': (
        "[Clause Chain Density]: Too many commas per sentence on average, "
        "indicating overly complex sentence structures. "
        "Fix: Break long sentences into shorter independent clauses. "
        "Aim for average of 2 or fewer commas per sentence."
    ),
    'sentence_openings': (
        "[Repetitive Sentence Openings]: Multiple sentences start with the "
        "same word or phrase pattern. "
        "Fix: Vary sentence openings. Use different grammatical structures: "
        "adverbial phrases, gerunds, questions, or inverted word order."
    ),
    'punctuation': (
        "[Punctuation Pattern]: Comma-to-period ratio is too high, "
        "suggesting run-on sentences and complex clause chains. "
        "Fix: Use more periods to create shorter, clearer sentences. "
        "Break compound sentences into simpler structures."
    ),
    'concluding_formula': (
        "[Concluding Formulas]: Detected formulaic concluding phrases "
        "(e.g., 'in conclusion', 'to summarize', 'taken together'). "
        "Fix: Remove all formulaic conclusions. Let paragraphs end naturally "
        "with a substantive statement rather than a summary marker."
    ),
    'hedging': (
        "[Excessive Hedging]: Too many hedging/qualifying phrases "
        "(e.g., 'may suggest', 'appears to be', 'to some extent'). "
        "Fix: Use more direct language when evidence supports it. "
        "Reserve hedging only for genuinely uncertain claims."
    ),
    'word_repetition': (
        "[Word Repetition]: High frequency of repeated content words. "
        "Fix: Use synonyms and pronominal references to reduce repetition. "
        "Vary terminology to avoid overusing the same key terms."
    ),
}


def generate_simple_feedback(ai_score: int, ai_details: dict, thresholds: dict, is_en: bool) -> str:
    """
    基于总分和得分最高的子指标，生成简洁的修复反馈
    只关注得分最高的3个问题，给出简短可操作的建议
    """
    if not ai_details or 'metrics' not in ai_details:
        if is_en:
            return f"Current AI score: {ai_score}/100 (target < 35). Continue removing AI patterns with minimal edits."
        return f"当前AI总分: {ai_score}/100（目标 < 35）。继续用最小幅度修改去除AI痕迹。"

    metrics = ai_details['metrics']
    scored_metrics = []
    for metric_key, threshold_info in thresholds.items():
        metric_data = metrics.get(metric_key, {})
        score = metric_data.get('score', 0)
        if score > 0:
            scored_metrics.append({
                'key': metric_key,
                'description': threshold_info['description'],
                'score': score,
                'details': metric_data.get('details', ''),
            })
    scored_metrics.sort(key=lambda x: x['score'], reverse=True)
    top_issues = scored_metrics[:3]

    feedback_lib = CHINESE_METRIC_FEEDBACK if not is_en else ENGLISH_METRIC_FEEDBACK

    if is_en:
        parts = [f"Current AI score: {ai_score}/100 (target < 35). Top issues to fix:"]
        for idx, m in enumerate(top_issues, 1):
            short_feedback = feedback_lib.get(m['key'], f"{m['description']} (score: {m['score']})")
            parts.append(f"{idx}. {short_feedback}")
        parts.append("Remember: minimal edits only. Keep 80%+ of the text unchanged.")
    else:
        parts = [f"当前AI总分: {ai_score}/100（目标 < 35）。主要问题："]
        for idx, m in enumerate(top_issues, 1):
            short_feedback = feedback_lib.get(m['key'], f"{m['description']}（得分: {m['score']}）")
            parts.append(f"{idx}. {short_feedback}")
        parts.append("记住：只做最小幅度修改，保持80%以上原文不变。")

    return "\n".join(parts)


def generate_metric_feedback(ai_details: dict, thresholds: dict, is_en: bool) -> str:
    """
    根据每项检测指标的通过情况，生成有针对性的可解释反馈
    
    对每一项未通过的指标：
    1. 说明具体是什么指标
    2. 解释为什么这是AI痕迹
    3. 给出可操作的修复建议
    """
    failing = get_failing_metrics(ai_details, thresholds)
    passing = get_passing_metrics(ai_details, thresholds)

    feedback_lib = CHINESE_METRIC_FEEDBACK if not is_en else ENGLISH_METRIC_FEEDBACK

    if not failing:
        if is_en:
            return ("The text still retains some machine-generated stiffness. "
                    "Please further vary sentence lengths, remove remaining formulaic transitions, "
                    "and ensure vocabulary is diverse and natural.")
        return ("文本仍残留一些机器生成的僵硬感。请进一步优化句长变化、"
                "消除剩余的公式化表达，确保语言自然流畅。")

    parts = []
    if is_en:
        parts.append(
            "The system detected that the following AI indicators still exceed acceptable thresholds. "
            "Each indicator below includes the detected issue and specific guidance for fixing it."
        )
    else:
        parts.append(
            "系统检测到以下AI指标仍未通过阈值。每条指标都附带了具体的修复指引，"
            "请逐一对照修改。注意是「替换」而非「删除」——每处修改都必须保留原有的实质信息。"
        )

    for idx, m in enumerate(failing, 1):
        desc = m['description']
        score = m['current_score']
        max_score = m['max_score']
        detail = m['details']

        specific_feedback = feedback_lib.get(m['key'], f"指标'{desc}'未通过（{score}分，阈值≤{max_score}分）")

        if is_en:
            parts.append(f"\n[{idx}] {specific_feedback}")
        else:
            parts.append(f"\n【问题{idx}】{specific_feedback}")
            parts.append(f"  当前检测值: {detail}")

    if is_en:
        parts.append(
            "\nPlease rewrite the text above, addressing each issue. "
            "Maintain all substantive content (arguments, data, methods, conclusions). "
            "Do not delete information - replace expressions while keeping the meaning."
        )
    else:
        parts.append(
            "\n请基于以上问题逐一自纠并重新输出。注意每一条被修改的表述都必须保留其原有的实质信息——"
            "替换的是表达方式，不是删内容。不得遗漏原文中的任何论点、实验结果或结论。"
        )

    passing_count = len(passing)
    if passing_count > 0:
        passed_names = [p['description'] for p in passing]
        if is_en:
            parts.append(f"\n✅ Passed indicators: {', '.join(passed_names)}. Keep these improvements.")
        else:
            parts.append(f"\n✅ 已通过指标: {'、'.join(passed_names)}。请保持这些指标的优化成果。")

    return "\n".join(parts)


def process_pipeline(text, lang, target_format, tone, api_base, api_key, model_id):
    """
    多轮迭代润色管道

    核心改进（相对旧版）：
    1. 每项检测指标都有独立阈值，所有指标通过才停止迭代
    2. 针对未通过指标生成可解释的、可操作的修复反馈
    3. 追踪文本质量变化，防止质量过度下降
    """
    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    tone_rules_en = {
        "学术书面 (Formal Academic)": "[Tone Directive]: Strictly formal academic writing. Use rigorous written scholarly language. ABSOLUTELY NO colloquialisms or informal phrasing. Maintain maximum professional depth.",
        "学术演讲 (Academic Presentation)": "[Tone Directive]: Academic presentation/conference style. Maintain scholarly rigor but use slightly shorter, speakable sentences. Phrases like 'We found that' or 'This implies' are acceptable.",
        "一般书面 (General Written)": "[Tone Directive]: General written/technical blog style. Remove overly dense academic jargon. Write cleanly and accessibly for a general educated audience.",
        "口语化 (Colloquial)": "[Tone Directive]: Highly conversational and colloquial style. Use casual, everyday language. Feel free to use phrases like 'pretty much', 'looks like', or conversational analogies."
    }

    tone_rules_zh = {
        "学术书面 (Formal Academic)": "【风格指令】：纯正客观的学术书面书写。使用高度严谨的书面语，【绝对禁止】将学术文本口语化。绝不使用\"算是\"、\"看上去不错\"等非正式表达，确保专业深度。但注意：学术严谨不等于堆砌对冲词，不要为了显得\"谨慎\"而反复使用\"似乎\"、\"可能表明\"、\"在一定程度上\"等——这恰恰是AI改写的典型痕迹。",
        "学术演讲 (Academic Presentation)": "【风格指令】：学术汇报/答辩演讲口吻。语言依然专业且保留核心术语，但句式长短更适宜讲述。允许出现\"我们发现\"、\"这说明\"等更具现场感的用语，避免过长的套娃式从句。不要堆砌对冲词。",
        "一般书面 (General Written)": "【风格指令】：标准的书面/科普表达。去除晦涩难懂的学术词语与大词，面向一般受众解答，语句通顺流畅，不过度堆砌名词，偏向技术博客或新闻报道的流畅质感。",
        "口语化 (Colloquial)": "【风格指令】：高度口语化与对话式的交流表达。使用通俗易懂的大白话、非正式用语，可以加入一些日常感情色彩词（如\"算是\"、\"其实\"、\"看上去不错\"），完全打破学术的严肃与刻板。"
    }

    extra_tone_en = tone_rules_en.get(tone, tone_rules_en["学术书面 (Formal Academic)"])
    extra_tone_zh = tone_rules_zh.get(tone, tone_rules_zh["学术书面 (Formal Academic)"])

    prompt_en = f"""You are an academic text editor. Goal: make text concise and direct, removing AI's over-elaboration.

{extra_tone_en}

【Core Operations】:
1. Split long sentences: find semantic breaks, replace commas with periods. One point per sentence.
2. Remove empty adverbs: "significantly" "effectively" "innovatively" "successfully" → delete
3. Replace AI phrasing (pick the most natural, don't substitute mechanically):

| AI phrasing | → | Natural |
|-------------|---|---------|
| in order to / due to the fact that | → | to / because |
| significantly improved / effectively enhanced | → | improved / enhanced |
| plays a crucial role / is of great importance | → | is important / is key |
| is widely used in / has been demonstrated | → | widely used in / demonstrates |
| It is worth noting that / Furthermore, | → | (delete) / Also, |
| via X to achieve Y | → | using X to Y / X achieves Y |

4. Keep LaTeX formulas, technical terms, and citations unchanged.

【Examples】:
- "This method, which was proposed in 2023, significantly improves accuracy, and has been widely used in many applications"
  → "This method (proposed in 2023) improves accuracy. It is widely used in many applications."

- "It is worth noting that this approach effectively addresses the problem"
  → "This approach addresses the problem."

- "plays a crucial role in achieving significant performance improvements"
  → "is important for performance improvement."

Output ONLY the edited text, nothing else.
"""

    prompt_zh = f"""你是学术文本编辑。目标：让文本简洁直接，去除AI"过度展开、一逗到底"的毛病。

{extra_tone_zh}

【核心操作】：
1. 拆长句：找到语义断点把逗号换句号，每句只说一件事
2. 删空洞副词："显著地""有效地""创新性地""成功地"→删
3. 替换AI用词（选最自然的，不要机械替换）：

| AI用词 | → | 人话 |
|--------|---|------|
| 通过X实现Y | → | 用X做到Y / 直接去掉"通过" |
| 融合/优化/采用 | → | 组合/调整/用 |
| 提升/增强/构建 | → | 提高/加强/搭建 |
| 提升了/实现了/解决了 | → | 提高了/做到了/处理了 |
| 证明了/展示了/验证了 | → | 说明/显示/确认 |
| 提升至/提升幅度 | → | 提高到/提高幅度 |
| 鲁棒性/范式/机制 | → | 稳定性/模式/方法 |
| 意味着/体现出/相较于 | → | 说明/体现/比 |
| 使得/，且/不仅...而且 | → | 让/，并/既...也 |
| 具有重要意义/发挥着关键作用 | → | 重要/关键 |
| 被广泛应用于 | → | 广泛用于 |
| 为X提供Y | → | 给X提供Y |

4. 保持LaTeX公式、专业术语、引用不变

【示例】：
- "该方法于2023年提出，显著提升了准确率，已被广泛应用于多个领域"
  → "该方法于2023年提出，提高了准确率。目前已用于多个领域。"

- "通过引入注意力机制实现了检测精度的显著提升"
  → "引入注意力机制，检测精度明显提高。"

- "本研究进一步完成了模型的可视化分析，其结果证明C2HiLo-YOLO在训练过程中收敛速度快"
  → "本研究还做了模型的可视化分析。结果证明C2HiLo-YOLO训练时收敛快。"

- "该模块通过高低频特征解耦与协同优化的方式，同时捕捉边缘纹理等细粒度特征和场景全局上下文"
  → "该模块把高低频特征分开处理再结合。既捕捉边缘纹理等细节，也获取全局上下文。"

直接输出修改后的文本，不要任何解释。
"""

    system_prompt = prompt_en if lang == "English" else prompt_zh
    is_en = (lang == "English")
    MAX_ROUNDS = 8
    thresholds = ENGLISH_METRIC_THRESHOLDS if is_en else CHINESE_METRIC_THRESHOLDS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    revised = text
    ai_score = 100
    ai_details = None
    original_quality = calculate_quality_metrics(text, lang)

    def make_api_call_with_retry(callable_obj, max_retries=3, initial_delay=1.0):
        """带自动重试的 API 调用包装器"""
        retry_count = 0
        delay = initial_delay
        last_exception = None
        while retry_count <= max_retries:
            try:
                return callable_obj(), None
            except (openai.APIConnectionError, openai.RateLimitError, openai.APIError) as e:
                last_exception = e
                retry_count += 1
                if retry_count > max_retries:
                    break
                st.warning(f"API 调用失败 (第 {retry_count}/{max_retries} 次): {str(e)}，{delay:.1f}秒后重试...")
                time.sleep(delay)
                delay *= 2
            except openai.AuthenticationError as e:
                return None, f"API认证失败: API Key 无效或已过期，请检查。详情: {str(e)}"
        return None, f"API调用失败，已重试 {max_retries} 次。末次错误: {str(last_exception)}"

    TARGET_THRESHOLD = 25
    round_results = []  # 记录每轮的分数和文本
    
    try:
        for round_num in range(1, MAX_ROUNDS + 1):
            if round_num == 1:
                st.info(f"Pipeline Round {round_num}/{MAX_ROUNDS}: 初始改写...")
            else:
                st.info(f"Pipeline Round {round_num}/{MAX_ROUNDS}: 当前总分 {ai_score}/100，目标 < {TARGET_THRESHOLD}")

            def make_round_call():
                return client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=0.7 + (round_num - 1) * 0.05,
                    frequency_penalty=0.2 + (round_num - 1) * 0.04,
                    presence_penalty=0.15 + (round_num - 1) * 0.03
                )

            response_obj, error_msg = make_api_call_with_retry(make_round_call)
            if error_msg:
                return error_msg, 100

            revised = response_obj.choices[0].message.content
            messages.append({"role": "assistant", "content": revised})

            ai_score, ai_details = calculate_ai_rate(revised, lang)
            round_results.append({"score": ai_score, "text": revised})  # 记录本轮结果
            st.write(f"🔬 第{round_num}轮评估的AI近似指纹分数: {ai_score}/100")

            if ai_details:
                with st.expander(f"第{round_num}轮AI评估详细指标"):
                    metrics = ai_details.get('metrics', {})
                    for metric_key, threshold_info in thresholds.items():
                        metric_data = metrics.get(metric_key, {})
                        metric_score = metric_data.get('score', 0)
                        desc = threshold_info['description']
                        details_str = metric_data.get('details', '')
                        st.write(f"• {desc}: {metric_score}分 — {details_str}")
                    
                    with st.expander("📄 查看本轮修复后的文本"):
                        st.text_area(f"第{round_num}轮修复文本", revised, height=300, key=f"revised_text_round_{round_num}")

            if HAS_PIPELINE and lang == "English" and round_num == 1:
                try:
                    from text_analyzer import TextAnalyzer
                    analyzer = TextAnalyzer(revised)
                    quality = analyzer.analyze()
                    with st.expander("📊 文本质量分析 (Text Quality Metrics)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("可读性 (Flesch)", quality.get('readability', {}).get('flesch_reading_ease', 'N/A'))
                            st.metric("学术词汇占比", f"{quality.get('academic_vocabulary', {}).get('percentage', 0):.1f}%")
                        with col2:
                            st.metric("词汇多样性 (TTR)", quality.get('vocabulary', {}).get('ttr', 'N/A'))
                            st.metric("平均句长 (词)", quality.get('sentence_stats', {}).get('avg_length', 'N/A'))
                except Exception:
                    pass

            current_quality = calculate_quality_metrics(revised, lang)
            quality_warning = check_quality_degradation(original_quality, current_quality)

            if ai_score < TARGET_THRESHOLD:
                st.success(f"✅ AI总分 {ai_score} < {TARGET_THRESHOLD}，达标！共迭代 {round_num} 轮。")
                break

            if quality_warning:
                st.warning(f"⚠️ 质量下降警告: {quality_warning}")

            if round_num < MAX_ROUNDS:
                feedback_str = generate_simple_feedback(ai_score, ai_details, thresholds, is_en)
                if quality_warning:
                    feedback_str += "\n\n" + get_quality_preservation_prompt(quality_warning, is_en)
                messages.append({"role": "user", "content": feedback_str})
            else:
                # 第8轮仍未达标，选择分数最低的结果
                best_result = min(round_results, key=lambda x: x['score'])
                st.warning(f"⚠️ 已进行 {MAX_ROUNDS} 轮迭代，总分 {ai_score} 仍 ≥ {TARGET_THRESHOLD}。")
                st.info(f"📊 选择分数最低的结果: {best_result['score']}/100")
                revised = best_result['text']
                ai_score = best_result['score']

        if lang == "Chinese":
            revised = format_clean_chinese(revised)

        revised = format_formulas(revised, target_format)

        return revised, ai_score

    except openai.APIConnectionError as e:
        return f"API连接失败: 无法连接到 {api_base}，请检查 Base URL 和网络连接。详情: {str(e)}", 100
    except openai.AuthenticationError as e:
        return f"API认证失败: API Key 无效或已过期，请检查。详情: {str(e)}", 100
    except openai.RateLimitError as e:
        return f"API速率限制: 请求过于频繁，请稍后重试。详情: {str(e)}", 100
    except openai.APIError as e:
        return f"API服务端错误: {str(e)}", 100
    except Exception as e:
        return f"API调用出错: {str(e)}", 100


def check_all_metrics_pass(ai_details: dict, thresholds: dict) -> tuple:
    """
    检查是否所有检测指标都通过了阈值

    Returns:
        (all_pass: bool, failing_metrics: list)
    """
    failing = get_failing_metrics(ai_details, thresholds)
    return len(failing) == 0, failing


st.title("🎓 Humanize Academic Paper Pipeline")
st.markdown("基于多轮API调用和规则过滤的AI论文防查重，自然化润色工具。")

with st.sidebar:
    st.header("⚙️ API Settings")
    api_base = st.text_input("Base URL", value="https://token.sensenova.cn/v1")
    api_key = st.text_input("API Key", type="password")
    model_id = st.text_input("Model ID", value="deepseek-v4-flash")

    st.header("📝 Options")
    lang_opt = st.radio("Language", ["中文 (Chinese)", "English"])
    format_opt = st.radio("Target Format", ["LaTeX", "Word (Separated Formulas)"])
    tone_opt = st.selectbox(
        "输出风格 (Output Tone)",
        ["学术书面 (Formal Academic)", "学术演讲 (Academic Presentation)", "一般书面 (General Written)", "口语化 (Colloquial)"]
    )

st.subheader("Input Text")

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def clear_input():
    st.session_state.input_text = ""

input_text = st.text_area(
    "在此粘贴需要润色的段落 (包含LaTeX公式请保留 $ 或 $$)：", 
    height=200,
    key="input_text"
)

col1, col2 = st.columns([1, 5])
with col1:
    run_button = st.button("🚀 运行 Humanize Pipeline")
with col2:
    clear_button = st.button("🗑️ 清除输入", on_click=clear_input)

if run_button:
    if not api_key:
        st.error("请输入 API Key")
    elif not input_text.strip():
        st.error("请输入需要处理的文本")
    else:
        lang_param = "English" if lang_opt == "English" else "Chinese"

        st.subheader("📊 原始文本分析")
        orig_score, orig_details = calculate_ai_rate(input_text, lang_param)

        if orig_score > 70:
            st.error(f"润色前原始文本AI评分: {orig_score}/100 (强AI痕迹)")
        elif orig_score > 45:
            st.warning(f"润色前原始文本AI评分: {orig_score}/100 (中等AI痕迹)")
        else:
            st.success(f"润色前原始文本AI评分: {orig_score}/100 (低AI痕迹)")

        if orig_details:
            st.expander("查看原始文本的AI特征详细指标").json(orig_details.get('metrics', {}))

        with st.spinner("Pipeline 运行中..."):
            final_text, final_score = process_pipeline(
                input_text, lang_param, format_opt, tone_opt, api_base, api_key, model_id
            )

        st.subheader("✅ 输出结果")
        st.metric(label="AI分数降幅", value=f"{final_score}/100", delta=f"{final_score - orig_score} 分", delta_color="inverse")

        if final_score < 40:
            st.success(f"最终AI味评分过关 ({final_score}/100) — 已接近低AI率水平")
        elif final_score < 65:
            st.warning(f"最终AI味评分尚可 ({final_score}/100)，建议手动微调")
        else:
            st.error(f"最终AI味评分偏高 ({final_score}/100)，建议重新润色")

        st.text_area("复制结果", final_text, height=300)
