import streamlit as st
import openai
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from detection_pipeline import DetectionPipeline
    from formatter import strip_latex, format_clean_chinese, format_formulas
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False


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


def process_pipeline(text, lang, target_format, tone, api_base, api_key, model_id):
    """
    处理文本润色管道

    去掉了领域(discipline)限制，专注于语气(tone)调整
    """
    client = openai.OpenAI(api_key=api_key, base_url=api_base)

    tone_rules_en = {
        "学术书面 (Formal Academic)": "[Tone Directive]: Strictly formal academic writing. Use rigorous written scholarly language. ABSOLUTELY NO colloquialisms or informal phrasing. Maintain maximum professional depth.",
        "学术演讲 (Academic Presentation)": "[Tone Directive]: Academic presentation/conference style. Maintain scholarly rigor but use slightly shorter, speakable sentences. Phrases like 'We found that' or 'This implies' are acceptable.",
        "一般书面 (General Written)": "[Tone Directive]: General written/technical blog style. Remove overly dense academic jargon. Write cleanly and accessibly for a general educated audience.",
        "口语化 (Colloquial)": "[Tone Directive]: Highly conversational and colloquial style. Use casual, everyday language. Feel free to use phrases like 'pretty much', 'looks like', or conversational analogies."
    }

    tone_rules_zh = {
        "学术书面 (Formal Academic)": "【风格指令】：纯正客观的学术书面书写。使用高度严谨的书面语，【绝对禁止】将学术文本口语化。绝不使用"算是"、"看上去不错"等非正式表达，确保专业深度。但注意：学术严谨不等于堆砌对冲词，不要为了显得"谨慎"而反复使用"似乎"、"可能表明"、"在一定程度上"等——这恰恰是AI改写的典型痕迹。",
        "学术演讲 (Academic Presentation)": "【风格指令】：学术汇报/答辩演讲口吻。语言依然专业且保留核心术语，但句式长短更适宜讲述。允许出现"我们发现"、"这说明"等更具现场感的用语，避免过长的套娃式从句。不要堆砌对冲词。",
        "一般书面 (General Written)": "【风格指令】：标准的书面/科普表达。去除晦涩难懂的学术词语与大词，面向一般受众解答，语句通顺流畅，不过度堆砌名词，偏向技术博客或新闻报道的流畅质感。",
        "口语化 (Colloquial)": "【风格指令】：高度口语化与对话式的交流表达。使用通俗易懂的大白话、非正式用语，可以加入一些日常感情色彩词（如"算是"、"其实"、"看上去不错"），完全打破学术的严肃与刻板。"
    }

    extra_tone_en = tone_rules_en.get(tone, tone_rules_en["学术书面 (Formal Academic)"])
    extra_tone_zh = tone_rules_zh.get(tone, tone_rules_zh["学术书面 (Formal Academic)"])

    prompt_en = f"""You are an expert editor who humanizes academic writing.
    Your goal is to transform the provided AI-generated text into authentic human writing according to the specified tone.

    {extra_tone_en}

    Apply the following core strategies:
    1. Vary Sentence Rhythm (Burstiness): Mix short punchy sentences (5-10 words) with medium (15-20) and long complex ones (25-35+). Break up uniform sentence lengths.
    2. Reduce Abstract Scaffolding: Remove vague placeholder phrases like "various aspects", "in terms of", "multiple factors". Replace them with specific concepts, named theories, or concrete examples.
    3. Eliminate Mechanical Transitions: Remove formulaic connectors like "Moreover,", "Furthermore,", "Additionally,". Use implicit logic or varied transitions.
    4. Add Natural Voice: Show critical engagement and use natural academic terminology appropriate to the field.
    5. Ground in Specificity: Use concrete contexts instead of generic statements.

    Format Constraints:
    - Keep LaTeX formulas and formatting exactly as they are.
    - Output ONLY the rewritten text, with no explanations or rationale block.
    """

    prompt_zh = f"""你是一位专门润色学术文本的资深人类编辑。
    你的核心任务是去除文本中浮夸、空洞、机械的AI生成痕迹，将其转化为符合指定风格要求的人类真实表述。

    {extra_tone_zh}

    【最高优先级——内容保真原则】：
    - 润色后必须保留原文所有实质性信息，包括论点、实验方法、实验结果、数据指标、结论。不得遗漏任何事实性内容。
    - 润色的目的是改写表达方式，而非删减内容。
    - 润色后字数应与原文大致相当，不得大幅缩水。

    请应用以下核心策略：
    1. 增加句式错落感（Burstiness）：打破平均句长的均匀分布，交叉使用长短句，刻意消除文本的高度对称排比。
    2. 提纯词汇（降维）：将AI高频的伪高级大词替换为具体客观的表述。注意是"替换"而非"删除"。
    3. 消除机械答题模式：极力避免编号逻辑结构（如"首先、其次、综上所述"），通过内容的内在逻辑来衔接段落。
    4. 移除机器排版风格：禁止将文字改写成加粗短语起手的垂直列表。
    5. 原样保留所有的LaTeX公式。

    【需要替换的AI味表达】：
    - 旨在 → 替换为"为了"、"目的是"或直接省略
    - 总体来看 / 总体而言 / 整体来看 → 替换为直接陈述结论
    - 似乎 / 可能表明 / 或可 / 或可为 → 替换为更确定的表述
    - 在一定程度上 / 在某种程度上 → 替换为具体化表述
    - 有效解决了 → 替换为具体描述解决了什么问题
    - 实现了...的良好平衡 → 替换为具体说明平衡了什么
    - 核心痛点 → 替换为"主要问题"、"难点"
    - 综上所述 / 总而言之 / 由此可见 → 替换为用内容自然收束
    - 值得注意的是 / 需要强调的是 → 替换为直接说重点
    - 不可或缺 / 至关重要 / 举足轻重 → 替换为"重要"、"关键"等朴素词
    - 发挥着重要作用 → 替换为更具体的描述
    - 具有重要的现实意义 → 替换为具体说明有什么实际用途
    - 本文将重点研究 / 本文旨在 / 本文拟 → 替换为更自然的论文引入方式

    【知网/万方/维普检测系统重点识别的AI特征——必须避免】：
    1. "随着...的..."模板句式：如"随着深度学习的兴起"——改为"深度学习兴起之后"等更自然的表达。
    2. "基于...的..."模板句式：如"基于深度学习的目标检测算法"——改为"使用深度学习的目标检测算法"。
    3. "是...的"定义式堆砌：如"建筑行业是国民经济的重要引擎"——改为更自然的主动句。
    4. 多层定语嵌套：改为简洁独立的句子。
    5. 段落结构模板化：避免"背景→问题→意义→本文方案"的机械四段式结构。
    6. 引用全部集中在句末：尝试将引用放在句中。
    7. "因此"、"然而"等逻辑连接词密度过高：用语义衔接替代显式连接词。
    8. 信息密度均匀：刻意制造起伏——有的句子很短（5-8字），有的很长（30+字）。

    【可以保留的正常学术表达】：
    - "构建了"、"提出了"、"确立了" — 标准学术动词
    - "智能化"、"自动化" — 在CS/工程领域是专业术语
    - "显著"、"高效" — 有具体数据支撑时可正常使用
    - "此外"、"另外" — 偶尔使用是正常的

    【关键原则】：
    - 写得像人，不是写得像"试图模仿人的AI"。人类写作的特点是：直接、具体、有主见、不绕弯子。
    - 不要为了显得"谨慎"而堆砌对冲词。
    - 每句话都要有信息增量，有信息量的句子绝对不能删。

    格式要求：
    - 不要解释，禁止输出排版花样，直接输出纯净还原为自然连贯的段落文本。
    """

    system_prompt = prompt_en if lang == "English" else prompt_zh
    is_en = (lang == "English")
    MAX_ROUNDS = 5
    AI_THRESHOLD = 45

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    revised = text
    ai_score = 100
    ai_details = None

    try:
        for round_num in range(1, MAX_ROUNDS + 1):
            if round_num == 1:
                st.info(f"Pipeline Round {round_num}/{MAX_ROUNDS}: 初始改写...")
            else:
                st.info(f"Pipeline Round {round_num}/{MAX_ROUNDS}: AI分数 {ai_score} > {AI_THRESHOLD}，触发定点消除...")

            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.75 + (round_num - 1) * 0.05,
                frequency_penalty=0.3 + (round_num - 1) * 0.05,
                presence_penalty=0.2 + (round_num - 1) * 0.05
            )
            revised = response.choices[0].message.content
            messages.append({"role": "assistant", "content": revised})

            ai_score, ai_details = calculate_ai_rate(revised, lang)
            st.write(f"🔬 第{round_num}轮评估的AI近似指纹分数: {ai_score}/100")
            if ai_details:
                st.expander(f"第{round_num}轮AI评估详细指标").json(ai_details.get('metrics', {}))

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

            if ai_score <= AI_THRESHOLD:
                st.success(f"✅ AI分数已降至 {ai_score}/100，低于阈值 {AI_THRESHOLD}，停止迭代。")
                break

            if round_num < MAX_ROUNDS:
                feedback_str = _generate_feedback(ai_details, is_en)
                messages.append({"role": "user", "content": feedback_str})
            else:
                st.warning(f"⚠️ 已进行 {MAX_ROUNDS} 轮迭代，AI分数仍为 {ai_score}/100。建议手动微调。")

        if lang == "Chinese":
            revised = format_clean_chinese(revised)

        revised = format_formulas(revised, target_format)

        return revised, ai_score

    except Exception as e:
        return f"API调用出错: {str(e)}", 100


def _generate_feedback(ai_details, is_en):
    """根据检测指标生成针对性的反馈提示"""
    feedback_points = []

    if not ai_details or 'metrics' not in ai_details:
        msg = "Please further vary sentence lengths, remove all formulaic transitions, and drastically reduce empty wording." if is_en else "请进一步打散句子长度，替换刻意的逻辑连接词为自然衔接，并将空泛用词替换为朴实具体的表述。"
        feedback_points.append(msg)
        return "\n".join(feedback_points)

    metrics = ai_details['metrics']

    uni_score = metrics.get('sentence_uniformity', {}).get('score', 0)
    if uni_score > 0.3:
        msg = "1. 句长分布依然过于均匀（缺乏Burstiness），请进一步刻意打散长短句。" if not is_en else "1. Sentence length distribution is too uniform (lacks burstiness). Mix very short sentences (5-10 words) with long complex ones (30+ words)."
        feedback_points.append(msg)

    trans_count = metrics.get('transition_overuse', {}).get('count', 0)
    if trans_count > 0:
        msg = f"2. 滥用了机械过渡词（被检测到 {trans_count} 次）。" if not is_en else f"2. Overused mechanical transition words (detected {trans_count} times)."
        feedback_points.append(msg)

    abs_metric = metrics.get('abstract_language', {})
    abs_count = abs_metric.get('count', abs_metric.get('total_count', 0))
    if abs_count > 0:
        msg = f"3. 存在较多空泛套话和大词（被检测到 {abs_count} 次）。" if not is_en else f"3. Contains abstract placeholder phrases (detected {abs_count} times)."
        feedback_points.append(msg)

    if not is_en:
        over_hedge = metrics.get('over_hedging', {})
        hedge_count = over_hedge.get('count', 0)
        if hedge_count >= 3:
            hedge_items = over_hedge.get('items', [])
            hedge_examples = "、".join([f'"{h[0]}"' for h in hedge_items[:5]])
            feedback_points.append(f'4. 【严重AI痕迹】过度使用对冲词（检测到 {hedge_count} 个，如{hedge_examples}）。请替换为更直接、确定的表述。')

        ttr_obj = metrics.get('bigram_ttr', {})
        if ttr_obj.get('value', 1.0) < 0.65:
            feedback_points.append("5. 高频二元词重复率过高（Bigram TTR 极低）。请大幅度更换近义词修饰与表达。")

        clause_obj = metrics.get('clause_chain_density', {})
        if clause_obj.get('value', 0) > 3.2:
            feedback_points.append("6. 句法嵌套过深。请立即将超长定语断开，转换为多个独立短句。")

        opening_rep = metrics.get('sentence_opening_repetition', {})
        if opening_rep.get('count', 0) >= 3:
            feedback_points.append(f'7. 句首模式重复（"{opening_rep.get("top_pattern", "")}" 开头出现了 {opening_rep.get("count", 0)} 次）。')

        idiom_obj = metrics.get('idiom_overuse', {})
        if idiom_obj.get('count', 0) >= 4:
            feedback_points.append(f"8. 成语/四字词组堆砌过多（{idiom_obj.get('count', 0)} 个）。请将多余的成语替换为平实表述。")

        punct_obj = metrics.get('punctuation_density', {})
        if punct_obj.get('comma_period_ratio', 0) > 3.5:
            feedback_points.append(f"9. 逗号/句号比过高，说明单句内从句嵌套过多。请多用句号断句。")

        concluding_obj = metrics.get('concluding_formula', {})
        if concluding_obj.get('count', 0) >= 2:
            feedback_points.append(f'10. 段末总结套话过多（{concluding_obj.get("count", 0)} 处）。')

        suizhe_obj = metrics.get('suizhe_template', {})
        if suizhe_obj.get('count', 0) >= 2:
            feedback_points.append(f'11. 【知网级AI特征】检测到 {suizhe_obj.get("count", 0)} 处"随着/基于...的..."模板句式。请改为"X之后，Y..."等自然表达。')

        para_template = metrics.get('paragraph_template', {})
        if para_template.get('marker_count', 0) >= 3:
            feedback_points.append(f'12. 【知网级AI特征】段落结构过于模板化。请打乱"背景→问题→意义→本文方案"的结构顺序。')

        def_pattern = metrics.get('definition_pattern', {})
        if def_pattern.get('count', 0) >= 2:
            feedback_points.append(f'13. 【知网级AI特征】检测到 {def_pattern.get("count", 0)} 处"是...的"定义式句式。')

        citation_obj = metrics.get('citation_distribution', {})
        if citation_obj.get('end_citation_ratio', 0) > 0.8 and citation_obj.get('total_citations', 0) >= 3:
            feedback_points.append(f'14. 【知网级AI特征】引用过度集中在句末。请将部分引用移到句中。')

    else:
        burst_obj = metrics.get('burstiness', {})
        if burst_obj.get('score', 0) > 0.3:
            feedback_points.append("4. Sentence burstiness is too low. Mix very short sentences with long complex ones.")

        hedge_obj = metrics.get('hedging_overuse', {})
        if hedge_obj.get('score', 0) > 0.3:
            feedback_points.append(f"5. Excessive hedging language ({hedge_obj.get('count', 0)} instances). Use more direct academic language.")

    if not feedback_points:
        msg = "Please further vary sentence lengths perfectly, remove all formulaic transitions, and drastically reduce empty wording." if is_en else "请进一步打散句子长度，使其长短交错，替换刻意的逻辑连接词为自然衔接。"
        feedback_points.append(msg)

    feedback_str = "\n".join(feedback_points)
    if is_en:
        return f"The previous output still retains machine-generated stiffness. The system detected the following AI markers:\n\n{feedback_str}\n\nPlease rewrite the text, maintaining logic and professional rigor, but absolutely eliminate the AI characteristics mentioned above."
    else:
        return '上一次的改写依然残留机器生成的生硬感。系统检测程序发现了以下机器味缺陷：\n\n' + feedback_str + '\n\n请基于上述缺陷逐一自纠并重新输出。注意是"替换"而非"删除"——每一条被修改的表述都必须保留其原有的实质信息。不得遗漏原文中的任何论点、实验结果或结论。'


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
input_text = st.text_area("在此粘贴需要润色的段落 (包含LaTeX公式请保留 $ 或 $$)：", height=200)

if st.button("🚀 运行 Humanize Pipeline"):
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
