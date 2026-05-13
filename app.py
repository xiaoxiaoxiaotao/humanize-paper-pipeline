import streamlit as st
import openai
import re
import numpy as np
import sys
import os

# 引入刚刚从 skill 中提取的高级检测算法
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
try:
    from ai_detector import AIDetector
    from zh_detector import analyze_chinese_text
    from text_analyzer import TextAnalyzer
    HAS_DETECTOR = True
    HAS_ANALYZER = True
except ImportError:
    HAS_DETECTOR = False
    HAS_ANALYZER = False

def calculate_ai_rate(text, lang):
    """
    一个简单的基于启发式规则的高级"AI率"估算。
    分数越高，代表文本越像AI生成的。返回0-100的参考分数。
    """
    if not text or not text.strip():
        return 0, None

    text_len = len(text)
    if text_len < 10:
        return 0, None

    clean_text = _strip_latex(text)
    clean_len = len(clean_text.strip())
    if clean_len < 10:
        st.warning("文本过短或主要为LaTeX公式，AI检测结果仅供参考")
        return 0, None

    latex_ratio = 1.0 - (clean_len / max(text_len, 1))
    if latex_ratio > 0.7:
        st.info(f"检测到文本中约 {latex_ratio*100:.0f}% 为LaTeX公式，已自动剥离后分析")

    if lang == "English" and HAS_DETECTOR:
        try:
            detector = AIDetector(clean_text)
            result = detector.analyze()
            score = int(result['overall_score'] * 100)
            return min(100, max(0, score)), result
        except Exception as e:
            pass

    if lang in ["Chinese", "中文 (Chinese)"]:
        try:
            return analyze_chinese_text(clean_text)
        except Exception as e:
            pass

    if not HAS_DETECTOR:
        st.warning("AI检测模块未加载，使用基础启发式回退检测（精度有限）")

    score = _fallback_ai_detection(clean_text, lang)
    return min(100, max(0, int(score))), None


def _strip_latex(text):
    """剥离LaTeX公式，保留纯文本用于分析"""
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*?\$', ' ', text)
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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

def format_clean_chinese(text):
    """清理中文文本中多余的空格和错用的引号"""
    # 1. 删除中英文/数字之间的空格
    text = re.sub(r'([\u4e00-\u9fa5])\s+([a-zA-Z0-9])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([\u4e00-\u9fa5])', r'\1\2', text)
    
    # 2. 替换英文引号为中文引号 (简单的状态翻转替换)
    # 此处假设配对出现，复杂的公式内的情况可能需要更细致的AST，这里用基础的正则/替换
    parts = text.split('"')
    for i in range(1, len(parts), 2):
        parts[i] = f'“{parts[i]}”'
    text = ''.join(parts)
    
    # 单引号替换
    parts = text.split("'")
    for i in range(1, len(parts), 2):
        parts[i] = f'‘{parts[i]}’'
    text = ''.join(parts)
    
    return text

def format_formulas(text, target_format):
    """根据word或是latex的格式调整公式"""
    if target_format == "LaTeX":
        return text # LaTeX 保持原样
    
    # Word: 提取独立行公式($$..$$)和行内公式($..$)，并提出去方便复制
    formulas = []
    
    # 提取多行公式
    def replace_block(match):
        formulas.append(match.group(0))
        return f"[公式 {len(formulas)} 见下方]"
        
    text = re.sub(r'\$\$.*?\$\$', replace_block, text, flags=re.DOTALL)
    
    # 将公式分列到文本结尾，Word下方便剥离复制
    if formulas:
        text += "\n\n=== 提取的数学公式表 ===\n"
        for i, f in enumerate(formulas, 1):
            text += f"\n公式 {i}:\n{f}\n"
            
    return text

def process_pipeline(text, lang, target_format, discipline, tone, api_base, api_key, model_id):
    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    
    # 领域特定的补充约束
    discipline_rules_en = {
        "Computer Science": "Emphasize technical precision, algorithmic logic, and system architecture. Do not over-embellish technical descriptions. It is acceptable and normal to use passive voice or straightforward active voice (e.g., 'We propose') when describing systems and methodologies.",
        "Engineering": "Focus on practical applications, design constraints, performance metrics, and methodology. Keep the tone highly objective and data-driven.",
        "Economics/Business": "Focus on empirical evidence, causal inference, and economic models. Use standard business or econometric terminology concisely.",
        "Sociology": "Use concepts like stratification, agency, and institutions. Qualitative descriptions should sound reflexive, while quantitative sections should be objective.",
        "Anthropology": "Values ethnographic detail, 'thick description', and reflexive voice. Subjective and descriptive language is more acceptable here.",
        "Political Science": "Emphasize institutional frameworks, power dynamics, and hypothesis testing.",
        "Education": "Focus on pedagogy, learning outcomes, and equity. Policy relevance is often highlighted.",
        "Psychology": "Use precise operational definitions and behavioral mechanisms. Experimental design descriptions should be rigid but natural.",
        "其他 (Other)": "Follow general academic writing conventions appropriate for your field. Maintain appropriate formality, precision, and logical flow. Use terminology standard in your discipline without unnecessary embellishment."
    }

    discipline_rules_zh = {
        "Computer Science": "强调技术精确性、算法逻辑和系统架构。不要对技术过程进行过度修饰。在描述系统和方法时，使用平实的陈述句或第一人称（如\"本文提出\"）是完全可以接受的。",
        "Engineering": "侧重于实际应用、设计约束、性能指标和方法论。保持极其客观、数据驱动的语气。",
        "Economics/Business": "侧重于实证证据、因果推断和经济模型。简洁地使用标准的计量经济学或商业术语。",
        "Sociology": "熟练使用资本、阶层、制度等社会学概念。定性描述应体现反思性，定量部分则保持客观。",
        "Anthropology": "看重民族志细节、\"深描\"和反思性语态。在这里，主观和描述性的语言更加被接受。",
        "Political Science": "强调制度框架、权力动态和假设检验语言。",
        "Education": "关注教学法、学习结果和教育公平。突出政策相关性。",
        "Psychology": "必须使用精确的操作性定义和行为机制术语。实验设计的描述应当严谨自然。",
        "其他 (Other)": "遵循您所在领域的一般学术写作规范。保持适当的正式性、精确性和逻辑流畅性。使用您所在学科的标准术语，避免不必要的修饰。"
    }

    extra_rule_en = discipline_rules_en.get(discipline, discipline_rules_en["其他 (Other)"])
    extra_rule_zh = discipline_rules_zh.get(discipline, discipline_rules_zh["其他 (Other)"])
    
    # 语气特定的补充约束
    tone_rules_en = {
        "学术书面 (Formal Academic)": "[Tone Directive]: Strictly formal academic writing. Use rigorous written scholarly language. ABSOLUTELY NO colloquialisms or informal phrasing (e.g., avoid 'looks like', 'kind of'). Maintain maximum professional depth.",
        "学术演讲 (Academic Presentation)": "[Tone Directive]: Academic presentation/conference style. Maintain scholarly rigor and terminology but use slightly shorter, speakable sentences. Phrases like 'We found that' or 'This implies' are acceptable for rhetorical flow.",
        "一般书面 (General Written)": "[Tone Directive]: General written/technical blog style. Remove overly dense academic jargon. Write cleanly and accessibly for a general educated audience without overusing nominalizations.",
        "口语化 (Colloquial)": "[Tone Directive]: Highly conversational and colloquial style. Use casual, everyday language. Feel free to use phrases like 'pretty much', 'looks like', or conversational analogies to completely break the rigid academic tone."
    }
    
    tone_rules_zh = {
        "学术书面 (Formal Academic)": '【风格指令】：纯正客观的学术书面书写。使用高度严谨的书面语，【绝对禁止】将学术文本口语化。绝不使用"算是"、"看上去不错"等非正式表达，确保专业深度。但注意：学术严谨不等于堆砌对冲词，不要为了显得"谨慎"而反复使用"似乎"、"可能表明"、"在一定程度上"等——这恰恰是AI改写的典型痕迹。',
        "学术演讲 (Academic Presentation)": '【风格指令】：学术汇报/答辩演讲口吻。语言依然专业且保留核心术语，但句式长短更适宜讲述。允许出现"我们发现"、"这说明"等更具现场感的用语，避免过长的套娃式从句。不要堆砌对冲词。',
        "一般书面 (General Written)": '【风格指令】：标准的书面/科普表达。去除晦涩难懂的学术词语与大词，面向一般受众解答，语句通顺流畅，不过度堆砌名词，偏向技术博客或新闻报道的流畅质感。',
        "口语化 (Colloquial)": '【风格指令】：高度口语化与对话式的交流表达。使用通俗易懂的大白话、非正式用语，可以加入一些日常感情色彩词（如"算是"、"其实"、"看上去不错"），完全打破学术的严肃与刻板。'
    }
    
    extra_tone_en = tone_rules_en.get(tone, "")
    extra_tone_zh = tone_rules_zh.get(tone, "")
    
    prompt_en = f"""You are an expert editor who humanizes academic writing, specifically in the field of {discipline}.
    Your goal is to transform the provided AI-generated text into authentic human writing according to the specified tone.
    
    {extra_tone_en}
    
    Domain-Specific Constraints for {discipline}:
    {extra_rule_en}
    
    Apply the following core strategies:
    1. Vary Sentence Rhythm (Burstiness): Mix short punchy sentences (5-10 words) with medium (15-20) and long complex ones (25-35+). Break up uniform sentence lengths.
    2. Reduce Abstract Scaffolding: Remove vague placeholder phrases like "various aspects", "in terms of", "multiple factors". Replace them with specific concepts, named theories, or concrete examples.
    3. Eliminate Mechanical Transitions: Remove formulaic connectors like "Moreover,", "Furthermore,", "Additionally,". Use implicit logic, content-driven connections, or phrases like "Building on this insight..."
    4. Add Scholarly Voice: Use appropriate hedging ("may suggest", "appears to", "potentially"), show critical engagement, and use natural {discipline} terminology.
    5. Ground in Specificity: Where appropriate, use concrete contexts instead of generic statements.
    
    Format Constraints:
    - Keep LaTeX formulas and formatting exactly as they are.
    - Output ONLY the rewritten text, with no explanations or rationale block.
    """
    
    prompt_zh = f"""你是一位专门为{discipline}领域润色的资深人类编辑。
    你的核心任务是去除文本中浮夸、空洞、机械的AI生成痕迹，将其转化为符合指定风格要求的人类真实表述。

    {extra_tone_zh}

    {discipline} 领域的专属写作约束：
    {extra_rule_zh}

    【最高优先级——内容保真原则】：
    - 润色后必须保留原文所有实质性信息，包括但不限于：论点、实验方法、实验结果、数据指标、结论。不得遗漏任何事实性内容。
    - 润色的目的是改写表达方式，而非删减内容。如果原文某句话包含具体信息，你必须用另一种更自然的表述方式重写它，而不是直接删掉。
    - 润色后字数应与原文大致相当，不得大幅缩水。如果发现字数明显减少，说明你删除了过多内容，这是错误的。

    请应用以下核心策略：
    1. 遵循风格与专业度：在满足【风格指令】的前提下，保留核心论证逻辑与案例事实。
    2. 增加句式错落感（Burstiness）：打破平均15-20字的均匀句式，交叉使用长短句，以及倒装、定语前置等符合人类习惯的复杂句型，刻意消除文本的高度对称排比。
    3. 提纯词汇（降维）：将AI高频的伪高级大词替换为具体客观的表述。注意是"替换"而非"删除"——例如"有效解决了"应改为具体描述解决了什么问题，而非直接删掉这句话。
    4. 消除机械答题模式：极力避免编号逻辑结构（如"首先、其次、综上所述"），不要在段末附加多余的总结套话，通过内容的内在逻辑来衔接段落结构。但注意：去掉总结套话后，套话中包含的实质信息仍需保留在正文中。
    5. 移除机器排版风格：如果是普通自然段落，坚决禁止将文字改写成频繁使用加粗短语起手的垂直列表（禁止使用如"**一、核心问题：**"格式）。
    6. 原样保留所有的LaTeX公式，绝对不要擅自更改数学符号或排版结构。

    【需要替换的AI味表达】（这些是AI生成文本的指纹，请替换为更自然的表述，而非直接删除）：
    - 旨在 → 替换为"为了"、"目的是"或直接省略
    - 总体来看 / 总体而言 / 整体来看 → 替换为直接陈述结论
    - 似乎 / 可能表明 / 或可 / 或可为 → 替换为更确定的表述，过度对冲是AI改写的头号特征
    - 在一定程度上 / 在某种程度上 / 某种平衡 → 替换为具体化表述
    - 有效解决了 → 替换为具体描述解决了什么问题、效果如何
    - 实现了...的良好平衡 → 替换为具体说明平衡了什么
    - 提供了...的技术方案 → 替换为更朴实的说法
    - 核心痛点 → 替换为"主要问题"、"难点"
    - 综上所述 / 总而言之 / 由此可见 → 替换为用内容自然收束
    - 值得注意的是 / 需要强调的是 → 替换为直接说重点
    - 不可或缺 / 至关重要 / 举足轻重 → 替换为"重要"、"关键"等朴素词
    - 发挥着重要作用 → 替换为更具体的描述，如"推动了..."、"支撑了..."
    - 具有重要的现实意义 → 替换为具体说明有什么实际用途
    - 本文将重点研究 / 本文旨在 / 本文拟 → 替换为更自然的论文引入方式，如直接陈述研究内容

    【知网/万方/维普检测系统重点识别的AI特征——必须避免】：
    1. "随着...的..."模板句式：如"随着深度学习的兴起"、"随着施工区域的扩大"——这是AI写论文的标志性句式。请改为："深度学习兴起之后"、"施工区域扩大时"等更自然的表达。
    2. "基于...的..."模板句式：如"基于深度学习的目标检测算法"——改为"使用深度学习的目标检测算法"或"以深度学习为基础的目标检测算法"。
    3. "是...的"定义式堆砌：如"建筑行业是国民经济的重要引擎"——改为"建筑行业推动国民经济发展"或"国民经济离不开建筑行业的支撑"。
    4. 多层定语嵌套：如"复杂的施工环境和高风险的工作内容"——改为"施工现场环境复杂、工作内容风险高"。
    5. 段落结构模板化：避免"背景→问题→意义→本文方案"的机械四段式结构。可以适当打乱顺序，或将意义和背景融合。
    6. 引用全部集中在句末：如"...重要作用[1]。"——尝试将引用放在句中或改变引用位置。
    7. "因此"、"然而"等逻辑连接词密度过高：每段出现2次以上就太多了。用语义衔接替代显式连接词。
    8. 信息密度均匀：AI每句话长度和信息量都差不多。请刻意制造起伏——有的句子很短（5-8字），有的很长（30+字）。

    【可以保留的正常学术表达】（以下表达在学术文本中是正常的，不要误删）：
    - "构建了"、"提出了"、"确立了"、"展现了" — 这是标准学术动词
    - "智能化"、"自动化"、"数字化" — 在CS/工程领域是专业术语，可正常使用
    - "显著"、"高效"、"优异" — 有具体数据支撑时可正常使用
    - "此外"、"另外" — 偶尔使用是正常的，不要每个都删
    - "关键问题"、"关键挑战" — 正常学术表达

    【关键原则】：
    - 写得像人，不是写得像"试图模仿人的AI"。人类写作的特点是：直接、具体、有主见、不绕弯子。
    - 不要为了显得"谨慎"而堆砌对冲词。真正的学术对冲是"有待进一步验证"、"尚需探讨"，而不是"似乎在一定程度上可能表明"。
    - 每句话都要有信息增量，不要写废话。但反过来，有信息量的句子绝对不能删。
    - 人类写论文引言不会严格按照"背景→问题→意义→方案"的模板，而是根据思路自然流动。

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

            if HAS_ANALYZER and lang == "English" and round_num == 1:
                try:
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

        if lang == "中文 (Chinese)":
            revised = format_clean_chinese(revised)

        revised = format_formulas(revised, target_format)

        return revised, ai_score

    except Exception as e:
        return f"API调用出错: {str(e)}", 100


def _generate_feedback(ai_details, is_en):
    """根据检测指标生成针对性的反馈提示"""
    feedback_points = []

    if not ai_details or 'metrics' not in ai_details:
        msg = "Please further vary sentence lengths perfectly, remove all formulaic transitions, and drastically reduce empty wording." if is_en else "请进一步打散句子长度，使其长短交错，替换刻意的逻辑连接词为自然衔接，并将空泛用词替换为朴实具体的表述。"
        feedback_points.append(msg)
        return "\n".join(feedback_points)

    metrics = ai_details['metrics']

    uni_score = metrics.get('sentence_uniformity', {}).get('score', 0)
    if uni_score > 0.3:
        msg = "1. Sentence length distribution is too uniform (lacks burstiness). Please break up the sentences drastically, mixing very short sentences (5-10 words) with long complex ones (30+ words)." if is_en else "1. 句长分布依然过于均匀（缺乏自然学术行文的长短句起伏/Burstiness），请进一步刻意打散长短句，插入5-10字短句与30字复杂长句交错。"
        feedback_points.append(msg)

    trans_count = metrics.get('transition_overuse', {}).get('count', 0)
    if trans_count > 0:
        msg = f"2. Overused mechanical transition words (detected {trans_count} times). Please remove these rigid connectors entirely and rely strictly on contextual meaning for transitions." if is_en else f"2. 滥用了机械刻板的过渡词（被检测到 {trans_count} 次），请将这些过渡词替换为更自然的语义衔接方式，或直接通过上下文逻辑承接。"
        feedback_points.append(msg)

    abs_metric = metrics.get('abstract_language', {})
    abs_count = abs_metric.get('count', abs_metric.get('total_count', 0))
    if abs_count > 0:
        msg = f"3. Contains abstract placeholder phrases or empty wording (detected {abs_count} times). Please replace vague scaffolding with concrete concepts and specific theories." if is_en else f"3. 存在较多空泛套话和大词（被检测到 {abs_count} 次），请将这些空泛表述替换为具体的论述或朴实的说法，注意保留原有实质信息。"
        feedback_points.append(msg)

    if is_en:
        burst_obj = metrics.get('burstiness', {})
        if burst_obj.get('score', 0) > 0.3:
            feedback_points.append(f"4. Sentence burstiness is too low (CV={burst_obj.get('cv', 'N/A')}). Mix very short sentences (5-8 words) with long complex ones (25+ words) to create natural rhythm variation.")

        bigram_obj = metrics.get('bigram_ttr', {})
        if bigram_obj.get('score', 0) > 0.3:
            feedback_points.append(f"5. Bigram repetition is too high (TTR={bigram_obj.get('value', 'N/A')}). AI tends to reuse the same word pairs. Vary your phrasing and word combinations significantly.")

        clause_obj = metrics.get('clause_chain_density', {})
        if clause_obj.get('score', 0) > 0.3:
            feedback_points.append(f"6. Excessive clause chaining ({clause_obj.get('value', 'N/A')} commas per sentence on average). Break long sentences into shorter, clearer ones.")

        opening_obj = metrics.get('sentence_opening_repetition', {})
        if opening_obj.get('score', 0) > 0.3:
            feedback_points.append(f"7. Repetitive sentence openings detected ('{opening_obj.get('top_pattern', '')}' appears {opening_obj.get('count', 0)} times). Vary how you start sentences.")

        hedge_obj = metrics.get('hedging_overuse', {})
        if hedge_obj.get('score', 0) > 0.3:
            feedback_points.append(f"8. Excessive hedging language ({hedge_obj.get('count', 0)} instances). AI overuses 'may suggest', 'could indicate', etc. Use more direct, confident academic language.")

        def_obj = metrics.get('definition_pattern', {})
        if def_obj.get('score', 0) > 0.3:
            feedback_points.append(f"9. Too many 'X is a Y' definition patterns ({def_obj.get('count', 0)} detected). AI loves defining things. Use more varied sentence structures.")

        conc_obj = metrics.get('concluding_formula', {})
        if conc_obj.get('score', 0) > 0.3:
            feedback_points.append(f"10. Formulaic concluding phrases detected ({conc_obj.get('count', 0)} instances). Let content naturally conclude rather than using 'In conclusion' or 'To summarize'.")

        punct_obj = metrics.get('punctuation_density', {})
        if punct_obj.get('score', 0) > 0.3:
            feedback_points.append(f"11. High comma-to-period ratio ({punct_obj.get('comma_period_ratio', 'N/A')}). Use more periods to create shorter, punchier sentences.")

        cit_obj = metrics.get('citation_distribution', {})
        if cit_obj.get('score', 0) > 0.3:
            feedback_points.append(f"12. Citations clustered at sentence ends ({cit_obj.get('end_citation_ratio', 0)*100:.0f}%). Distribute citations more naturally within sentences.")

        info_obj = metrics.get('info_density_uniformity', {})
        if info_obj.get('score', 0) > 0.3:
            feedback_points.append("13. Information density is too uniform across sentences. Create natural peaks and valleys in information load.")

        template_obj = metrics.get('paragraph_template', {})
        if template_obj.get('score', 0) > 0.3:
            feedback_points.append(f"14. Template paragraph structure detected ({template_obj.get('marker_count', 0)}/4 markers). Avoid the mechanical background->problem->significance->approach pattern.")

        word_rep = metrics.get('word_repetition', {})
        if word_rep.get('score', 0) > 0.3:
            feedback_points.append(f"15. Excessive word repetition ({word_rep.get('repeat_ratio', 0)*100:.1f}% of words repeated 3+ times). Use synonyms and vary vocabulary.")

        sw_ttr = metrics.get('sliding_window_ttr', {})
        if sw_ttr.get('score', 0) > 0.3:
            feedback_points.append(f"16. Local vocabulary repetition detected (sliding window TTR={sw_ttr.get('avg_ttr', 'N/A')}). Vary word choice within paragraphs.")

    if not is_en:
        over_hedge = metrics.get('over_hedging', {})
        hedge_count = over_hedge.get('count', 0)
        if hedge_count >= 3:
            hedge_items = over_hedge.get('items', [])
            hedge_examples = "、".join([f'"{h[0]}"' for h in hedge_items[:5]])
            feedback_points.append(f'4. 【严重AI痕迹】过度使用对冲词/含糊表达（检测到 {hedge_count} 个，如{hedge_examples}）。这是AI改写文本的头号特征——AI为了模仿"学术谨慎"会堆砌"似乎"、"可能表明"、"在一定程度上"等词，但人类学者不会这样写。请将这些过度对冲词替换为更直接、确定的表述。如需表达不确定性，使用"有待验证"、"尚需探讨"等真正的人类学术表达。')

        ttr_obj = metrics.get('bigram_ttr', {})
        if ttr_obj.get('value', 1.0) < 0.65:
            feedback_points.append("5. 机器指纹暴露：高频二元词重复率过高（Bigram TTR 极低）。AI极其喜欢反复套用熟练度高的固定词组。请大幅度更换近义词修饰与表达，绝对不要在一段内反复复用相似的组合或词汇。")

        clause_obj = metrics.get('clause_chain_density', {})
        if clause_obj.get('value', 0) > 3.2:
            feedback_points.append("6. 机器指纹暴露：句法嵌套过深（平均单句逗号数太多，Clause Density偏高）。大模型写作特喜欢叠床架屋地使用绵长定语从句。请立即将超长定语断开，转换为多个清爽独立的短陈述句。")

        opening_rep = metrics.get('sentence_opening_repetition', {})
        if opening_rep.get('count', 0) >= 3:
            feedback_points.append(f"7. 机器指纹暴露：句首模式重复（\"{opening_rep.get('top_pattern', '')}\" 开头出现了 {opening_rep.get('count', 0)} 次）。请变换句首表达，避免同一模式反复出现。")

        idiom_obj = metrics.get('idiom_overuse', {})
        if idiom_obj.get('count', 0) >= 4:
            feedback_points.append(f"8. 机器指纹暴露：成语/四字词组堆砌过多（检测到 {idiom_obj.get('count', 0)} 个）。AI生成中文时特别喜欢堆砌成语，人类使用更克制。请将多余的成语替换为平实表述。")

        punct_obj = metrics.get('punctuation_density', {})
        if punct_obj.get('comma_period_ratio', 0) > 3.5:
            feedback_points.append(f"9. 机器指纹暴露：逗号/句号比过高（{punct_obj.get('comma_period_ratio', 0)}），说明单句内从句嵌套过多。请多用句号断句，减少逗号连接的长定语。")

        concluding_obj = metrics.get('concluding_formula', {})
        if concluding_obj.get('count', 0) >= 2:
            feedback_points.append(f'10. 机器指纹暴露：段末总结套话/本文指向句过多（检测到 {concluding_obj.get("count", 0)} 处，如"总体来看"、"本文将重点研究"等）。请将这些套话替换为内容的自然收束，或将"本文"指向句改为直接陈述研究内容。')

        suizhe_obj = metrics.get('suizhe_template', {})
        if suizhe_obj.get('count', 0) >= 2:
            examples = suizhe_obj.get('examples', [])
            ex_str = '、'.join(examples[:3]) if examples else ''
            feedback_points.append(f'11. 【知网级AI特征】检测到 {suizhe_obj.get("count", 0)} 处"随着/基于/通过...的..."模板句式（如"{ex_str}"）。这是知网/万方检测系统重点识别的AI标志。请将"随着X的Y"改为"X之后，Y..."、"当X时，Y..."等自然表达；将"基于X的Y"改为"使用X的Y"、"以X为基础的Y"。')

        para_template = metrics.get('paragraph_template', {})
        if para_template.get('marker_count', 0) >= 3:
            feedback_points.append(f'12. 【知网级AI特征】段落结构过于模板化（检测到 {para_template.get("marker_count", 0)}/4 个结构标记：背景→问题→意义→本文方案）。知网检测系统能识别这种机械结构。请打乱顺序，或将背景和意义融合，避免严格按照模板写作。')

        def_pattern = metrics.get('definition_pattern', {})
        if def_pattern.get('count', 0) >= 2:
            feedback_points.append(f'13. 【知网级AI特征】检测到 {def_pattern.get("count", 0)} 处"是...的"定义式句式。AI喜欢用"XX是YY"下定义。请将定义句改为主谓宾结构，如"XX推动了YY"而非"XX是YY的重要引擎"。')

        citation_obj = metrics.get('citation_distribution', {})
        if citation_obj.get('end_citation_ratio', 0) > 0.8 and citation_obj.get('total_citations', 0) >= 3:
            feedback_points.append(f'14. 【知网级AI特征】引用过度集中在句末（{citation_obj.get("total_citations", 0)} 个引用，{citation_obj.get("end_citation_ratio", 0)*100:.0f}% 在句末）。AI生成文本的引用位置非常机械。请将部分引用移到句中，或改变引用位置。')

        info_density = metrics.get('info_density_uniformity', {})
        if '惩罚' in info_density.get('details', ''):
            feedback_points.append('15. 信息密度过于均匀，每句话长度和信息量相近。请刻意制造起伏——插入短句（5-8字）打破节奏，或扩展某句话使其明显长于其他句子。')

    if not feedback_points:
        msg = "Please further vary sentence lengths perfectly, remove all formulaic transitions, and drastically reduce empty wording." if is_en else "请进一步打散句子长度，使其长短交错，替换刻意的逻辑连接词为自然衔接，并将空泛用词替换为朴实具体的表述。"
        feedback_points.append(msg)

    feedback_str = "\n".join(feedback_points)
    if is_en:
        return f"The previous output still retains machine-generated stiffness. The system detected the following critical AI markers:\n\n{feedback_str}\n\nPlease rigorously self-correct based on these specific flaws and rewrite the text. Maintain logic and professional rigor, but absolutely eliminate the AI characteristics mentioned above."
    else:
        return '上一次的改写依然残留机器生成的生硬感。系统检测程序发现了以下机器味缺陷：\n\n' + feedback_str + '\n\n请基于上述缺陷逐一自纠并重新输出。核心原则：写得像人，不是写得像"试图模仿人的AI"。将过度对冲词（如"似乎"、"可能表明"、"在一定程度上"）替换为更直接确定的表述，将总结套话（如"总体来看"、"综上所述"）替换为内容自然收束，将空泛大词替换为具体朴实的表述。注意是"替换"而非"删除"——每一条被修改的表述都必须保留其原有的实质信息。不得遗漏原文中的任何论点、实验结果或结论。'

st.title("🎓 Humanize Academic Paper Pipeline")
st.markdown("基于多轮API调用和规则过滤的AI论文防查重、自然化润色工具。")

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
    discipline_opt = st.selectbox(
        "Discipline",
        ["Computer Science", "Engineering", "Economics/Business", "Sociology", "Anthropology", "Political Science", "Education", "Psychology", "其他 (Other)"]
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
            st.error(f"润色前原始文本AI评分: {orig_score}/100 (强AI痕迹，与知网/万方检测水平相当)")
        elif orig_score > 45:
            st.warning(f"润色前原始文本AI评分: {orig_score}/100 (中等AI痕迹)")
        else:
            st.success(f"润色前原始文本AI评分: {orig_score}/100 (低AI痕迹，可能无需过度润色)")
            
        if orig_details:
            st.expander("查看原始文本的AI特征详细抓取指标").json(orig_details.get('metrics', {}))

        with st.spinner("Pipeline 运行中..."):
            final_text, final_score = process_pipeline(
                input_text, lang_param, format_opt, discipline_opt, tone_opt, api_base, api_key, model_id
            )
            
        st.subheader("✅ 输出结果")
        st.metric(label="AI分数降幅", value=f"{final_score}/100", delta=f"{final_score - orig_score} 分", delta_color="inverse")
        
        if final_score < 40:
            st.success(f"最终AI味评分过关 ({final_score}/100) — 已接近知网/万方低AI率水平")
        elif final_score < 65:
            st.warning(f"最终AI味评分尚可 ({final_score}/100)，建议手动微调")
        else:
            st.error(f"最终AI味评分偏高 ({final_score}/100)，建议重新润色")
            
        st.text_area("复制结果", final_text, height=300)
