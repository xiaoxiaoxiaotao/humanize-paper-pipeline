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
    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

def calculate_ai_rate(text, lang):
    """
    一个简单的基于启发式规则的高级“AI率”估算。
    分数越高，代表文本越像AI生成的。返回0-100的参考分数。
    """
    if lang == "English" and HAS_DETECTOR:
        try:
            detector = AIDetector(text)
            result = detector.analyze()
            # 将 0.0-1.0 映射到 0-100
            score = int(result['overall_score'] * 100)
            details = result
            return min(100, max(0, score)), details
        except Exception as e:
            pass

    if lang in ["Chinese", "中文 (Chinese)"]:
        try:
            return analyze_chinese_text(text)
        except Exception as e:
            pass

    # 针对其他情况的回退保护
    score = 0
    words = len(text.split()) if lang == "English" else len(text)
    if words == 0: return 0, None

    return min(100, int(score)), None

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
        "Psychology": "Use precise operational definitions and behavioral mechanisms. Experimental design descriptions should be rigid but natural."
    }
    
    discipline_rules_zh = {
        "Computer Science": "强调技术精确性、算法逻辑和系统架构。不要对技术过程进行过度修饰。在描述系统和方法时，使用平实的陈述句或第一人称（如“本文提出”）是完全可以接受的。",
        "Engineering": "侧重于实际应用、设计约束、性能指标和方法论。保持极其客观、数据驱动的语气。",
        "Economics/Business": "侧重于实证证据、因果推断和经济模型。简洁地使用标准的计量经济学或商业术语。",
        "Sociology": "熟练使用资本、阶层、制度等社会学概念。定性描述应体现反思性，定量部分则保持客观。",
        "Anthropology": "看重民族志细节、“深描”和反思性语态。在这里，主观和描述性的语言更加被接受。",
        "Political Science": "强调制度框架、权力动态和假设检验语言。",
        "Education": "关注教学法、学习结果和教育公平。突出政策相关性。",
        "Psychology": "必须使用精确的操作性定义和行为机制术语。实验设计的描述应当严谨自然。"
    }
    
    extra_rule_en = discipline_rules_en.get(discipline, "")
    extra_rule_zh = discipline_rules_zh.get(discipline, "")
    
    # 语气特定的补充约束
    tone_rules_en = {
        "学术书面 (Formal Academic)": "[Tone Directive]: Strictly formal academic writing. Use rigorous written scholarly language. ABSOLUTELY NO colloquialisms or informal phrasing (e.g., avoid 'looks like', 'kind of'). Maintain maximum professional depth.",
        "学术演讲 (Academic Presentation)": "[Tone Directive]: Academic presentation/conference style. Maintain scholarly rigor and terminology but use slightly shorter, speakable sentences. Phrases like 'We found that' or 'This implies' are acceptable for rhetorical flow.",
        "一般书面 (General Written)": "[Tone Directive]: General written/technical blog style. Remove overly dense academic jargon. Write cleanly and accessibly for a general educated audience without overusing nominalizations.",
        "口语化 (Colloquial)": "[Tone Directive]: Highly conversational and colloquial style. Use casual, everyday language. Feel free to use phrases like 'pretty much', 'looks like', or conversational analogies to completely break the rigid academic tone."
    }
    
    tone_rules_zh = {
        "学术书面 (Formal Academic)": "【风格指令】：纯正客观的学术书面书写。使用高度严谨的书面语，【绝对禁止】将学术文本口语化。请务必保持学术严谨性，绝不使用“算是”、“似乎”、“看上去不错”等非正式或口语化表达，确保原有的专业深度。",
        "学术演讲 (Academic Presentation)": "【风格指令】：学术汇报/答辩演讲口吻。语言依然专业且保留核心术语，但句式长短更适宜讲述。允许出现“我们发现”、“这说明”、“值得注意的是”等更具现场感的用语，避免过长的套娃式从句。",
        "一般书面 (General Written)": "【风格指令】：标准的书面/科普表达。去除晦涩难懂的学术词语与大词，面向一般受众解答，语句通顺流畅，不过度堆砌名词，偏向技术博客或新闻报道的流畅质感。",
        "口语化 (Colloquial)": "【风格指令】：高度口语化与对话式的交流表达。使用通俗易懂的大白话、非正式用语，可以加入一些日常感情色彩词（如“算是”、“其实”、“看上去不错”），完全打破学术的严肃与刻板。"
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

    请应用以下核心策略：
    1. 遵循风格与专业度：在满足【风格指令】的前提下，保留核心论证逻辑与案例事实，润色后字数原则上不要剧烈缩水。
    2. 增加句式错落感（Burstiness）：打破平均15-20字的均匀句式，交叉使用长短句，以及倒装、定语前置等符合人类习惯的复杂句型，刻意消除文本的高度对称排比。
    3. 提纯词汇（降维）：删除AI高频的伪高级大词（如“具有深远意义”、“不可忽视”、“各个方面”等），替换为具体客观的表述。避免浮夸的辞藻。
    4. 消除机械答题模式：极力避免编号逻辑结构（如“首先、其次、综上所述”），不要在段末附加多余的总结套话，通过内容的内在逻辑来衔接段落结构。
    5. 移除机器排版风格：如果是普通自然段落，坚决禁止将文字改写成频繁使用加粗短语起手的垂直列表（禁止使用如“**一、核心问题：**”格式）。
    6. 原样保留所有的LaTeX公式，绝对不要擅自更改数学符号或排版结构。

    格式要求：
    - 不要解释，禁止输出排版花样，直接输出纯净还原为自然连贯的段落文本。
    """
    
    system_prompt = prompt_en if lang == "English" else prompt_zh
    
    st.info("Pipeline Step 1: Requesting initial humanization rewrite...")
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.75,
            frequency_penalty=0.4,
            presence_penalty=0.3
        )
        revised = response.choices[0].message.content
        
        # Calculate AI score
        ai_score, ai_details = calculate_ai_rate(revised, lang)
        st.write(f"🔬 第一轮评估的AI近似指纹分数: {ai_score}/100")
        if ai_details:
            st.expander("AI评估详细指标").json(ai_details.get('metrics', {}))
        
        if ai_score > 35:
            st.info("AI分数高于阈值，触发Pipeline Step 2: 依据探针自动给出的反馈意见要求大模型进行定点消除...")
            
            # 动态提取启发式反馈，组织反馈话术
            feedback_points = []
            is_en = (lang == "English")
            if ai_details and 'metrics' in ai_details:
                metrics = ai_details['metrics']
                # 检查句子均匀度
                uni_score = metrics.get('sentence_uniformity', {}).get('score', 0)
                if uni_score > 0.3:
                    msg = "1. Sentence length distribution is too uniform (lacks burstiness). Please break up the sentences drastically, mixing very short sentences (5-10 words) with long complex ones (30+ words)." if is_en else "1. 句长分布依然过于均匀分布（缺乏自然学术行文的长短句起伏/Burstiness），请进一步刻意打散长短句，插入5-10字短句与30字复杂长句交错。"
                    feedback_points.append(msg)
                
                # 检查过多过渡词
                trans_count = metrics.get('transition_overuse', {}).get('count', 0)
                if trans_count > 0:
                    msg = f"2. Overused mechanical transition words (detected {trans_count} times). Please remove these rigid connectors entirely and rely strictly on contextual meaning for transitions." if is_en else f"2. 滥用了机械刻板的过渡词（如“此外”、“总而言之”等被检测到 {trans_count} 次），请全部删除这些僵硬的连接，完全通过上下文语义本身来自然承接逻辑。"
                    feedback_points.append(msg)
                
                # 检查套话大词
                abs_count = metrics.get('abstract_language', {}).get('count', 0)
                if abs_count == 0 and 'total_count' in metrics.get('abstract_language', {}): # 兼容英文分析器返回格式
                    abs_count = metrics.get('abstract_language', {}).get('total_count', 0)
                    
                if abs_count > 0:
                    msg = f"3. Contains abstract placeholder phrases or empty wording (detected {abs_count} times). Please replace vague scaffolding with concrete concepts and specific theories." if is_en else f"3. 存在较多空泛、喜欢卖弄的虚词和套话（如“极具重要意”、“多种因素”等套路被检测到 {abs_count} 次），请坚决删掉这些毫无信息密度的字眼，改为具体的理论论述或直接说事，用词必须朴实。"
                    feedback_points.append(msg)
                
                # 学术对冲词检查
                if not is_en:
                    hedge_count = metrics.get('scholarly_hedging', {}).get('count', 0)
                    if hedge_count == 0:
                        feedback_points.append("4. 行文过于具备机器般的绝对权威感，缺乏真实大学生/学者的谦逊与克制（缺乏对冲语气）。请在合适的地方加入严谨的对冲表达（如“可能表明”、“似乎”、“在探讨某种程度上”等）。")

                # 提取 NLP 技术特征反馈 (仅针对中文版)
                if not is_en:
                    ttr_obj = metrics.get('bigram_ttr', {})
                    if ttr_obj.get('value', 1.0) < 0.65:
                        feedback_points.append("5. 机器指纹暴露：高频二元词重复率过高（Bigram TTR 极低）。AI极其喜欢反复套用熟练度高的固定词组。请大幅度更换近义词修饰与表达，绝对不要在一段内反复复用相似的组合或词汇。")
                        
                    clause_obj = metrics.get('clause_chain_density', {})
                    if clause_obj.get('value', 0) > 3.8:
                        feedback_points.append("6. 机器指纹暴露：句法嵌套过深（平均单句逗号数太多，Clause Density极高）。大模型写作特喜欢叠床架屋地使用绵长定语从句。请立即挥刀将超长定语断开，转换为多个清爽独立的短陈述句进行论述。")

            if not feedback_points:
                msg = "Please further vary sentence lengths perfectly, remove all formulaic transitions, and drastically reduce empty wording." if is_en else "请进一步打散句子长度，使其长短交错，完全隐去刻意的逻辑连接词，并降低用词的虚无与卖弄感。"
                feedback_points.append(msg)
                
            feedback_str = "\n".join(feedback_points)
            refine_prompt = (f"The previous output still retains machine-generated stiffness. The system detected the following critical AI markers:\n\n{feedback_str}\n\nPlease rigorously self-correct based on these specific flaws and rewrite the text. Maintain logic and professional rigor, but absolutely eliminate the AI characteristics mentioned above." 
                             if is_en else
                             f"上一次的改写依然残留机器生成的生硬感。系统检测程序发现了以下致命的机器味缺陷：\n\n{feedback_str}\n\n请严格基于上述缺陷逐一自纠并重新输出一遍结果。除了修复这些问题，保持其余论证部分逻辑和专业度。必须保证彻底消灭以上指出的AI特征。")
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": revised},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.85,
                frequency_penalty=0.6,
                presence_penalty=0.4
            )
            revised = response.choices[0].message.content
            ai_score, ai_details2 = calculate_ai_rate(revised, lang)
            st.write(f"🔬 第二轮评估的AI近似指纹分数: {ai_score}/100")
            if ai_details2:
                st.expander("第二轮AI评估详细指标").json(ai_details2.get('metrics', {}))
            
        # Post-Processing
        if lang == "中文 (Chinese)":
            revised = format_clean_chinese(revised)
            
        revised = format_formulas(revised, target_format)
            
        return revised, ai_score
        
    except Exception as e:
        return f"API调用出错: {str(e)}", 100

st.title("🎓 Humanize Academic Paper Pipeline")
st.markdown("基于多轮API调用和规则过滤的AI论文防查重、自然化润色工具。")

with st.sidebar:
    st.header("⚙️ API Settings")
    api_base = st.text_input("Base URL", value="https://api.openai.com/v1")
    api_key = st.text_input("API Key", type="password")
    model_id = st.text_input("Model ID", value="gpt-4o")
    
    st.header("📝 Options")
    lang_opt = st.radio("Language", ["中文 (Chinese)", "English"])
    format_opt = st.radio("Target Format", ["LaTeX", "Word (Separated Formulas)"])
    tone_opt = st.selectbox(
        "输出风格 (Output Tone)",
        ["学术书面 (Formal Academic)", "学术演讲 (Academic Presentation)", "一般书面 (General Written)", "口语化 (Colloquial)"]
    )
    discipline_opt = st.selectbox(
        "Discipline",
        ["Computer Science", "Engineering", "Economics/Business", "Sociology", "Anthropology", "Political Science", "Education", "Psychology"]
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
        
        if orig_score > 60:
            st.error(f"润色前原始文本AI评分: {orig_score}/100 (强AI痕迹)")
        elif orig_score > 35:
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
        
        if final_score < 35:
            st.success(f"最终AI味评分过关 ({final_score}/100)")
        elif final_score < 60:
            st.warning(f"最终AI味评分尚可 ({final_score}/100)，建议手动微调")
        else:
            st.error(f"最终AI味评分偏高 ({final_score}/100)")
            
        st.text_area("复制结果", final_text, height=300)
