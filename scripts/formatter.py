import re


def strip_latex(text):
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\$[^$]*?\$', ' ', text)
    text = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_clean_chinese(text):
    text = re.sub(r'([\u4e00-\u9fa5])\s+([a-zA-Z0-9])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z0-9])\s+([\u4e00-\u9fa5])', r'\1\2', text)

    parts = text.split('"')
    for i in range(1, len(parts), 2):
        parts[i] = f'\u201c{parts[i]}\u201d'
    text = ''.join(parts)

    parts = text.split("'")
    for i in range(1, len(parts), 2):
        parts[i] = f'\u2018{parts[i]}\u2019'
    text = ''.join(parts)

    return text


def format_formulas(text, target_format):
    if target_format == "LaTeX":
        return text

    formulas = []

    def replace_block(match):
        formulas.append(match.group(0))
        return f"[公式 {len(formulas)} 见下方]"

    text = re.sub(r'\$\$.*?\$\$', replace_block, text, flags=re.DOTALL)

    if formulas:
        text += "\n\n=== 提取的数学公式表 ===\n"
        for i, f in enumerate(formulas, 1):
            text += f"\n公式 {i}:\n{f}\n"

    return text
