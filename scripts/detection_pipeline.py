import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse
from typing import Dict, Tuple, Optional

from ai_detector import AIDetector
from zh_detector_enhanced import analyze_chinese_text
from text_analyzer import TextAnalyzer
from formatter import strip_latex
from enhancements import AdversarialRewriter, PerplexitySurrogate, humanize_with_adversarial_rules


class DetectionPipeline:
    """
    统一的AI检测管道

    整合所有检测功能:
    - 英文文本检测 (ai_detector.py)
    - 中文文本检测 (zh_detector_enhanced.py)
    - 文本质量分析 (text_analyzer.py)
    - 困惑度代理 (enhancements.py)
    """

    def __init__(self, lang: str = 'auto'):
        self.lang = lang
        self.adversarial_rewriter = None
        self.perplexity_surrogate = None

    def detect(self, text: str, lang: Optional[str] = None) -> Tuple[int, Optional[Dict]]:
        """
        执行完整的AI检测

        Args:
            text: 待检测文本
            lang: 语言 ('en', 'zh', or 'auto')

        Returns:
            Tuple of (ai_score, details_dict)
        """
        target_lang = lang if lang and lang != 'auto' else self._detect_language(text)

        clean_text = strip_latex(text)
        clean_len = len(clean_text.strip())

        if clean_len < 10:
            return 0, {'error': 'Text too short for analysis'}

        if target_lang == 'zh':
            return self._detect_chinese(clean_text)
        else:
            return self._detect_english(clean_text)

    def _detect_language(self, text: str) -> str:
        """自动检测语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        total_chars = len(text.strip())

        if total_chars == 0:
            return 'en'

        chinese_ratio = chinese_chars / total_chars

        if chinese_ratio > 0.3:
            return 'zh'
        return 'en'

    def _detect_chinese(self, text: str) -> Tuple[int, Optional[Dict]]:
        """检测中文文本"""
        ai_score, details = analyze_chinese_text(text, use_advanced=True)

        self.perplexity_surrogate = PerplexitySurrogate(lang='zh')
        ppl_metrics = self.perplexity_surrogate.calculate_surrogate_perplexity(text)

        if 'nlp_metrics' not in details:
            details['nlp_metrics'] = {}
        details['nlp_metrics']['perplexity_surrogate'] = ppl_metrics

        return ai_score, details

    def _detect_english(self, text: str) -> Tuple[int, Optional[Dict]]:
        """检测英文文本"""
        try:
            detector = AIDetector(text)
            result = detector.analyze()
            score = int(result['overall_score'] * 100)

            self.perplexity_surrogate = PerplexitySurrogate(lang='en')
            ppl_metrics = self.perplexity_surrogate.calculate_surrogate_perplexity(text)
            result['nlp_metrics'] = {'perplexity_surrogate': ppl_metrics}

            return min(100, max(0, score)), result
        except Exception as e:
            return 50, {'error': str(e)}

    def analyze_quality(self, text: str, lang: Optional[str] = None) -> Dict:
        """分析文本质量指标"""
        target_lang = lang if lang and lang != 'auto' else self._detect_language(text)

        if target_lang == 'zh':
            return {'note': 'Chinese quality analysis not implemented yet'}
        else:
            try:
                analyzer = TextAnalyzer(text)
                return analyzer.analyze()
            except Exception as e:
                return {'error': str(e)}

    def apply_adversarial_rules(self, text: str, metrics: Optional[Dict] = None) -> Tuple[str, list]:
        """
        应用对抗性改写规则

        这可以帮助将AI文本改写得更像人类写作

        Returns:
            Tuple of (rewritten_text, list_of_changes)
        """
        target_lang = self.lang if self.lang != 'auto' else self._detect_language(text)

        if self.adversarial_rewriter is None or self.adversarial_rewriter.lang != target_lang:
            self.adversarial_rewriter = AdversarialRewriter(lang=target_lang)

        return self.adversarial_rewriter.apply_adversarial_rewrite(text)

    def generate_feedback(self, metrics: Dict, is_en: bool) -> str:
        """生成检测反馈"""
        if self.adversarial_rewriter is None:
            self.adversarial_rewriter = AdversarialRewriter(lang='zh' if not is_en else 'en')

        return self.adversarial_rewriter.generate_feedback_from_detection(metrics, is_en)

    def full_pipeline(self, text: str, lang: Optional[str] = None,
                     apply_humanization: bool = False) -> Dict:
        """
        完整管道: 检测 + 分析 + (可选)人类化

        Args:
            text: 待处理文本
            lang: 语言
            apply_humanization: 是否应用对抗性人类化规则

        Returns:
            Dict with detection results, quality metrics, and optionally humanized text
        """
        target_lang = lang if lang and lang != 'auto' else self._detect_language(text)

        ai_score, details = self.detect(text, target_lang)

        quality_metrics = self.analyze_quality(text, target_lang)

        result = {
            'ai_score': ai_score,
            'detected_language': target_lang,
            'details': details,
            'quality_metrics': quality_metrics
        }

        if apply_humanization:
            humanized, changes = self.apply_adversarial_rules(text, details)
            result['humanized_text'] = humanized
            result['humanization_changes'] = changes

        feedback = self.generate_feedback(
            details.get('metrics', {}) if details else {},
            is_en=(target_lang != 'zh')
        )
        result['feedback'] = feedback

        return result


def main():
    parser = argparse.ArgumentParser(
        description='Unified AI Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detection_pipeline.py input.txt
  python detection_pipeline.py input.txt --lang zh
  python detection_pipeline.py input.txt --humanize
  python detection_pipeline.py input.txt --json
        """
    )

    parser.add_argument('input_file', help='Text file to analyze')
    parser.add_argument('--lang', choices=['en', 'zh', 'auto'], default='auto',
                       help='Language of the text (default: auto)')
    parser.add_argument('--humanize', action='store_true',
                       help='Apply adversarial humanization rules')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--quality', action='store_true',
                       help='Include text quality metrics')

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    if not text.strip():
        print("Error: Input file is empty", file=sys.stderr)
        sys.exit(1)

    pipeline = DetectionPipeline(lang=args.lang)

    import json

    result = pipeline.full_pipeline(
        text,
        lang=args.lang if args.lang != 'auto' else None,
        apply_humanization=args.humanize
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("=" * 70)
        print(f"AI DETECTION RESULTS")
        print("=" * 70)
        print(f"Detected Language: {result['detected_language'].upper()}")
        print(f"AI Score: {result['ai_score']}/100")

        if result['ai_score'] > 70:
            print("Assessment: Strong AI patterns detected")
        elif result['ai_score'] > 45:
            print("Assessment: Moderate AI patterns detected")
        else:
            print("Assessment: Low AI patterns detected")

        print()
        print("Feedback:")
        print("-" * 70)
        print(result['feedback'])

        if args.humanize and 'humanized_text' in result:
            print()
            print("HUMANIZED TEXT:")
            print("-" * 70)
            print(result['humanized_text'])
            print()
            print("Changes applied:")
            for change in result.get('humanization_changes', []):
                print(f"  - {change}")


if __name__ == '__main__':
    main()
