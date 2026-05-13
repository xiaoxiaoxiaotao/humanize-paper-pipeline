#!/usr/bin/env python3
"""
测试脚本 - 验证AI检测管道有效性

运行方式:
    python test_samples/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.detection_pipeline import DetectionPipeline
from scripts.enhancements import AdversarialRewriter


def load_samples():
    samples = {
        'zh_ai_1': """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。人工智能技术在图像识别，自然语言处理等领域发挥着至关重要的作用。然而，目前的研究仍然存在一定的局限性。首先，数据标注成本高昂，严重制约了模型的性能提升。其次，模型的泛化能力有待提高，难以应对复杂多变的实际应用场景。基于上述问题，本文提出了一种基于自监督学习的改进方法，旨在有效解决传统方法所面临的挑战。通过大量实验验证，本文方法在多个基准数据集上取得了良好的效果，展现出了显著的性能优势。""",

        'zh_ai_2': """随着深度学习技术的不断进步，计算机视觉领域取得了长足的发展。神经网络模型在图像分类、目标检测等任务中发挥着重要作用。建筑行业作为国民经济的重要引擎，其信息化水平直接影响到工程质量和安全性能。传统的施工管理模式存在诸多不足，智能化转型已成为行业发展的必然趋势。本文旨在研究基于深度学习的建筑施工安全智能监测系统，通过计算机视觉技术实现对施工安全隐患的自动识别。值得关注的是，该研究具有重要的现实意义，为建筑行业的数字化转型提供了理论支撑。""",

        'zh_human': """深度学习火了之后，到处都在用它做项目。我之前做过一个图像分类的毕设，用的ResNet，效果还行。但数据不够是个问题，后来试了试数据增强，缓解了不少。说白了，神经网络就是拟合数据分布的东西，太简单反而欠拟合，太复杂又过拟合。调参是个经验活，没啥捷径。""",

        'en_ai_1': """In the ever-evolving landscape of artificial intelligence, deep learning has emerged as a pivotal technology that plays a crucial role in various domains. Moreover, the advancement of neural networks has significantly transformed the way we approach complex problems. It is important to note that the traditional machine learning methods face numerous challenges in terms of scalability and generalization. Furthermore, the utilization of large-scale datasets has become indispensable for achieving state-of-the-art performance. In addition to these considerations, researchers have been exploring novel architectures to enhance model capabilities. The present study aims to address these challenges by proposing an innovative approach that leverages the power of self-supervised learning techniques.""",

        'en_ai_2': """The rapid adoption of electric vehicles is a testament to innovative engineering. In today's rapidly evolving business environment, companies must leverage cutting-edge technologies to maintain competitive advantage. Moreover, the seamless integration of renewable energy sources into existing infrastructure presents both challenges and opportunities. It should be noted that robust data analytics capabilities can foster sustainable growth. Furthermore, organizations that harness the power of artificial intelligence will be better positioned to capitalize on emerging market trends. In conclusion, this paradigm shift towards digital transformation represents a game-changer for industries across the globe.""",

        'en_human': """I tried using ChatGPT to help me write this report. Honestly? It made things weird. The text sounded too formal, like it was trying too hard to sound smart. My advisor caught it immediately. I've since learned that AI-generated content often sounds stilted - it uses words like "leverage" when "use" would be fine, or adds hedging phrases like "it is important to note that" when nobody actually talks like that. The best writing comes from actual thinking, not from pattern matching.""",
    }
    return samples


def test_detection():
    print("=" * 70)
    print("AI DETECTION PIPELINE TEST")
    print("=" * 70)
    print()

    samples = load_samples()
    pipeline = DetectionPipeline()

    results = {}

    for name, text in samples.items():
        print(f"Testing: {name}")
        print("-" * 50)

        lang = 'zh' if name.startswith('zh') else 'en'

        score, details = pipeline.detect(text, lang=lang)

        results[name] = {
            'score': score,
            'details': details,
            'expected': 'ai' if '_ai_' in name else 'human'
        }

        print(f"  Detected Language: {lang.upper()}")
        print(f"  AI Score: {score}/100")

        if details and 'metrics' in details:
            metrics = details['metrics']
            print(f"  Key Metrics:")
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, dict):
                    print(f"    - {key}: {value.get('score', 'N/A')}")
                else:
                    print(f"    - {key}: {value}")

        expected = results[name]['expected']
        if expected == 'ai' and score >= 45:
            print(f"  ✓ PASS (Expected AI, got {score})")
        elif expected == 'human' and score < 45:
            print(f"  ✓ PASS (Expected Human, got {score})")
        elif expected == 'ai':
            print(f"  ~ MARGINAL (Expected AI, got {score} - may need tuning)")
        else:
            print(f"  ✗ FAIL (Expected Human, got {score})")

        print()
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ai_correct = sum(1 for r in results.values() if r['expected'] == 'ai' and r['score'] >= 45)
    human_correct = sum(1 for r in results.values() if r['expected'] == 'human' and r['score'] < 45)

    total_ai = sum(1 for r in results.values() if r['expected'] == 'ai')
    total_human = sum(1 for r in results.values() if r['expected'] == 'human')

    print(f"AI Samples Correctly Identified: {ai_correct}/{total_ai}")
    print(f"Human Samples Correctly Identified: {human_correct}/{total_human}")
    print(f"Overall Accuracy: {(ai_correct + human_correct)}/{len(results)}")


def test_adversarial_rewriter():
    print()
    print("=" * 70)
    print("ADVERSARIAL REWRITER TEST")
    print("=" * 70)
    print()

    test_text_zh = """随着深度学习技术的不断发展，人工智能在各领域发挥着至关重要的作用。本文旨在研究基于神经网络的智能识别系统。值得注意的是，该研究具有重要的现实意义。"""

    test_text_en = """In the ever-evolving landscape of technology, deep learning plays a crucial role. Moreover, it is important to note that neural networks serve as a pivotal technology. The present study aims to address these challenges."""

    print("Chinese Adversarial Rewrite Test:")
    print("-" * 50)
    print(f"Original:\n{test_text_zh}")
    print()

    rewriter_zh = AdversarialRewriter(lang='zh')
    rewritten_zh, changes_zh = rewriter_zh.apply_adversarial_rewrite(test_text_zh)

    print(f"Rewritten:\n{rewritten_zh}")
    print()
    print(f"Changes ({len(changes_zh)}):")
    for change in changes_zh[:5]:
        print(f"  - {change}")
    print()

    print("English Adversarial Rewrite Test:")
    print("-" * 50)
    print(f"Original:\n{test_text_en}")
    print()

    rewriter_en = AdversarialRewriter(lang='en')
    rewritten_en, changes_en = rewriter_en.apply_adversarial_rewrite(test_text_en)

    print(f"Rewritten:\n{rewritten_en}")
    print()
    print(f"Changes ({len(changes_en)}):")
    for change in changes_en[:5]:
        print(f"  - {change}")


def test_perplexity_surrogate():
    print()
    print("=" * 70)
    print("PERPLEXITY SURROGATE TEST")
    print("=" * 70)
    print()

    from scripts.enhancements import PerplexitySurrogate

    samples = [
        ("zh_ai", """随着人工智能技术的不断发展，深度学习在各领域发挥着重要作用。本文旨在研究相关技术。"""),
        ("zh_human", """深度学习这玩意儿挺有意思。我用它跑过不少项目，效果还行。"""),
        ("en_ai", """In the ever-evolving landscape of technology, deep learning plays a crucial role. Moreover, it is important to note that neural networks serve as a pivotal technology."""),
        ("en_human", """I tried using deep learning for this project. It worked pretty well, honestly. The key is getting enough data.""")
    ]

    for name, text in samples:
        lang = 'zh' if name.startswith('zh') else 'en'
        surrogate = PerplexitySurrogate(lang=lang)
        metrics = surrogate.calculate_surrogate_perplexity(text)

        print(f"{name}:")
        print(f"  Surrogate PPL Score: {metrics['surrogate_ppl']}")
        print(f"  Interpretation: {metrics['interpretation']}")
        print()


if __name__ == '__main__':
    test_detection()
    test_adversarial_rewriter()
    test_perplexity_surrogate()
    print()
    print("=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
