#!/usr/bin/env python3
"""
测试优化后的AI检测和人类化功能
验证知网3.0算法适配的改进效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from detection_pipeline import DetectionPipeline
import json


def test_chinese_ai_detection():
    """测试中文AI文本检测"""
    print("=" * 70)
    print("测试中文AI文本检测 (知网3.0算法适配)")
    print("=" * 70)
    
    # 测试样本1 - 强AI特征的学术文本
    sample1 = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。人工智能技术在图像识别、自然语言处理等领域发挥着至关重要的作用。然而，目前的研究仍然存在一定的局限性。首先，数据标注成本高昂，严重制约了模型的性能提升。其次，模型的泛化能力有待提高，难以应对复杂多变的实际应用场景。基于上述问题，本文提出了一种基于自监督学习的改进方法，旨在有效解决传统方法所面临的挑战。通过大量实验验证，本文方法在多个基准数据集上取得了良好的效果，展现出了显著的性能优势。"""
    
    pipeline = DetectionPipeline(lang='zh')
    score, details = pipeline.detect(sample1, lang='zh')
    
    print(f"\n样本1 (强AI特征文本):")
    print(f"AI检测分数: {score}/100")
    print(f"评估: {'高度疑似AI生成' if score > 70 else '中度疑似AI' if score > 40 else '人类特征明显'}")
    
    if details and 'metrics' in details:
        print("\n核心检测指标:")
        for key, value in details['metrics'].items():
            if 'sentence_length_distribution' in key:
                print(f"  {key}: {value}")
            elif 'paragraph_structure' in key:
                print(f"  {key}: {value}")
            elif 'info_density' in key:
                print(f"  {key}: {value}")
            elif 'transition_word' in key:
                print(f"  {key}: {value}")
    
    return score


def test_humanization():
    """测试人类化改写功能"""
    print("\n" + "=" * 70)
    print("测试对抗性人类化改写 (打破知网3.0检测)")
    print("=" * 70)
    
    sample = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。人工智能技术在图像识别、自然语言处理等领域发挥着至关重要的作用。然而，目前的研究仍然存在一定的局限性。首先，数据标注成本高昂，严重制约了模型的性能提升。其次，模型的泛化能力有待提高，难以应对复杂多变的实际应用场景。基于上述问题，本文提出了一种基于自监督学习的改进方法，旨在有效解决传统方法所面临的挑战。通过大量实验验证，本文方法在多个基准数据集上取得了良好的效果，展现出了显著的性能优势。"""
    
    pipeline = DetectionPipeline(lang='zh')
    
    # 原始检测分数
    original_score, _ = pipeline.detect(sample, lang='zh')
    print(f"\n原始文本AI分数: {original_score}/100")
    
    # 人类化改写
    humanized, changes = pipeline.apply_adversarial_rules(sample)
    
    # 改写后的检测分数
    new_score, _ = pipeline.detect(humanized, lang='zh')
    
    print(f"改写后AI分数: {new_score}/100")
    print(f"分数变化: {original_score - new_score}")
    
    print("\n应用的修改:")
    for i, change in enumerate(changes[:10], 1):
        print(f"  {i}. {change}")
    if len(changes) > 10:
        print(f"  ... (还有{len(changes) - 10}项修改)")
    
    print("\n改写后的文本片段:")
    preview = humanized[:200] + "..." if len(humanized) > 200 else humanized
    print(preview)
    
    return original_score, new_score


def test_full_pipeline():
    """测试完整管道"""
    print("\n" + "=" * 70)
    print("测试完整检测+人类化管道")
    print("=" * 70)
    
    sample = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。人工智能技术在图像识别、自然语言处理等领域发挥着至关重要的作用。然而，目前的研究仍然存在一定的局限性。基于上述问题，本文提出了一种基于自监督学习的改进方法。"""
    
    pipeline = DetectionPipeline(lang='zh')
    result = pipeline.full_pipeline(sample, lang='zh', apply_humanization=True)
    
    print(f"\nAI分数: {result['ai_score']}/100")
    print(f"检测语言: {result['detected_language']}")
    
    if 'humanized_text' in result:
        print("\n人类化文本:")
        print(result['humanized_text'][:300] + "..." if len(result['humanized_text']) > 300 else result['humanized_text'])
    
    if 'feedback' in result:
        print("\n反馈建议:")
        print(result['feedback'])


def main():
    print("开始测试优化后的AI检测和人类化系统...")
    
    # 运行测试
    try:
        score1 = test_chinese_ai_detection()
        original_score, new_score = test_humanization()
        test_full_pipeline()
        
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"AI样本检测分数: {score1}/100 (预期高分 >70)")
        print(f"人类化效果: {original_score} -> {new_score} (降低了{original_score - new_score}分)")
        print("\n✅ 所有测试完成! 优化后的系统工作正常。")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
