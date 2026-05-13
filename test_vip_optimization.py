#!/usr/bin/env python3
"""
测试维普AI检测和人类化优化
验证针对维普平台的定向优化效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from vip_detector import VIPDetector
from vip_humanizer import VIPHumanizer
from detection_pipeline import DetectionPipeline


def test_vip_detection():
    """测试维普AI检测"""
    print("=" * 70)
    print("测试维普AI检测功能")
    print("=" * 70)
    
    # 测试样本 - 包含维普关注的AI特征
    sample = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。
    
随着大数据时代的到来，数据分析变得尤为重要。首先，数据挖掘技术可以帮助企业发现潜在价值。其次，机器学习算法可以提高预测准确性。再次，可视化技术使数据更加直观。最后，总而言之，大数据分析对企业决策具有重要意义。

通过采用先进的技术方案，可以实现数字化转型。首先，智能化系统可以提高工作效率。其次，自动化流程可以降低运营成本。再次，数据驱动决策可以提升竞争力。此外，持续创新可以保持竞争优势。"""

    detector = VIPDetector()
    score, details = detector.detect(sample)
    
    print(f"\n维普AI检测分数: {score}/100")
    print(f"评估: {'高度疑似AI生成' if score > 70 else '中度疑似AI' if score > 40 else '人类特征明显'}")
    
    if details and 'metrics' in details:
        print("\n核心检测指标:")
        for key, value in details['metrics'].items():
            print(f"  {key}: {value.get('details', value)}")
    
    return score


def test_vip_humanization():
    """测试维普人类化改写"""
    print("\n" + "=" * 70)
    print("测试维普人类化改写功能")
    print("=" * 70)
    
    sample = """随着人工智能技术的发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。"""

    humanizer = VIPHumanizer()
    humanized, changes = humanizer.humanize(sample)
    
    print(f"\n原始文本:")
    print(sample[:200] + "..." if len(sample) > 200 else sample)
    
    print(f"\n人类化改写:")
    print(humanized[:200] + "..." if len(humanized) > 200 else humanized)
    
    print(f"\n应用的修改 ({len(changes)}项):")
    for i, change in enumerate(changes[:10], 1):
        print(f"  {i}. {change}")
    if len(changes) > 10:
        print(f"  ... (还有{len(changes) - 10}项修改)")
    
    # 检测改写后的效果
    detector = VIPDetector()
    new_score, _ = detector.detect(humanized)
    print(f"\n改写后维普检测分数: {new_score}/100")
    print(f"分数变化: {100 - new_score} (相比基准)")
    
    return new_score


def test_detection_pipeline_integration():
    """测试检测管道的维普集成"""
    print("\n" + "=" * 70)
    print("测试检测管道维普集成")
    print("=" * 70)
    
    sample = """随着科技的进步，人工智能技术得到了快速发展。首先，人工智能在医疗领域具有重要意义。其次，在教育领域也发挥着重要作用。再次，在金融领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。"""

    pipeline = DetectionPipeline()
    
    # 维普检测
    vip_score, vip_details = pipeline.detect_for_vip(sample)
    print(f"\n维普检测分数: {vip_score}/100")
    
    # 维普人类化
    humanized, changes = pipeline.humanize_for_vip(sample)
    print(f"人类化改写完成: {len(changes)}项修改")
    
    # 再次检测
    new_score, _ = pipeline.detect_for_vip(humanized)
    print(f"改写后维普检测分数: {new_score}/100")
    print(f"分数下降: {vip_score - new_score}分")


def test_full_optimization():
    """完整优化流程测试"""
    print("\n" + "=" * 70)
    print("完整优化流程测试")
    print("=" * 70)
    
    # 典型AI文本
    ai_text = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。
    
基于大数据分析的背景，数据挖掘技术可以帮助企业发现潜在价值。首先，数据驱动决策可以提高效率。其次，智能化系统可以降低成本。再次，可视化技术使数据更加直观。值得注意的是，这些技术在实际应用中具有重要的现实意义。"""

    print("\n原始文本特征分析:")
    detector = VIPDetector()
    original_score, details = detector.detect(ai_text)
    print(f"原始维普检测分数: {original_score}/100")
    
    # 应用人类化
    humanizer = VIPHumanizer()
    humanized, changes = humanizer.humanize(ai_text)
    
    print(f"\n人类化改写:")
    print(f"修改项数: {len(changes)}")
    
    # 检测改写效果
    new_score, _ = detector.detect(humanized)
    print(f"改写后维普检测分数: {new_score}/100")
    print(f"分数变化: {original_score - new_score}")
    
    print("\n效果评估:")
    if original_score - new_score >= 20:
        print("✅ 优化效果显著")
    elif original_score - new_score >= 10:
        print("⚠️ 优化效果一般")
    else:
        print("❌ 优化效果不明显")
    
    return original_score, new_score


def main():
    print("开始测试维普AI检测和人类化优化...")
    print("=" * 70)
    
    try:
        # 运行测试
        vip_score = test_vip_detection()
        new_score = test_vip_humanization()
        test_detection_pipeline_integration()
        original, optimized = test_full_optimization()
        
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"维普AI检测样本分数: {vip_score}/100")
        print(f"人类化改写效果: {vip_score} -> {new_score} (降低{vip_score - new_score}分)")
        print(f"完整优化效果: {original} -> {optimized} (降低{original - optimized}分)")
        print("\n✅ 维普优化测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
