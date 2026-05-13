#!/usr/bin/env python3
"""
测试重构后的AI检测和人类化系统
验证新的模块化架构是否正常工作
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from detection_pipeline import DetectionPipeline


def test_general_detection():
    """测试通用检测功能"""
    print("=" * 70)
    print("测试通用AI检测功能")
    print("=" * 70)
    
    # 中文AI文本样本
    sample_text = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。

基于大数据分析的背景，数据挖掘技术可以帮助企业发现潜在价值。首先，数据驱动决策可以提高效率。其次，智能化系统可以降低成本。再次，可视化技术使数据更加直观。值得注意的是，这些技术在实际应用中具有重要的现实意义。
"""
    
    pipeline = DetectionPipeline(lang='zh')
    
    # 检测
    score, details = pipeline.detect(sample_text, lang='zh')
    
    print(f"检测分数: {score}/100")
    print(f"评估: {'高度疑似AI生成' if score > 70 else '中度疑似AI' if score > 40 else '人类特征明显'}")
    
    if details and 'metrics' in details:
        print("\n检测指标:")
        for key, value in details['metrics'].items():
            print(f"  - {key}: {value.get('details', value)[:50]}...")
    
    return score


def test_vip_detection():
    """测试维普检测功能"""
    print("\n" + "=" * 70)
    print("测试维普平台AI检测功能")
    print("=" * 70)
    
    sample_text = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。
"""
    
    pipeline = DetectionPipeline()
    score, details = pipeline.detect_for_vip(sample_text)
    
    print(f"维普检测分数: {score}/100")
    
    if details and 'metrics' in details:
        print("\n维普检测指标:")
        for key, value in details['metrics'].items():
            print(f"  - {key}: {value.get('details', value)}")
    
    return score


def test_humanization():
    """测试人类化功能"""
    print("\n" + "=" * 70)
    print("测试人类化功能")
    print("=" * 70)
    
    sample_text = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。综上所述，人工智能技术的发展具有重要意义。
"""
    
    pipeline = DetectionPipeline()
    
    # 通用人类化
    print("\n通用人类化:")
    humanized, changes = pipeline.humanize(sample_text, platform='general')
    print(f"应用了 {len(changes)} 个改动")
    
    # 维普人类化
    print("\n维普人类化:")
    vip_humanized, vip_changes = pipeline.humanize_for_vip(sample_text)
    print(f"应用了 {len(vip_changes)} 个改动")
    
    return len(changes), len(vip_changes)


def test_full_pipeline():
    """测试完整管道功能"""
    print("\n" + "=" * 70)
    print("测试完整检测+人类化管道")
    print("=" * 70)
    
    sample_text = """随着人工智能技术的飞速发展，深度学习在各个领域得到了广泛的应用。首先，人工智能技术在图像识别领域发挥着重要作用。其次，在自然语言处理领域也取得了显著进展。再次，在计算机视觉领域应用广泛。最后，综上所述，人工智能技术的发展具有重要意义。
"""
    
    pipeline = DetectionPipeline(lang='zh')
    
    # 通用管道
    print("\n通用管道:")
    result = pipeline.full_pipeline(sample_text, lang='zh', apply_humanization=True)
    print(f"检测分数: {result['ai_score']}/100")
    print(f"人类化改动数: {len(result.get('humanization_changes', []))}")
    
    # 维普管道
    print("\n维普管道:")
    vip_result = pipeline.full_pipeline(sample_text, lang='zh', apply_humanization=True, platform='vip')
    print(f"维普检测分数: {vip_result['ai_score']}/100")
    print(f"维普人类化改动数: {len(vip_result.get('humanization_changes', []))}")
    
    return result['ai_score'], vip_result['ai_score']


def main():
    print("开始测试重构后的AI检测和人类化系统...")
    
    try:
        # 测试各个功能
        general_score = test_general_detection()
        vip_score = test_vip_detection()
        general_changes, vip_changes = test_humanization()
        pipeline_score, pipeline_vip_score = test_full_pipeline()
        
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"通用检测分数: {general_score}/100")
        print(f"维普检测分数: {vip_score}/100")
        print(f"通用人类化改动数: {general_changes}")
        print(f"维普人类化改动数: {vip_changes}")
        print(f"完整管道通用分数: {pipeline_score}/100")
        print(f"完整管道维普分数: {pipeline_vip_score}/100")
        print("\n✅ 重构后的代码测试成功!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
