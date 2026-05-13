# humanize-paper-pipeline

AI文本检测与人类化工具，支持中英文多平台检测和优化。

## 功能特性

### AI检测功能

- **通用检测平台**：支持中英文文本的AI检测
- **中文检测**：知网3.0算法适配，包含10+个检测维度
- **维普平台**：维普AIGC检测算法适配，专门针对维普检测特征
- **轻量级**：所有检测都是启发式方法，无需GPU，适合2核CPU

### 人类化改写功能

- **通用人类化**：通用文本人类化规则
- **维普人类化**：针对维普平台的定向优化
- **认知特征注入**：模拟人类思考过程的表达
- **打破语义指纹**：去除AI典型的高频句式

## 项目结构

```
scripts/
├── detectors/              # 检测器模块
│   ├── __init__.py
│   ├── base_detector.py    # 检测器基类
│   ├── english_detector.py # 英文AI检测器
│   ├── chinese_detector.py # 中文AI检测器（知网3.0算法）
│   └── vip_detector.py     # 维普平台专用检测器
├── humanizers/             # 人类化器模块
│   ├── __init__.py
│   ├── base_humanizer.py   # 人类化器基类
│   ├── chinese_humanizer.py # 通用中文人类化器
│   └── vip_humanizer.py     # 维普平台专用人类化器
├── detection_pipeline.py    # 统一检测管道
├── enhancements.py         # 增强功能（包含PerplexitySurrogate）
├── text_analyzer.py        # 文本分析器
└── formatter.py            # 文本格式化工具
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

```bash
# 基本检测
python scripts/detection_pipeline.py input.txt

# 中文检测
python scripts/detection_pipeline.py input.txt --lang zh

# 维普平台检测
python scripts/detection_pipeline.py input.txt --platform vip

# 检测并人类化
python scripts/detection_pipeline.py input.txt --humanize

# JSON输出
python scripts/detection_pipeline.py input.txt --json
```

### 代码使用

#### 基本检测

```python
from scripts.detection_pipeline import DetectionPipeline

# 创建管道
pipeline = DetectionPipeline()

# 检测文本
text = "随着人工智能技术的飞速发展..."
score, details = pipeline.detect(text, lang='zh')
print(f"AI分数: {score}/100")
```

#### 维普平台检测

```python
from scripts.detection_pipeline import DetectionPipeline

pipeline = DetectionPipeline()

# 维普检测
vip_score, vip_details = pipeline.detect_for_vip(text)
print(f"维普分数: {vip_score}/100")
```

#### 人类化改写

```python
from scripts.detection_pipeline import DetectionPipeline

pipeline = DetectionPipeline()

# 通用人类化
humanized, changes = pipeline.humanize(text, platform='general')

# 维普人类化
vip_humanized, vip_changes = pipeline.humanize_for_vip(text)
```

#### 完整管道

```python
from scripts.detection_pipeline import DetectionPipeline

pipeline = DetectionPipeline()

# 完整流程：检测 + 人类化
result = pipeline.full_pipeline(
    text,
    lang='zh',
    apply_humanization=True,
    platform='vip'
)

print(result['ai_score'])
print(result['humanized_text'])
```

## 检测指标说明

### 中文检测指标（知网3.0算法）

1. **句长分布**：检测句长是否集中在15-25字区间
2. **段落结构相似性**：检测段落结构是否过于统一
3. **信息密度**：检测信息密度是否稳定在65-75%
4. **连接词分布**：检测连接词是否过度使用且分布均匀
5. **句式重复**：检测是否有重复的句式开头
6. **词汇多样性**：检测词汇多样性是否偏低
7. **抽象语言**：检测是否过多使用抽象术语
8. **词汇突发性**：检测词汇分布是否均匀
9. **模板句式**：检测"随着...的..."等模板句式

### 维普检测指标

1. **语义指纹**：检测AI高频句式
2. **机械模式**：检测"首先...其次...最后"等结构
3. **段落结构**：检测段落结构是否机械
4. **数据真实性**：检测是否有可疑的虚构数据

## Web服务

项目包含Flask Web服务，支持HTTP接口调用：

```bash
# 启动服务
python app.py
```

服务默认运行在 http://localhost:5000

## 支持的平台

- **通用平台**：适用于大多数AI检测场景
- **知网**：知网AIGC检测算法适配
- **维普**：维普AIGC检测算法适配

## 技术特点

1. **启发式检测**：不依赖深度学习模型，运行速度快
2. **轻量级**：适合2核CPU环境
3. **可扩展**：模块化架构，易于添加新的检测和人类化规则
4. **多语言**：支持中英文检测

## 许可证

本项目仅供学习和研究使用。
