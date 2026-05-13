"""
AI检测模块 - 包含各类AI文本检测器
"""

from .base_detector import BaseDetector
from .english_detector import EnglishDetector
from .chinese_detector import ChineseDetector
from .vip_detector import VIPDetector

__all__ = [
    'BaseDetector',
    'EnglishDetector',
    'ChineseDetector', 
    'VIPDetector'
]
