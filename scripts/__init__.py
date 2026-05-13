#!/usr/bin/env python3
"""
AI Detection Pipeline - Refactored

This module provides a high-level interface for AI text detection.
All detection logic has been moved to the scripts/ directory.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection_pipeline import DetectionPipeline, main as pipeline_main

__all__ = ['DetectionPipeline']


def detect_ai_text(text: str, lang: str = 'auto') -> tuple:
    """
    Convenience function for AI text detection.

    Args:
        text: Text to analyze
        lang: Language ('en', 'zh', or 'auto')

    Returns:
        Tuple of (ai_score, details)
    """
    pipeline = DetectionPipeline(lang=lang)
    return pipeline.detect(text, lang=lang if lang != 'auto' else None)


def humanize_text(text: str, lang: str = 'auto', metrics: dict = None) -> tuple:
    """
    Apply adversarial humanization rules to reduce AI indicators.

    Args:
        text: Text to humanize
        lang: Language
        metrics: Optional detection metrics to guide humanization

    Returns:
        Tuple of (humanized_text, list_of_changes)
    """
    pipeline = DetectionPipeline(lang=lang)
    return pipeline.apply_adversarial_rules(text, metrics)


if __name__ == '__main__':
    pipeline_main()
