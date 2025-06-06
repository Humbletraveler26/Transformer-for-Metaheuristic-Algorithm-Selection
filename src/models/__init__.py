"""
Machine learning models for metaheuristic algorithm selection.

This package provides implementations of various ML models including
baseline models (Random Forest, SVM, etc.) and the Transformer architecture
for intelligent algorithm selection.
"""

from .baseline_models import BaselineModelTrainer, ModelEvaluator
from .feature_preprocessor import FeaturePreprocessor
from .transformer_models import (
    AlgorithmSelectionTransformer,
    AlgorithmDataset,
    TransformerTrainer,
    AttentionVisualizer,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock
)

__all__ = [
    # Baseline models
    'BaselineModelTrainer',
    'ModelEvaluator', 
    'FeaturePreprocessor',
    
    # Transformer models
    'AlgorithmSelectionTransformer',
    'AlgorithmDataset',
    'TransformerTrainer',
    'AttentionVisualizer',
    'PositionalEncoding',
    'MultiHeadAttention',
    'TransformerBlock'
] 