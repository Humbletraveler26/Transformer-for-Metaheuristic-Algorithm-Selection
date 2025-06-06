"""
Transformer architecture for metaheuristic algorithm selection.

This module implements a Transformer-based model that can process sequential
algorithm performance data and make intelligent algorithm recommendations
using attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import math
from torch.utils.data import Dataset, DataLoader
import pickle


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for feature relationship learning."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Compute scaled dot-product attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        self.attention_weights = attention_weights
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer encoder block with attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights


class AlgorithmSelectionTransformer(nn.Module):
    """
    Transformer model for metaheuristic algorithm selection.
    
    This model processes problem characteristics and algorithm performance
    features to predict the best algorithm for a given optimization problem.
    """
    
    def __init__(self, 
                 input_dim: int = 16,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 n_algorithms: int = 4,
                 max_seq_len: int = 100,
                 dropout: float = 0.1):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward network dimension
            n_algorithms: Number of algorithms to choose from
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_algorithms = n_algorithms
        
        # Input embedding layers
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_algorithms)
        )
        
        # For binary classification (is_best)
        self.binary_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = []
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_attention: bool = False):
        """
        Forward pass through the Transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            if return_attention:
                attention_weights.append(attn_weights)
        
        # Global average pooling across sequence dimension
        pooled = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # Multi-class classification (algorithm selection)
        multi_class_logits = self.classifier(pooled)
        multi_class_probs = F.softmax(multi_class_logits, dim=-1)
        
        # Binary classification (is_best algorithm)
        binary_logits = self.binary_classifier(pooled)
        binary_probs = torch.sigmoid(binary_logits)
        
        results = {
            'multi_class_logits': multi_class_logits,
            'multi_class_probs': multi_class_probs,
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'features': pooled
        }
        
        if return_attention:
            results['attention_weights'] = attention_weights
            self.attention_weights = attention_weights
        
        return results


class AlgorithmDataset(Dataset):
    """Dataset class for algorithm selection data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 algorithm_labels: Optional[np.ndarray] = None,
                 seq_len: int = 1):
        """
        Initialize dataset.
        
        Args:
            features: Feature matrix [n_samples, n_features]
            targets: Binary targets [n_samples]
            algorithm_labels: Multi-class algorithm labels [n_samples]
            seq_len: Sequence length for transformer input
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.algorithm_labels = torch.LongTensor(algorithm_labels) if algorithm_labels is not None else None
        self.seq_len = seq_len
        
        # Reshape features for sequence modeling
        if seq_len > 1:
            # Repeat features across sequence dimension
            self.features = self.features.unsqueeze(1).repeat(1, seq_len, 1)
        else:
            # Add sequence dimension
            self.features = self.features.unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'binary_target': self.targets[idx]
        }
        
        if self.algorithm_labels is not None:
            item['algorithm_label'] = self.algorithm_labels[idx]
        
        return item


class TransformerTrainer:
    """Training manager for the Transformer model."""
    
    def __init__(self, model: AlgorithmSelectionTransformer, 
                 device: str = 'cpu', learning_rate: float = 1e-4):
        """
        Initialize trainer.
        
        Args:
            model: Transformer model instance
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.binary_criterion = nn.BCELoss()
        self.multi_criterion = nn.CrossEntropyLoss()
        
        self.training_history = {
            'train_loss': [],
            'train_binary_acc': [],
            'train_multi_acc': [],
            'val_loss': [],
            'val_binary_acc': [],
            'val_multi_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader, use_multi_class: bool = False):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        binary_correct = 0
        multi_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            binary_targets = batch['binary_target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            
            # Binary classification loss
            binary_loss = self.binary_criterion(
                outputs['binary_probs'].squeeze(),
                binary_targets
            )
            
            loss = binary_loss
            
            # Multi-class loss (if available)
            if use_multi_class and 'algorithm_label' in batch:
                algorithm_targets = batch['algorithm_label'].to(self.device)
                multi_loss = self.multi_criterion(
                    outputs['multi_class_logits'],
                    algorithm_targets
                )
                loss += multi_loss
                
                # Multi-class accuracy
                multi_pred = outputs['multi_class_probs'].argmax(dim=1)
                multi_correct += (multi_pred == algorithm_targets).sum().item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            binary_pred = (outputs['binary_probs'].squeeze() > 0.5).float()
            binary_correct += (binary_pred == binary_targets).sum().item()
            
            total_loss += loss.item()
            total_samples += len(features)
        
        avg_loss = total_loss / len(train_loader)
        binary_acc = binary_correct / total_samples
        multi_acc = multi_correct / total_samples if use_multi_class else 0
        
        return avg_loss, binary_acc, multi_acc
    
    def evaluate(self, val_loader: DataLoader, use_multi_class: bool = False):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        binary_correct = 0
        multi_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                binary_targets = batch['binary_target'].to(self.device)
                
                outputs = self.model(features)
                
                # Binary classification loss
                binary_loss = self.binary_criterion(
                    outputs['binary_probs'].squeeze(),
                    binary_targets
                )
                
                loss = binary_loss
                
                # Multi-class loss (if available)
                if use_multi_class and 'algorithm_label' in batch:
                    algorithm_targets = batch['algorithm_label'].to(self.device)
                    multi_loss = self.multi_criterion(
                        outputs['multi_class_logits'],
                        algorithm_targets
                    )
                    loss += multi_loss
                    
                    # Multi-class accuracy
                    multi_pred = outputs['multi_class_probs'].argmax(dim=1)
                    multi_correct += (multi_pred == algorithm_targets).sum().item()
                
                # Calculate accuracy
                binary_pred = (outputs['binary_probs'].squeeze() > 0.5).float()
                binary_correct += (binary_pred == binary_targets).sum().item()
                
                total_loss += loss.item()
                total_samples += len(features)
        
        avg_loss = total_loss / len(val_loader)
        binary_acc = binary_correct / total_samples
        multi_acc = multi_correct / total_samples if use_multi_class else 0
        
        return avg_loss, binary_acc, multi_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 100, use_multi_class: bool = False, verbose: bool = True):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            use_multi_class: Whether to use multi-class classification
            verbose: Whether to print training progress
        """
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_binary_acc, train_multi_acc = self.train_epoch(
                train_loader, use_multi_class
            )
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_binary_acc'].append(train_binary_acc)
            self.training_history['train_multi_acc'].append(train_multi_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_binary_acc, val_multi_acc = self.evaluate(
                    val_loader, use_multi_class
                )
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_binary_acc'].append(val_binary_acc)
                self.training_history['val_multi_acc'].append(val_multi_acc)
                
                # Save best model
                if val_binary_acc > best_val_acc:
                    best_val_acc = val_binary_acc
                    self.save_model('models/transformer/best_model.pth')
            
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{epochs}:")
                    print(f"  Train Loss: {train_loss:.4f}, Binary Acc: {train_binary_acc:.4f}")
                    print(f"  Val Loss: {val_loss:.4f}, Binary Acc: {val_binary_acc:.4f}")
                    if use_multi_class:
                        print(f"  Train Multi Acc: {train_multi_acc:.4f}, Val Multi Acc: {val_multi_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}: Loss: {train_loss:.4f}, Acc: {train_binary_acc:.4f}")
    
    def save_model(self, path: str):
        """Save model state."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']


class AttentionVisualizer:
    """Utility class for visualizing attention weights."""
    
    def __init__(self, model: AlgorithmSelectionTransformer):
        self.model = model
    
    def get_attention_weights(self, features: torch.Tensor, feature_names: List[str] = None):
        """Extract attention weights for interpretability."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(features.unsqueeze(0), return_attention=True)
            attention_weights = outputs['attention_weights']
        
        # Process attention weights for visualization
        attention_data = []
        for layer_idx, layer_attention in enumerate(attention_weights):
            # Average across heads and batch
            avg_attention = layer_attention.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
            attention_data.append({
                'layer': layer_idx,
                'attention_matrix': avg_attention.cpu().numpy(),
                'feature_names': feature_names
            })
        
        return attention_data
    
    def plot_attention_heatmap(self, attention_data: List[Dict], save_path: str = None):
        """Plot attention heatmap for interpretation."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_layers = len(attention_data)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, layer_data in enumerate(attention_data):
            if i >= 6:  # Limit to 6 layers for visualization
                break
                
            attention_matrix = layer_data['attention_matrix']
            
            sns.heatmap(
                attention_matrix,
                ax=axes[i],
                cmap='Blues',
                cbar=True,
                square=True
            )
            axes[i].set_title(f'Layer {layer_data["layer"]} Attention')
            
            if layer_data['feature_names']:
                axes[i].set_xticklabels(layer_data['feature_names'], rotation=45)
                axes[i].set_yticklabels(layer_data['feature_names'], rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 