"""
Baseline machine learning models for metaheuristic algorithm selection.

This module implements various baseline ML models and provides training,
evaluation, and comparison capabilities.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns


class BaselineModelTrainer:
    """
    Trains and manages baseline ML models for algorithm selection.
    
    Supports multiple algorithms with hyperparameter tuning and evaluation.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the baseline model trainer.
        
        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.fitted_models = {}
        self.model_scores = {}
        
        # Initialize baseline models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize baseline ML models with default parameters."""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            
            'svm_rbf': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'svm_linear': SVC(
                kernel='linear',
                C=1.0,
                probability=True,
                random_state=self.random_state
            ),
            
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                alpha=0.0001,
                random_state=self.random_state
            ),
            
            'naive_bayes': GaussianNB(),
            
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=self.n_jobs
            )
        }
    
    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Train all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training results
        """
        training_results = {}
        
        if verbose:
            print(f"Training {len(self.models)} baseline models...")
            print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print("-" * 60)
        
        for model_name, model in self.models.items():
            if verbose:
                print(f"Training {model_name}...")
            
            start_time = time.time()
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Store fitted model
                self.fitted_models[model_name] = model
                
                training_time = time.time() - start_time
                
                # Calculate training accuracy
                train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)
                
                training_results[model_name] = {
                    'training_time': training_time,
                    'train_accuracy': train_accuracy,
                    'status': 'success'
                }
                
                if verbose:
                    print(f"  ✓ {model_name}: {training_time:.2f}s, Train Acc: {train_accuracy:.4f}")
                
            except Exception as e:
                training_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if verbose:
                    print(f"  ✗ {model_name}: Failed - {e}")
        
        if verbose:
            successful_models = sum(1 for r in training_results.values() if r['status'] == 'success')
            print(f"\nTraining complete: {successful_models}/{len(self.models)} models trained successfully")
        
        return training_results
    
    def evaluate_models(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       X_train: np.ndarray = None,
                       y_train: np.ndarray = None,
                       cv_folds: int = 5,
                       verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            X_train: Training features (for cross-validation)
            y_train: Training targets (for cross-validation)
            cv_folds: Number of cross-validation folds
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_results = {}
        
        if verbose:
            print(f"\nEvaluating {len(self.fitted_models)} trained models...")
            print(f"Test data: {X_test.shape[0]} samples")
            print("-" * 60)
        
        for model_name, model in self.fitted_models.items():
            if verbose:
                print(f"Evaluating {model_name}...")
            
            try:
                # Test set predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Cross-validation scores (if training data provided)
                cv_scores = None
                if X_train is not None and y_train is not None:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                
                # ROC AUC (for binary classification)
                roc_auc = None
                if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                evaluation_results[model_name] = {
                    'test_accuracy': test_accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean() if cv_scores is not None else None,
                    'cv_std': cv_scores.std() if cv_scores is not None else None,
                    'status': 'success'
                }
                
                if verbose:
                    cv_info = f", CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}" if cv_scores is not None else ""
                    print(f"  ✓ {model_name}: Acc: {test_accuracy:.4f}, F1: {f1:.4f}{cv_info}")
                
            except Exception as e:
                evaluation_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if verbose:
                    print(f"  ✗ {model_name}: Failed - {e}")
        
        self.model_scores = evaluation_results
        
        if verbose:
            self._print_model_ranking()
        
        return evaluation_results
    
    def _print_model_ranking(self):
        """Print model ranking based on test accuracy."""
        print(f"\nModel Ranking (by Test Accuracy):")
        print("-" * 50)
        
        # Filter successful models and sort by accuracy
        successful_models = {
            name: scores for name, scores in self.model_scores.items()
            if scores['status'] == 'success'
        }
        
        sorted_models = sorted(
            successful_models.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            accuracy = scores['test_accuracy']
            f1 = scores['f1_score']
            cv_info = f" (CV: {scores['cv_mean']:.4f}±{scores['cv_std']:.4f})" if scores['cv_mean'] else ""
            print(f"{i:2d}. {model_name:<20} | Acc: {accuracy:.4f} | F1: {f1:.4f}{cv_info}")
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            models_to_tune: List[str] = None,
                            cv_folds: int = 3,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for selected models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_tune: List of model names to tune (None for top 3)
            cv_folds: Number of CV folds for tuning
            verbose: Whether to print progress
            
        Returns:
            Dictionary with tuning results
        """
        if models_to_tune is None:
            # Select top 3 models based on initial performance
            if self.model_scores:
                successful_models = {
                    name: scores for name, scores in self.model_scores.items()
                    if scores['status'] == 'success'
                }
                sorted_models = sorted(
                    successful_models.items(),
                    key=lambda x: x[1]['test_accuracy'],
                    reverse=True
                )
                models_to_tune = [name for name, _ in sorted_models[:3]]
            else:
                models_to_tune = ['random_forest', 'gradient_boosting', 'svm_rbf']
        
        if verbose:
            print(f"\nHyperparameter tuning for {len(models_to_tune)} models...")
            print("-" * 60)
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'svm_rbf': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        tuning_results = {}
        
        for model_name in models_to_tune:
            if model_name not in param_grids:
                continue
            
            if verbose:
                print(f"Tuning {model_name}...")
            
            try:
                # Get base model
                base_model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=self.n_jobs,
                    verbose=0
                )
                
                start_time = time.time()
                grid_search.fit(X_train, y_train)
                tuning_time = time.time() - start_time
                
                # Update model with best parameters
                self.fitted_models[f"{model_name}_tuned"] = grid_search.best_estimator_
                
                tuning_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'tuning_time': tuning_time,
                    'status': 'success'
                }
                
                if verbose:
                    print(f"  ✓ {model_name}: Score: {grid_search.best_score_:.4f}, Time: {tuning_time:.1f}s")
                    print(f"    Best params: {grid_search.best_params_}")
                
            except Exception as e:
                tuning_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if verbose:
                    print(f"  ✗ {model_name}: Failed - {e}")
        
        return tuning_results
    
    def get_feature_importance(self, model_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_names: List of model names to analyze
            
        Returns:
            Dictionary with feature importances
        """
        if model_names is None:
            model_names = list(self.fitted_models.keys())
        
        importance_dict = {}
        
        for model_name in model_names:
            if model_name in self.fitted_models:
                model = self.fitted_models[model_name]
                
                if hasattr(model, 'feature_importances_'):
                    importance_dict[model_name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importance_dict[model_name] = np.abs(model.coef_[0])
        
        return importance_dict
    
    def save_models(self, save_dir: str = "models/baseline"):
        """Save trained models to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.fitted_models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save model scores
        scores_path = os.path.join(save_dir, "model_scores.pkl")
        with open(scores_path, 'wb') as f:
            pickle.dump(self.model_scores, f)
        
        print(f"Saved {len(self.fitted_models)} models to {save_dir}")
    
    def load_models(self, save_dir: str = "models/baseline"):
        """Load trained models from disk."""
        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl') and filename != 'model_scores.pkl':
                model_name = filename[:-4]  # Remove .pkl extension
                model_path = os.path.join(save_dir, filename)
                
                with open(model_path, 'rb') as f:
                    self.fitted_models[model_name] = pickle.load(f)
        
        # Load model scores if available
        scores_path = os.path.join(save_dir, "model_scores.pkl")
        if os.path.exists(scores_path):
            with open(scores_path, 'rb') as f:
                self.model_scores = pickle.load(f)
        
        print(f"Loaded {len(self.fitted_models)} models from {save_dir}")


class ModelEvaluator:
    """
    Comprehensive evaluation and analysis of ML models.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize the model evaluator.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
    
    def create_comparison_report(self, 
                               model_scores: Dict[str, Dict[str, float]],
                               feature_names: List[str] = None,
                               model_trainer: BaselineModelTrainer = None) -> pd.DataFrame:
        """
        Create a comprehensive comparison report.
        
        Args:
            model_scores: Dictionary with model evaluation results
            feature_names: List of feature names
            model_trainer: Trained model trainer for feature importance
            
        Returns:
            DataFrame with comparison results
        """
        # Filter successful models
        successful_models = {
            name: scores for name, scores in model_scores.items()
            if scores.get('status') == 'success'
        }
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, scores in successful_models.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': scores.get('test_accuracy', 0),
                'Precision': scores.get('precision', 0),
                'Recall': scores.get('recall', 0),
                'F1 Score': scores.get('f1_score', 0),
                'ROC AUC': scores.get('roc_auc', None),
                'CV Mean': scores.get('cv_mean', None),
                'CV Std': scores.get('cv_std', None)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test Accuracy', ascending=False)
        
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        print(comparison_df.round(4).to_string(index=False))
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
        
        # Feature importance analysis
        if model_trainer and feature_names:
            self._analyze_feature_importance(model_trainer, feature_names)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Plot model comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].barh(comparison_df['Model'], comparison_df['Test Accuracy'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_xlabel('Accuracy')
        
        # F1 Score comparison
        axes[0, 1].barh(comparison_df['Model'], comparison_df['F1 Score'])
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xlabel('F1 Score')
        
        # Precision vs Recall
        axes[1, 0].scatter(comparison_df['Recall'], comparison_df['Precision'])
        for i, model in enumerate(comparison_df['Model']):
            axes[1, 0].annotate(model, (comparison_df['Recall'].iloc[i], comparison_df['Precision'].iloc[i]))
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        
        # Cross-validation results
        cv_data = comparison_df[comparison_df['CV Mean'].notna()]
        if len(cv_data) > 0:
            axes[1, 1].errorbar(range(len(cv_data)), cv_data['CV Mean'], yerr=cv_data['CV Std'], 
                               marker='o', capsize=5)
            axes[1, 1].set_xticks(range(len(cv_data)))
            axes[1, 1].set_xticklabels(cv_data['Model'], rotation=45)
            axes[1, 1].set_title('Cross-Validation Results')
            axes[1, 1].set_ylabel('CV Accuracy')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _analyze_feature_importance(self, model_trainer: BaselineModelTrainer, feature_names: List[str]):
        """Analyze and plot feature importance."""
        importance_dict = model_trainer.get_feature_importance()
        
        if not importance_dict:
            return
        
        # Plot feature importance for tree-based models
        fig, axes = plt.subplots(len(importance_dict), 1, figsize=(12, 6 * len(importance_dict)))
        if len(importance_dict) == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(importance_dict.items()):
            # Get top 15 features
            top_indices = np.argsort(importance)[-15:]
            top_importance = importance[top_indices]
            top_features = [feature_names[j] if j < len(feature_names) else f"Feature_{j}" for j in top_indices]
            
            axes[i].barh(range(len(top_features)), top_importance)
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features)
            axes[i].set_title(f'Feature Importance - {model_name}')
            axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        
        plt.show() 