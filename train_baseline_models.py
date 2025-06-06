#!/usr/bin/env python3
"""
Comprehensive baseline model training and evaluation.

This script trains and evaluates multiple baseline ML models for metaheuristic
algorithm selection using the preprocessed feature data.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.append('src')

from models import BaselineModelTrainer, ModelEvaluator, FeaturePreprocessor


def main():
    """Main function to train and evaluate baseline models."""
    
    print("TRANSFORMER METAHEURISTIC SELECTION - BASELINE MODEL TRAINING")
    print("=" * 75)
    print()
    
    # Find the latest feature data
    feature_files = [f for f in os.listdir('data/processed') if f.startswith('simple_features')]
    
    if not feature_files:
        print("❌ No feature data found. Please run feature extraction first.")
        return 1
    
    latest_feature_file = max(feature_files)
    feature_path = f'data/processed/{latest_feature_file}'
    
    print(f"Loading feature data from: {feature_path}")
    
    # Load the feature data
    try:
        df = pd.read_csv(feature_path)
        print(f"✓ Loaded feature data: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Target distribution: {df['is_best'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Error loading feature data: {e}")
        return 1
    
    print()
    
    # Initialize preprocessor
    print("STEP 1: PREPROCESSING FEATURES")
    print("-" * 40)
    
    preprocessor = FeaturePreprocessor(
        target_column='is_best',
        problem_id_columns=['problem_name', 'algorithm_name'],
        scaling_method='standard',
        handle_missing='median',
        random_state=42
    )
    
    # Fit and transform the data
    X, y, metadata = preprocessor.fit_transform(df)
    
    print(f"✓ Preprocessing complete:")
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target vector: {y.shape}")
    print(f"  Target distribution: {metadata['target_distribution']}")
    print(f"  Missing value strategy: {metadata['missing_values_handled']}")
    print(f"  Scaling method: {metadata['scaling_method']}")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        X, y, test_size=0.3, stratify=True
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print()
    
    # Initialize model trainer
    print("STEP 2: TRAINING BASELINE MODELS")
    print("-" * 40)
    
    trainer = BaselineModelTrainer(random_state=42, n_jobs=-1)
    
    # Train all models
    training_results = trainer.train_all_models(X_train, y_train, verbose=True)
    print()
    
    # Evaluate models
    print("STEP 3: EVALUATING MODELS")
    print("-" * 40)
    
    evaluation_results = trainer.evaluate_models(
        X_test, y_test, 
        X_train, y_train,
        cv_folds=5,
        verbose=True
    )
    print()
    
    # Hyperparameter tuning for top models
    print("STEP 4: HYPERPARAMETER TUNING")
    print("-" * 40)
    
    tuning_results = trainer.hyperparameter_tuning(
        X_train, y_train,
        models_to_tune=None,  # Will automatically select top 3
        cv_folds=3,
        verbose=True
    )
    print()
    
    # Re-evaluate tuned models
    if tuning_results:
        print("STEP 5: EVALUATING TUNED MODELS")
        print("-" * 40)
        
        tuned_evaluation = trainer.evaluate_models(
            X_test, y_test,
            X_train, y_train,
            cv_folds=5,
            verbose=True
        )
        print()
    
    # Create comprehensive evaluation report
    print("STEP 6: COMPREHENSIVE EVALUATION REPORT")
    print("-" * 50)
    
    # Create model evaluator
    evaluator = ModelEvaluator(save_plots=True, plot_dir="plots/baseline_models")
    
    # Get feature names for importance analysis
    feature_names = preprocessor.get_feature_names()
    
    # Create detailed comparison report
    comparison_df = evaluator.create_comparison_report(
        trainer.model_scores,
        feature_names=feature_names,
        model_trainer=trainer
    )
    
    print()
    
    # Save results and models
    print("STEP 7: SAVING RESULTS")
    print("-" * 40)
    
    # Save trained models
    trainer.save_models("models/baseline")
    
    # Save evaluation results
    os.makedirs('results', exist_ok=True)
    
    # Save comparison results
    timestamp = int(time.time())
    comparison_df.to_csv(f'results/baseline_model_comparison_{timestamp}.csv', index=False)
    
    # Save detailed results
    results_summary = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'tuning_results': tuning_results,
        'preprocessing_info': preprocessor.get_preprocessing_info(),
        'data_info': {
            'total_samples': len(df),
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'n_features': X.shape[1],
            'feature_file': latest_feature_file
        }
    }
    
    import json
    with open(f'results/baseline_training_summary_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"✓ Saved model comparison to: results/baseline_model_comparison_{timestamp}.csv")
    print(f"✓ Saved training summary to: results/baseline_training_summary_{timestamp}.json")
    print(f"✓ Saved trained models to: models/baseline/")
    print()
    
    # Final summary
    print("FINAL SUMMARY")
    print("=" * 50)
    
    successful_models = [name for name, result in evaluation_results.items() 
                        if result.get('status') == 'success']
    
    if successful_models:
        # Find best model
        best_model = max(successful_models, 
                        key=lambda x: evaluation_results[x]['test_accuracy'])
        best_accuracy = evaluation_results[best_model]['test_accuracy']
        
        print(f"🏆 Best performing model: {best_model}")
        print(f"   Test accuracy: {best_accuracy:.4f}")
        print(f"   F1 score: {evaluation_results[best_model]['f1_score']:.4f}")
        
        if evaluation_results[best_model]['cv_mean']:
            cv_mean = evaluation_results[best_model]['cv_mean']
            cv_std = evaluation_results[best_model]['cv_std']
            print(f"   Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}")
        
        print()
        print(f"✅ Successfully trained {len(successful_models)} baseline models")
        print(f"📊 Results and plots saved to results/ and plots/ directories")
        print()
        
        print("🚀 Next steps:")
        print("1. Analyze feature importance from best models")
        print("2. Implement ensemble methods")
        print("3. Begin Transformer architecture development")
        print("4. Implement advanced evaluation metrics")
        
    else:
        print("❌ No models trained successfully. Check data and configuration.")
        return 1
    
    return 0


def run_quick_baseline_test():
    """Run a quick test with a smaller subset for rapid iteration."""
    
    print("QUICK BASELINE MODEL TEST")
    print("=" * 50)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Generate synthetic binary targets (algorithm selection)
    y = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    print(f"Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize and train models
    trainer = BaselineModelTrainer(random_state=42)
    
    print("\nTraining baseline models...")
    training_results = trainer.train_all_models(X_train, y_train, verbose=True)
    
    print("\nEvaluating models...")
    evaluation_results = trainer.evaluate_models(X_test, y_test, verbose=True)
    
    print("\n✅ Quick test completed successfully!")
    
    successful_models = sum(1 for r in evaluation_results.values() if r.get('status') == 'success')
    print(f"   {successful_models}/{len(trainer.models)} models trained successfully")
    
    return 0


if __name__ == "__main__":
    # Check if we should run quick test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        exit(run_quick_baseline_test())
    else:
        exit(main()) 