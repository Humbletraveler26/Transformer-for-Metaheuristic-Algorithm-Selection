"""
Feature preprocessing for machine learning models.

This module provides functionality to preprocess features for training
ML models for metaheuristic algorithm selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class FeaturePreprocessor:
    """
    Preprocesses features for machine learning models.
    
    Handles missing values, feature scaling, encoding, and train/test splits.
    """
    
    def __init__(self, 
                 target_column: str = 'is_best',
                 problem_id_columns: List[str] = None,
                 scaling_method: str = 'standard',
                 handle_missing: str = 'median',
                 random_state: int = 42):
        """
        Initialize the feature preprocessor.
        
        Args:
            target_column: Name of the target variable column
            problem_id_columns: Columns that identify problems (not features)
            scaling_method: Method for feature scaling ('standard', 'minmax', 'none')
            handle_missing: Method for handling missing values ('median', 'mean', 'drop')
            random_state: Random state for reproducibility
        """
        self.target_column = target_column
        self.problem_id_columns = problem_id_columns or ['problem_name', 'algorithm_name']
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.random_state = random_state
        
        # Initialize preprocessors
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.feature_columns = None
        self.fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit preprocessors and transform the data.
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Tuple of (X, y, metadata)
        """
        # Identify feature columns
        self.feature_columns = [col for col in df.columns 
                              if col not in self.problem_id_columns + [self.target_column]]
        
        print(f"Preprocessing {len(self.feature_columns)} features...")
        
        # Extract features and target
        X_df = df[self.feature_columns].copy()
        y = df[self.target_column].values
        
        # Handle missing values
        X_df = self._handle_missing_values(X_df, fit=True)
        
        # Encode categorical features
        X_df = self._encode_categorical_features(X_df, fit=True)
        
        # Scale numerical features
        X = self._scale_features(X_df, fit=True)
        
        # Store metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'target_distribution': pd.Series(y).value_counts().to_dict(),
            'missing_values_handled': self.handle_missing,
            'scaling_method': self.scaling_method
        }
        
        self.fitted = True
        print(f"Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, metadata
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            Transformed feature matrix
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Extract features
        X_df = df[self.feature_columns].copy()
        
        # Apply transformations
        X_df = self._handle_missing_values(X_df, fit=False)
        X_df = self._encode_categorical_features(X_df, fit=False)
        X = self._scale_features(X_df, fit=False)
        
        return X
    
    def _handle_missing_values(self, X_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if self.handle_missing == 'drop':
            # Drop rows with any missing values
            return X_df.dropna()
        
        elif self.handle_missing in ['mean', 'median']:
            # Impute missing values
            if fit:
                strategy = self.handle_missing
                self.imputer = SimpleImputer(strategy=strategy)
                
                # Separate numeric and non-numeric columns
                numeric_cols = X_df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    X_df[numeric_cols] = self.imputer.fit_transform(X_df[numeric_cols])
                
                # For non-numeric, use mode
                non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns
                for col in non_numeric_cols:
                    if X_df[col].isnull().any():
                        mode_value = X_df[col].mode()
                        if len(mode_value) > 0:
                            X_df[col].fillna(mode_value[0], inplace=True)
            else:
                # Transform using fitted imputer
                numeric_cols = X_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and self.imputer is not None:
                    X_df[numeric_cols] = self.imputer.transform(X_df[numeric_cols])
        
        return X_df
    
    def _encode_categorical_features(self, X_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_cols = X_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                # Fit label encoder
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Transform using fitted encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X_df[col] = X_df[col].astype(str)
                    mask = X_df[col].isin(le.classes_)
                    X_df.loc[mask, col] = le.transform(X_df.loc[mask, col])
                    X_df.loc[~mask, col] = -1  # Assign -1 to unseen categories
        
        return X_df
    
    def _scale_features(self, X_df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale numerical features."""
        if self.scaling_method == 'none':
            return X_df.values
        
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            
            X = self.scaler.fit_transform(X_df)
        else:
            if self.scaler is not None:
                X = self.scaler.transform(X_df)
            else:
                X = X_df.values
        
        return X
    
    def create_train_test_split(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              test_size: float = 0.2,
                              stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split with optional stratification.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            stratify: Whether to stratify the split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify and len(np.unique(y)) > 1 else None
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
    
    def get_feature_names(self) -> List[str]:
        """Get the names of processed features."""
        return self.feature_columns if self.feature_columns is not None else []
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing steps."""
        return {
            'fitted': self.fitted,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'scaling_method': self.scaling_method,
            'missing_value_strategy': self.handle_missing,
            'categorical_encoders': list(self.label_encoders.keys()),
            'target_column': self.target_column,
            'problem_id_columns': self.problem_id_columns
        } 