"""
Data Preprocessing Module
========================
Handles data cleaning, normalization, and feature engineering
for the Career Prediction System.

This module provides functions for:
- Loading and cleaning student data
- Normalizing features to 0-1 scale
- Handling missing values
- Creating derived features
- Splitting data for training/testing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses student data for machine learning models.
    Handles normalization, feature engineering, and data cleaning.
    """

    def __init__(self):
        """Initialize the preprocessor with scalers."""
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.categorical_mapping = {}

    def load_and_clean_data(self, df):
        """
        Load and clean the dataset.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info(f"Cleaning dataset with shape: {df.shape}")
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        logger.info("Missing values handled")
        return df

    def normalize_features(self, df, features_to_normalize):
        """
        Normalize specified features to 0-1 scale.
        
        Args:
            df (pd.DataFrame): Dataset to normalize
            features_to_normalize (list): List of column names to normalize
            
        Returns:
            pd.DataFrame: Dataset with normalized features
        """
        df_copy = df.copy()
        
        for feature in features_to_normalize:
            if feature in df_copy.columns:
                min_val = df_copy[feature].min()
                max_val = df_copy[feature].max()
                
                if max_val - min_val > 0:
                    df_copy[feature] = (df_copy[feature] - min_val) / (max_val - min_val)
                else:
                    df_copy[feature] = 0
        
        logger.info(f"Normalized {len(features_to_normalize)} features")
        return df_copy

    def encode_categorical(self, df, categorical_columns):
        """
        Encode categorical variables to numerical values.
        
        Args:
            df (pd.DataFrame): Dataset with categorical features
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Dataset with encoded categories
        """
        df_copy = df.copy()
        
        for col in categorical_columns:
            if col in df_copy.columns:
                unique_values = df_copy[col].unique()
                self.categorical_mapping[col] = {v: i for i, v in enumerate(unique_values)}
                df_copy[col] = df_copy[col].map(self.categorical_mapping[col])
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
        return df_copy

    def create_derived_features(self, df):
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Original dataset
            
        Returns:
            pd.DataFrame: Dataset with new derived features
        """
        df_copy = df.copy()
        
        # Academic strength indicator
        if 'gpa' in df_copy.columns and 'exam_score' in df_copy.columns:
            df_copy['academic_strength'] = (df_copy['gpa'] + df_copy['exam_score']) / 2
        
        # Technical proficiency index
        if 'programming_skills' in df_copy.columns and 'data_science_skills' in df_copy.columns:
            df_copy['technical_proficiency'] = (df_copy['programming_skills'] + df_copy['data_science_skills']) / 2
        
        # Overall readiness score
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_copy['overall_readiness'] = df_copy[numeric_cols].mean(axis=1)
        
        logger.info("Derived features created")
        return df_copy

    def get_feature_statistics(self, df):
        """
        Get statistical summary of features.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Feature statistics
        """
        stats = {
            'mean': df.select_dtypes(include=[np.number]).mean().to_dict(),
            'std': df.select_dtypes(include=[np.number]).std().to_dict(),
            'min': df.select_dtypes(include=[np.number]).min().to_dict(),
            'max': df.select_dtypes(include=[np.number]).max().to_dict(),
        }
        return stats
