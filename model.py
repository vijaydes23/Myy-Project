"""
Machine Learning Model Module
=============================
Implements career prediction model training and inference.

This module provides:
- Model training with multiple algorithms
- Career prediction for individual students
- Model evaluation metrics
- Feature importance analysis
- Model persistence (save/load)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocessing import DataPreprocessor
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Career mapping
CAREER_OPTIONS = {
    0: 'Software Engineer',
    1: 'Data Scientist',
    2: 'Product Manager',
    3: 'DevOps Engineer',
    4: 'UX/UI Designer',
    5: 'Business Analyst',
    6: 'ML Engineer',
    7: 'Cybersecurity Specialist',
    8: 'Cloud Architect',
    9: 'Technical Writer'
}

REVERSE_CAREER_MAP = {v: k for k, v in CAREER_OPTIONS.items()}


class CareerPredictionModel:
    """
    Machine Learning model for predicting suitable careers for students.
    Uses ensemble methods for robust predictions.
    """

    def __init__(self):
        """Initialize the model with multiple classifiers."""
        self.preprocessor = DataPreprocessor()
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            subsample=0.8
        )
        self.feature_names = None
        self.feature_importance = None
        self.model_trained = False
        self.metrics = {}

    def train(self, X, y, test_size=0.2, verbose=True):
        """
        Train the career prediction model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable (career labels)
            test_size (float): Proportion of data for testing
            verbose (bool): Print training progress
            
        Returns:
            dict: Training metrics
        """
        logger.info(f"Starting model training with {len(X)} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if verbose:
            logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        self.model_trained = True
        
        if verbose:
            logger.info(f"Model trained successfully")
            logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")
            logger.info(f"Test F1-Score: {self.metrics['test_f1']:.4f}")
        
        return self.metrics

    def predict_career(self, student_data):
        """
        Predict suitable careers for a student.
        
        Args:
            student_data (pd.DataFrame): Student features (1 row)
            
        Returns:
            dict: Prediction results with probabilities
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure features are in correct order
        student_data = student_data[self.feature_names]
        
        # Get prediction and probabilities
        prediction = self.model.predict(student_data)[0]
        probabilities = self.model.predict_proba(student_data)[0]
        
        # Create results
        career_probs = {
            CAREER_OPTIONS[i]: float(prob) for i, prob in enumerate(probabilities)
        }
        
        # Sort by probability
        sorted_careers = sorted(career_probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_career': CAREER_OPTIONS[prediction],
            'confidence': float(probabilities[prediction]),
            'top_5_careers': sorted_careers[:5],
            'all_probabilities': career_probs
        }

    def get_feature_importance(self, top_n=10):
        """
        Get top N most important features.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            dict: Top features and their importance scores
        """
        if self.feature_importance is None:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])

    def save_model(self, filepath):
        """
        Save trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.model_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load trained model from disk.
        
        Args:
            filepath (str): Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.metrics = model_data['metrics']
        self.model_trained = True
        logger.info(f"Model loaded from {filepath}")

    def get_metrics_summary(self):
        """
        Get summary of model metrics.
        
        Returns:
            dict: Model performance metrics
        """
        return self.metrics
