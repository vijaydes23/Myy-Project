"""
Model Training Script
====================
Standalone script to train and save the career prediction model.
Can be run separately to update the trained model.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import LabelEncoder

from model import CareerPredictionModel
from preprocessing import DataPreprocessor
from data_generator import generate_sample_data, save_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_and_save_model(output_path='models/career_model.pkl'):
    """
    Train the career prediction model and save it.
    
    Args:
        output_path (str): Path to save the trained model
    """
    
    logger.info("="*60)
    logger.info("CAREER PREDICTION MODEL TRAINING")
    logger.info("="*60)
    
    # Create models directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Step 1: Load or generate data
    logger.info("\nStep 1: Loading data...")
    data_path = 'data/students.csv'
    
    if os.path.exists(data_path):
        logger.info(f"Loading existing data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        logger.info("Generating new synthetic data...")
        df = save_sample_data(data_path)
    
    logger.info(f"Loaded {len(df)} student records")
    
    # Step 2: Preprocess data
    logger.info("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_clean_data(df)
    logger.info("Data cleaning complete")
    
    # Step 3: Prepare features and target
    logger.info("\nStep 3: Preparing features...")
    X = df.drop(columns=['student_id', 'target_career'])
    y = df['target_career']
    
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Samples: {X.shape[0]}")
    logger.info(f"Career classes: {y.nunique()}")
    
    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Step 4: Train model
    logger.info("\nStep 4: Training model...")
    model = CareerPredictionModel()
    metrics = model.train(X, y_encoded, test_size=0.2, verbose=True)
    
    # Step 5: Save model
    logger.info("\nStep 5: Saving model...")
    model.save_model(output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Step 6: Display summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Model Type: Gradient Boosting Classifier")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Test F1-Score: {metrics['test_f1']:.4f}")
    logger.info(f"CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    logger.info("="*60)
    
    return model, metrics


if __name__ == "__main__":
    train_and_save_model()
    logger.info("\nTraining completed successfully!")
