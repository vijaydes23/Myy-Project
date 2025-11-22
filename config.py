"""
Configuration Module
===================
Centralized configuration for the application.
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Data
DATA_PATH = DATA_DIR / 'students.csv'
MODEL_PATH = MODELS_DIR / 'career_model.pkl'

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 8,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.8,
    'random_state': 42
}

# Training parameters
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# Application settings
APP_TITLE = "Student Career Prediction System"
APP_ICON = "🎓"

# Colors (matching Streamlit theme)
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#D62828'
}

# Careers
CAREERS = [
    'Software Engineer',
    'Data Scientist',
    'Product Manager',
    'DevOps Engineer',
    'UX/UI Designer',
    'Business Analyst',
    'ML Engineer',
    'Cybersecurity Specialist',
    'Cloud Architect',
    'Technical Writer'
]

# Features
FEATURE_GROUPS = {
    'academic': [
        'gpa',
        'exam_score',
        'class_attendance',
        'assignment_completion'
    ],
    'technical': [
        'programming_skills',
        'data_science_skills',
        'database_knowledge',
        'cloud_skills',
        'problem_solving'
    ],
    'soft_skills': [
        'communication',
        'leadership',
        'teamwork',
        'creativity'
    ],
    'interests': [
        'interest_backend',
        'interest_frontend',
        'interest_data',
        'interest_devops',
        'interest_security'
    ],
    'personality': [
        'analytical',
        'organized',
        'detail_oriented'
    ],
    'experience': [
        'projects_completed',
        'internships',
        'hackathons',
        'github_contributions'
    ]
}
