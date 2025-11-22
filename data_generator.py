"""
Sample Data Generator
====================
Generates synthetic student data for demonstration and testing.
Can be replaced with real data source.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data(n_samples=500):
    """
    Generate synthetic student dataset.
    
    Args:
        n_samples (int): Number of student records to generate
        
    Returns:
        pd.DataFrame: Generated dataset with student information
    """
    np.random.seed(42)
    
    # Career labels
    careers = [
        'Software Engineer', 'Data Scientist', 'Product Manager',
        'DevOps Engineer', 'UX/UI Designer', 'Business Analyst',
        'ML Engineer', 'Cybersecurity Specialist', 'Cloud Architect',
        'Technical Writer'
    ]
    
    # Generate data
    data = {
        'student_id': [f'STU{str(i).zfill(4)}' for i in range(n_samples)],
        
        # Academic Performance
        'gpa': np.random.uniform(2.0, 4.0, n_samples),
        'exam_score': np.random.uniform(50, 100, n_samples),
        'class_attendance': np.random.uniform(60, 100, n_samples),
        'assignment_completion': np.random.uniform(70, 100, n_samples),
        
        # Technical Skills (1-10 scale)
        'programming_skills': np.random.uniform(1, 10, n_samples),
        'data_science_skills': np.random.uniform(1, 10, n_samples),
        'database_knowledge': np.random.uniform(1, 10, n_samples),
        'cloud_skills': np.random.uniform(1, 10, n_samples),
        'problem_solving': np.random.uniform(1, 10, n_samples),
        
        # Soft Skills (1-10 scale)
        'communication': np.random.uniform(1, 10, n_samples),
        'leadership': np.random.uniform(1, 10, n_samples),
        'teamwork': np.random.uniform(1, 10, n_samples),
        'creativity': np.random.uniform(1, 10, n_samples),
        
        # Interests & Aptitudes
        'interest_backend': np.random.uniform(1, 10, n_samples),
        'interest_frontend': np.random.uniform(1, 10, n_samples),
        'interest_data': np.random.uniform(1, 10, n_samples),
        'interest_devops': np.random.uniform(1, 10, n_samples),
        'interest_security': np.random.uniform(1, 10, n_samples),
        
        # Personality Traits (1-5 scale)
        'analytical': np.random.uniform(1, 5, n_samples),
        'creative': np.random.uniform(1, 5, n_samples),
        'organized': np.random.uniform(1, 5, n_samples),
        'detail_oriented': np.random.uniform(1, 5, n_samples),
        'social': np.random.uniform(1, 5, n_samples),
        
        # Project Experience
        'projects_completed': np.random.randint(1, 20, n_samples),
        'internships': np.random.randint(0, 5, n_samples),
        'hackathons': np.random.randint(0, 10, n_samples),
        'github_contributions': np.random.randint(0, 500, n_samples),
        
        # Career Target (Target Variable)
        'target_career': np.random.choice(careers, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation between features and target
    for idx, row in df.iterrows():
        career = row['target_career']
        
        if career == 'Software Engineer':
            df.loc[idx, 'programming_skills'] = min(10, df.loc[idx, 'programming_skills'] + np.random.uniform(1, 2))
            df.loc[idx, 'projects_completed'] = min(20, df.loc[idx, 'projects_completed'] + np.random.randint(2, 5))
        elif career == 'Data Scientist':
            df.loc[idx, 'data_science_skills'] = min(10, df.loc[idx, 'data_science_skills'] + np.random.uniform(1, 2))
            df.loc[idx, 'analytical'] = min(5, df.loc[idx, 'analytical'] + np.random.uniform(0.5, 1))
        elif career == 'Product Manager':
            df.loc[idx, 'leadership'] = min(10, df.loc[idx, 'leadership'] + np.random.uniform(1, 1.5))
            df.loc[idx, 'communication'] = min(10, df.loc[idx, 'communication'] + np.random.uniform(1, 1.5))
        elif career == 'DevOps Engineer':
            df.loc[idx, 'cloud_skills'] = min(10, df.loc[idx, 'cloud_skills'] + np.random.uniform(1.5, 2))
            df.loc[idx, 'organized'] = min(5, df.loc[idx, 'organized'] + np.random.uniform(0.5, 1))
        elif career == 'UX/UI Designer':
            df.loc[idx, 'creativity'] = min(5, df.loc[idx, 'creativity'] + np.random.uniform(0.5, 1))
            df.loc[idx, 'social'] = min(5, df.loc[idx, 'social'] + np.random.uniform(0.5, 1))
        elif career == 'ML Engineer':
            df.loc[idx, 'data_science_skills'] = min(10, df.loc[idx, 'data_science_skills'] + np.random.uniform(1.5, 2))
            df.loc[idx, 'programming_skills'] = min(10, df.loc[idx, 'programming_skills'] + np.random.uniform(1, 1.5))
    
    return df


def save_sample_data(filepath='data/students.csv'):
    """
    Generate and save sample data.
    
    Args:
        filepath (str): Path to save the CSV file
    """
    df = generate_sample_data(500)
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    return df
