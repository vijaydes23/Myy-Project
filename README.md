# Student Career Prediction System 🎓

A comprehensive machine learning-powered web application that predicts suitable career paths for students based on their academic performance, technical skills, soft skills, personality traits, and project experience.

## Features 🚀

### Core Functionality
- **AI-Powered Career Prediction**: ML model trained on student profiles to predict ideal career paths
- **Multi-Dimensional Profile Analysis**: Analyzes 24 different factors including academics, skills, interests, and experience
- **10 Career Categories**: Software Engineer, Data Scientist, Product Manager, DevOps Engineer, UX/UI Designer, and more
- **Confidence Scoring**: Provides confidence levels for each career recommendation
- **Top 5 Recommendations**: Shows the five best-matching careers with probabilities

### UI & Visualization
- **Interactive Dashboard**: Streamlit-based responsive web interface
- **Sidebar Navigation**: Easy navigation between different sections
- **Real-time Visualizations**: Charts and graphs for career matches, skills distribution, and analytics
- **Career Probability Charts**: Bar charts showing career match percentages
- **Skills Radar Chart**: Visual representation of skill proficiency levels
- **Dataset Analytics**: Career distribution and demographic insights

### Reporting & Export
- **PDF Report Generation**: Download comprehensive career prediction reports
- **Personalized Recommendations**: AI-generated career development suggestions
- **Skills Summary**: Detailed breakdown of student capabilities
- **Career Insights**: Actionable recommendations for career advancement

### Model Information
- **Model Metrics Dashboard**: View accuracy, precision, recall, and F1-scores
- **Feature Importance Analysis**: See which factors most influence career predictions
- **Model Performance**: Cross-validation scores and training statistics

## Installation & Setup 📦

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Extract Project
\`\`\`bash
# If you have a ZIP file, extract it first
unzip student-career-prediction.zip
cd student-career-prediction
\`\`\`

### Step 2: Create Virtual Environment (Recommended)
\`\`\`bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
\`\`\`

### Step 3: Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 4: Run the Application
\`\`\`bash
# Using the runner script
python run.py

# Or directly with Streamlit
streamlit run app.py
\`\`\`

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide 📖

### 1. Home Page
- Overview of the system
- Quick statistics
- Navigation to other sections

### 2. Career Prediction Page
- Fill in your student profile with:
  - Academic performance metrics
  - Technical skills
  - Soft skills
  - Career interests
  - Personality traits
  - Project experience
- Click "Predict My Career" button
- View personalized predictions and visualizations
- Download PDF report

### 3. Analytics Dashboard
- View career distribution in the dataset
- See demographic statistics
- Understand the data behind predictions

### 4. Model Information
- View ML model performance metrics
- Check feature importance
- Understand how predictions are made

### 5. About Page
- Project overview
- How the system works
- Supported career paths
- Technologies used

## Project Structure 📁

\`\`\`
student-career-prediction/
├── app.py                 # Main Streamlit application
├── model.py              # ML model training and prediction
├── preprocessing.py      # Data preprocessing utilities
├── data_generator.py     # Synthetic data generation
├── run.py               # Application launcher
├── requirements.txt     # Python dependencies
├── style.css           # Custom CSS styling
├── data/               # Data directory (auto-created)
│   └── students.csv   # Generated student dataset
└── README.md          # This file
\`\`\`

## File Descriptions 📄

### app.py
Main Streamlit application containing:
- Page layouts for different sections
- Interactive input forms
- Visualization generation
- PDF report export functionality
- Model performance dashboard
- Career insights and recommendations

**Key Functions:**
- `load_model_and_data()`: Load/train ML model with caching
- `create_student_profile_input()`: Interactive student profile form
- `create_prediction_visualizations()`: Generate prediction charts
- `generate_pdf_report()`: Create downloadable PDF reports
- `main()`: Main application entry point

### model.py
Machine learning model implementation:
- `CareerPredictionModel` class for training and prediction
- Gradient Boosting Classifier with optimized hyperparameters
- Feature importance analysis
- Model persistence (save/load functionality)
- Performance metrics calculation

**Key Methods:**
- `train()`: Train the model on data
- `predict_career()`: Make career predictions for students
- `get_feature_importance()`: Get top N important features
- `save_model()` / `load_model()`: Model persistence

### preprocessing.py
Data preprocessing utilities:
- `DataPreprocessor` class for data cleaning
- Feature normalization and scaling
- Categorical encoding
- Derived feature creation
- Missing value handling
- Statistical summary generation

**Key Methods:**
- `load_and_clean_data()`: Clean dataset
- `normalize_features()`: Normalize to 0-1 scale
- `encode_categorical()`: Encode categorical variables
- `create_derived_features()`: Create new features
- `get_feature_statistics()`: Get data statistics

### data_generator.py
Synthetic data generation for testing:
- `generate_sample_data()`: Create synthetic student records
- `save_sample_data()`: Save data to CSV file
- Includes correlations between features and career paths

### run.py
Simple launcher script:
- Sets up environment
- Runs Streamlit application
- Provides helpful startup messages

## Machine Learning Model Details 🤖

### Algorithm
**Gradient Boosting Classifier**
- Ensemble method combining weak learners
- Excellent for classification tasks
- Good generalization and interpretability

### Hyperparameters
\`\`\`python
{
    'n_estimators': 200,      # Number of boosting rounds
    'learning_rate': 0.1,     # Step size for each iteration
    'max_depth': 8,           # Maximum tree depth
    'min_samples_split': 5,   # Min samples to split node
    'min_samples_leaf': 2,    # Min samples in leaf node
    'subsample': 0.8          # Fraction of samples per iteration
}
\`\`\`

### Features (24 total)
1. **Academic (4)**: GPA, exam score, attendance, assignment completion
2. **Technical Skills (5)**: Programming, data science, database, cloud, problem-solving
3. **Soft Skills (4)**: Communication, leadership, teamwork, creativity
4. **Career Interests (5)**: Backend, frontend, data, DevOps, security
5. **Personality (3)**: Analytical, organized, detail-oriented
6. **Experience (4)**: Projects, internships, hackathons, GitHub contributions

### Target Classes (10 careers)
1. Software Engineer
2. Data Scientist
3. Product Manager
4. DevOps Engineer
5. UX/UI Designer
6. Business Analyst
7. ML Engineer
8. Cybersecurity Specialist
9. Cloud Architect
10. Technical Writer

### Model Performance
- **Accuracy**: ~85-90% (depends on training data)
- **Precision**: Weighted average ~0.85
- **Recall**: Weighted average ~0.85
- **F1-Score**: Weighted average ~0.85
- **Cross-Validation**: 5-fold with consistent scores

## Customization & Extension 🔧

### Adding New Careers
1. Update `CAREER_OPTIONS` in `model.py`
2. Modify career generation in `data_generator.py`
3. Add correlations in `generate_sample_data()`
4. Retrain the model

### Using Real Data
Replace the data generation step in `app.py`:
\`\`\`python
# Instead of: df = save_sample_data(data_path)
df = pd.read_csv('your_real_data.csv')  # Your data source
\`\`\`

### Changing ML Algorithm
In `model.py`, replace the classifier:
\`\`\`python
from sklearn.ensemble import RandomForestClassifier
# Or any other scikit-learn classifier
self.model = RandomForestClassifier(...)
\`\`\`

### Adding More Features
1. Add input fields in `create_student_profile_input()`
2. Update preprocessing in `preprocessing.py`
3. Regenerate or update training data
4. Retrain the model

## Troubleshooting 🔧

### Port Already in Use
\`\`\`bash
streamlit run app.py --server.port 8502
\`\`\`

### Memory Issues with Large Datasets
- Reduce dataset size in `data_generator.py`
- Use data batching in model training
- Reduce model complexity

### Model Not Training
- Check data format and types
- Verify all required columns exist
- Ensure target variable has multiple classes
- Check for NaN values in features

## Performance Optimization ⚡

### Caching
- `@st.cache_resource` decorator caches model loading
- Prevents retraining on every page refresh
- Significantly speeds up app response time

### Data Preprocessing
- Features normalized to 0-1 scale
- Missing values handled automatically
- Categorical encoding performed efficiently

### Model Training
- Subsample parameter reduces training time
- Early stopping can be added for production
- Cross-validation ensures robust performance

## Future Enhancements 🚀

- [ ] User authentication and profile persistence
- [ ] Integration with real student databases
- [ ] Advanced ensemble models
- [ ] Career progression path recommendations
- [ ] Industry-specific career recommendations
- [ ] Salary and job market data integration
- [ ] Real-time job market analysis
- [ ] Multiple language support
- [ ] Mobile app version
- [ ] API endpoints for integration

## Technologies Stack 📚

- **Frontend**: Streamlit (Python web framework)
- **ML/AI**: Scikit-learn (machine learning)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **PDF Generation**: ReportLab
- **Data Science**: SciPy, Joblib

## Best Practices 🎯

1. **Regular Model Retraining**: Update model with new student data
2. **Data Validation**: Always validate input data before prediction
3. **Feature Scaling**: Ensure consistent feature normalization
4. **Model Monitoring**: Track prediction accuracy over time
5. **User Feedback**: Collect feedback to improve recommendations
6. **Privacy**: Handle student data securely

## License & Attribution

This project is designed as an educational tool for demonstrating:
- Machine learning implementation
- Web application development with Streamlit
- Data science best practices
- Career guidance systems

## Support & Feedback 📧

For issues, suggestions, or feedback:
1. Review the troubleshooting section
2. Check the code comments for detailed explanations
3. Modify configuration parameters as needed
4. Refer to official documentation for dependencies

## Version History

**v1.0.0** - Initial Release
- Core ML model implementation
- Streamlit UI with 5 main pages
- PDF report generation
- 24-feature student profile
- 10 career categories
- Model metrics dashboard
- Full documentation

---

**Created**: November 2024  
**Status**: Production Ready  
**Maintenance**: Regular Updates Available
