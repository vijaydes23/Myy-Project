"""
Student Career Prediction System
=================================
Main Streamlit application for career prediction.

Features:
- Student profile creation and management
- Real-time career prediction with confidence scores
- Interactive visualizations and analytics
- PDF report generation
- Model performance metrics
- Career recommendations engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import os

from model import CareerPredictionModel, CAREER_OPTIONS
from preprocessing import DataPreprocessor
from data_generator import generate_sample_data, save_sample_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Career Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #2E86AB;
            --secondary-color: #A23B72;
            --success-color: #06A77D;
            --warning-color: #F18F01;
            --danger-color: #D62828;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f0f2f6;
        }
        
        /* Main content area */
        .main {
            background-color: #ffffff;
        }
        
        /* Header styling */
        h1 {
            color: #2E86AB;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        h2 {
            color: #2E86AB;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        
        h3 {
            color: #A23B72;
            font-weight: 500;
            margin-top: 15px;
        }
        
        /* Card styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #2E86AB;
            color: white;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 5px;
        }
        
        .stButton > button:hover {
            background-color: #1f5a7a;
        }
        
        /* Input field styling */
        .stSlider > div > div > div > div {
            color: #2E86AB;
        }
        
        /* Success message */
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Info message */
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """
    Load or train the machine learning model.
    Uses caching to avoid retraining on every app run.
    
    Returns:
        tuple: (trained_model, training_data)
    """
    logger.info("Loading model and data...")
    
    # Generate or load data
    data_path = 'data/students.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = save_sample_data(data_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean and preprocess data
    df = preprocessor.load_and_clean_data(df)
    
    # Separate features and target
    X = df.drop(columns=['student_id', 'target_career'])
    y = df['target_career']
    
    # Encode target variable
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Initialize and train model
    model = CareerPredictionModel()
    model.train(X, y_encoded, test_size=0.2, verbose=True)
    
    logger.info("Model loaded successfully")
    return model, df, X.columns.tolist()


def create_student_profile_input():
    """
    Create interactive input form for student profile.
    
    Returns:
        dict: Student profile data
    """
    st.subheader("📋 Student Profile Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Academic Performance")
        gpa = st.slider(
            "GPA (0.0 - 4.0)",
            0.0, 4.0, 3.5,
            help="Cumulative Grade Point Average"
        )
        exam_score = st.slider(
            "Exam Score (0-100)",
            0, 100, 85,
            help="Average exam performance"
        )
        attendance = st.slider(
            "Class Attendance (%)",
            0, 100, 90,
            help="Percentage of classes attended"
        )
        assignment_completion = st.slider(
            "Assignment Completion (%)",
            0, 100, 95,
            help="Percentage of assignments completed"
        )
    
    with col2:
        st.markdown("### Technical Skills (1-10)")
        programming = st.slider(
            "Programming Skills",
            1, 10, 7,
            help="Proficiency in coding languages"
        )
        data_science = st.slider(
            "Data Science Skills",
            1, 10, 6,
            help="Knowledge of ML, statistics, data analysis"
        )
        database = st.slider(
            "Database Knowledge",
            1, 10, 5,
            help="SQL and database design skills"
        )
        cloud = st.slider(
            "Cloud Skills",
            1, 10, 4,
            help="AWS, Azure, GCP experience"
        )
        problem_solving = st.slider(
            "Problem Solving",
            1, 10, 8,
            help="Ability to solve complex problems"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Soft Skills (1-10)")
        communication = st.slider(
            "Communication",
            1, 10, 7,
            help="Verbal and written communication"
        )
        leadership = st.slider(
            "Leadership",
            1, 10, 6,
            help="Ability to lead and inspire"
        )
        teamwork = st.slider(
            "Teamwork",
            1, 10, 8,
            help="Collaboration and team skills"
        )
        creativity = st.slider(
            "Creativity",
            1, 10, 6,
            help="Innovation and creative thinking"
        )
    
    with col4:
        st.markdown("### Career Interests (1-10)")
        interest_backend = st.slider(
            "Backend Development",
            1, 10, 7,
            help="Interest in backend/server-side development"
        )
        interest_frontend = st.slider(
            "Frontend Development",
            1, 10, 6,
            help="Interest in frontend/UI development"
        )
        interest_data = st.slider(
            "Data Analytics & ML",
            1, 10, 8,
            help="Interest in data science and ML"
        )
        interest_devops = st.slider(
            "DevOps & Cloud",
            1, 10, 5,
            help="Interest in DevOps and cloud infrastructure"
        )
        interest_security = st.slider(
            "Cybersecurity",
            1, 10, 4,
            help="Interest in security and privacy"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("### Personality Traits (1-5)")
        analytical = st.slider(
            "Analytical",
            1, 5, 4,
            help="Analytical thinking ability"
        )
        organized = st.slider(
            "Organized",
            1, 5, 4,
            help="Organization and planning skills"
        )
        detail_oriented = st.slider(
            "Detail-Oriented",
            1, 5, 4,
            help="Attention to detail"
        )
    
    with col6:
        st.markdown("### Project Experience")
        projects = st.slider(
            "Projects Completed",
            0, 20, 8,
            help="Number of completed projects"
        )
        internships = st.slider(
            "Internships",
            0, 5, 1,
            help="Number of internships completed"
        )
        hackathons = st.slider(
            "Hackathons Participated",
            0, 10, 3,
            help="Number of hackathon participations"
        )
        github = st.slider(
            "GitHub Contributions",
            0, 500, 150,
            help="Approximate GitHub contributions"
        )
    
    # Create profile dictionary
    profile = {
        'gpa': gpa,
        'exam_score': exam_score,
        'class_attendance': attendance,
        'assignment_completion': assignment_completion,
        'programming_skills': programming,
        'data_science_skills': data_science,
        'database_knowledge': database,
        'cloud_skills': cloud,
        'problem_solving': problem_solving,
        'communication': communication,
        'leadership': leadership,
        'teamwork': teamwork,
        'creativity': creativity,
        'interest_backend': interest_backend,
        'interest_frontend': interest_frontend,
        'interest_data': interest_data,
        'interest_devops': interest_devops,
        'interest_security': interest_security,
        'analytical': analytical,
        'organized': organized,
        'detail_oriented': detail_oriented,
        'projects_completed': projects,
        'internships': internships,
        'hackathons': hackathons,
        'github_contributions': github,
    }
    
    return profile


def create_prediction_visualizations(profile, prediction_result):
    """
    Create interactive visualizations for prediction results.
    
    Args:
        profile (dict): Student profile data
        prediction_result (dict): Model prediction results
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Top Career Matches")
        
        # Create career probability chart
        careers_data = prediction_result['top_5_careers']
        careers = [c[0] for c in careers_data]
        probs = [c[1] * 100 for c in careers_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=probs,
                y=careers,
                orientation='h',
                marker=dict(
                    color=probs,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f'{p:.1f}%' for p in probs],
                textposition='outside',
            )
        ])
        
        fig.update_layout(
            title="Career Match Probability",
            xaxis_title="Confidence (%)",
            yaxis_title="Career",
            height=300,
            margin=dict(l=150, r=50, t=50, b=50),
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Skills Distribution")
        
        # Create skills radar chart
        skills_dict = {
            'Programming': profile['programming_skills'],
            'Data Science': profile['data_science_skills'],
            'Problem Solving': profile['problem_solving'],
            'Communication': profile['communication'],
            'Leadership': profile['leadership'],
            'Teamwork': profile['teamwork'],
        }
        
        fig = go.Figure(data=go.Scatterpolar(
            r=list(skills_dict.values()),
            theta=list(skills_dict.keys()),
            fill='toself',
            marker=dict(color='#2E86AB'),
            line=dict(color='#2E86AB'),
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            title="Skills Profile",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_insights_section(profile, prediction_result):
    """
    Create personalized career insights and recommendations.
    
    Args:
        profile (dict): Student profile data
        prediction_result (dict): Model prediction results
    """
    st.markdown("### 💡 Career Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Top Career Match",
            value=prediction_result['top_career'],
            delta=f"Confidence: {prediction_result['confidence']*100:.1f}%"
        )
    
    with col2:
        avg_technical = np.mean([
            profile['programming_skills'],
            profile['data_science_skills'],
            profile['database_knowledge'],
            profile['cloud_skills']
        ])
        st.metric(
            label="Technical Proficiency",
            value=f"{avg_technical:.1f}/10",
            delta="Above Average" if avg_technical > 5 else "Below Average"
        )
    
    with col3:
        project_score = profile['projects_completed'] + profile['internships'] * 3 + profile['hackathons']
        st.metric(
            label="Experience Score",
            value=f"{project_score}",
            delta="Strong Portfolio" if project_score > 15 else "Growing"
        )
    
    # Detailed recommendations
    st.markdown("#### Personalized Recommendations:")
    
    recommendations = generate_recommendations(profile, prediction_result)
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")


def generate_recommendations(profile, prediction_result):
    """
    Generate personalized career recommendations.
    
    Args:
        profile (dict): Student profile
        prediction_result (dict): Prediction results
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Strength-based recommendations
    if profile['programming_skills'] > 7:
        recommendations.append(
            "✅ Strong programming skills! Consider specializing in backend development or ML engineering."
        )
    
    if profile['data_science_skills'] > 7:
        recommendations.append(
            "✅ Excellent data science foundation! Data Scientist or ML Engineer roles would be ideal."
        )
    
    if profile['leadership'] > 3.5 and profile['communication'] > 7:
        recommendations.append(
            "✅ Great leadership and communication skills! Product Manager or Team Lead positions suit you."
        )
    
    if profile['creativity'] > 3.5:
        recommendations.append(
            "✅ High creativity! Consider UX/UI Design or Product Management roles."
        )
    
    # Area for improvement
    if profile['cloud_skills'] < 4:
        recommendations.append(
            "📚 Consider learning cloud technologies (AWS, Azure, GCP) to expand career options."
        )
    
    if profile['projects_completed'] < 5:
        recommendations.append(
            "📚 Build more portfolio projects to strengthen your candidacy."
        )
    
    if profile['communication'] < 6:
        recommendations.append(
            "📚 Improve communication skills through practice and public speaking courses."
        )
    
    # Career-specific tips
    top_career = prediction_result['top_career']
    if 'Data Scientist' in top_career and profile['data_science_skills'] < 8:
        recommendations.append(
            f"🎯 To excel as a {top_career}, focus on advanced ML algorithms and big data tools."
        )
    
    if 'Software Engineer' in top_career and profile['database_knowledge'] < 6:
        recommendations.append(
            f"🎯 To excel as a {top_career}, deepen your database design and optimization knowledge."
        )
    
    if 'DevOps Engineer' in top_career and profile['cloud_skills'] < 7:
        recommendations.append(
            f"🎯 To excel as a {top_career}, get certified in cloud platforms (AWS/GCP)."
        )
    
    return recommendations


def create_model_metrics_section(model):
    """
    Display model performance metrics and statistics.
    
    Args:
        model: Trained CareerPredictionModel instance
    """
    st.markdown("### 📈 Model Performance Metrics")
    
    metrics = model.get_metrics_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{metrics['test_accuracy']:.2%}",
            delta="Test Set"
        )
    
    with col2:
        st.metric(
            label="Precision",
            value=f"{metrics['test_precision']:.2%}",
            delta="Weighted"
        )
    
    with col3:
        st.metric(
            label="Recall",
            value=f"{metrics['test_recall']:.2%}",
            delta="Weighted"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value=f"{metrics['test_f1']:.2%}",
            delta="Weighted"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        feature_importance = model.get_feature_importance(top_n=10)
        
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker=dict(color='#A23B72'),
                    text=[f'{imp:.3f}' for imp in importances],
                    textposition='outside',
                )
            ])
            
            fig.update_layout(
                title="Top 10 Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                margin=dict(l=150, r=50, t=50, b=50),
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model statistics
        st.markdown("#### Model Training Statistics")
        stats_text = f"""
        - **Test Accuracy**: {metrics['test_accuracy']:.4f}
        - **Cross-Validation Mean**: {metrics['cv_mean']:.4f}
        - **Cross-Validation Std**: {metrics['cv_std']:.4f}
        - **Training Accuracy**: {metrics['train_accuracy']:.4f}
        - **Model Type**: Gradient Boosting Classifier
        - **Total Features**: 23
        - **Career Classes**: 10
        
        The model is trained on synthetic student data and achieves
        strong performance across multiple evaluation metrics.
        """
        st.markdown(stats_text)


def generate_pdf_report(profile, prediction_result, model):
    """
    Generate a PDF report of the career prediction.
    
    Args:
        profile (dict): Student profile
        prediction_result (dict): Prediction results
        model: CareerPredictionModel instance
        
    Returns:
        bytes: PDF file content
    """
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from io import BytesIO
    
    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#A23B72'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Add title
    story.append(Paragraph("Career Prediction Report", title_style))
    story.append(Spacer(1, 0.2 * 72))
    
    # Add timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<i>Generated on {timestamp}</i>", styles['Normal']))
    story.append(Spacer(1, 0.3 * 72))
    
    # Primary Prediction
    story.append(Paragraph("🎯 Primary Career Match", heading_style))
    primary_data = [
        ['Career', prediction_result['top_career']],
        ['Confidence', f"{prediction_result['confidence']*100:.1f}%"],
    ]
    primary_table = Table(primary_data, colWidths=[2*72, 3*72])
    primary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    story.append(primary_table)
    story.append(Spacer(1, 0.2 * 72))
    
    # Top 5 Careers
    story.append(Paragraph("Top 5 Career Recommendations", heading_style))
    career_data = [['Rank', 'Career', 'Confidence']]
    for i, (career, prob) in enumerate(prediction_result['top_5_careers'], 1):
        career_data.append([str(i), career, f"{prob*100:.1f}%"])
    
    career_table = Table(career_data, colWidths=[1*72, 2.5*72, 1.5*72])
    career_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    story.append(career_table)
    story.append(Spacer(1, 0.3 * 72))
    
    # Skills Summary
    story.append(Paragraph("📊 Skills Summary", heading_style))
    skills_data = [
        ['Skill', 'Level (out of 10)'],
        ['Programming', str(round(profile['programming_skills'], 1))],
        ['Data Science', str(round(profile['data_science_skills'], 1))],
        ['Problem Solving', str(round(profile['problem_solving'], 1))],
        ['Communication', str(round(profile['communication'], 1))],
        ['Leadership', str(round(profile['leadership'], 1))],
    ]
    skills_table = Table(skills_data, colWidths=[3*72, 2*72])
    skills_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    story.append(skills_table)
    story.append(Spacer(1, 0.2 * 72))
    
    # Recommendations
    story.append(Paragraph("💡 Personalized Recommendations", heading_style))
    recommendations = generate_recommendations(profile, prediction_result)
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 0.1 * 72))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer.getvalue()


def main():
    """Main Streamlit application."""
    
    # Sidebar navigation
    st.sidebar.title("🎓 Career Prediction System")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔮 Predict Career", "📊 Analytics", "📈 Model Info", "❓ About"],
        key="page_nav"
    )
    
    # Load model and data
    model, df, feature_names = load_model_and_data()
    
    if page == "🏠 Home":
        st.title("🎓 Student Career Prediction System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Welcome to Your Career Journey! 🚀
            
            This intelligent system uses advanced machine learning to help you discover
            the most suitable career path based on your:
            
            - 📚 **Academic Performance** (GPA, exam scores, attendance)
            - 💻 **Technical Skills** (programming, data science, cloud computing)
            - 🧠 **Soft Skills** (communication, leadership, teamwork)
            - ⭐ **Personality Traits** (analytical, creative, organized)
            - 📁 **Project Experience** (projects, internships, hackathons)
            - 🎯 **Career Interests** (backend, frontend, data, DevOps, security)
            
            **Get Started**: Navigate to the "Predict Career" section in the sidebar
            to complete your profile and receive personalized career recommendations!
            """)
        
        with col2:
            st.markdown("### Quick Stats")
            st.metric("Total Careers", "10")
            st.metric("Input Features", "24")
            st.metric("Training Samples", "500")
            st.metric("Model Accuracy", f"{model.metrics['test_accuracy']:.1%}")
    
    elif page == "🔮 Predict Career":
        st.title("🔮 Career Prediction")
        
        # Get student profile input
        profile = create_student_profile_input()
        
        if st.button("🎯 Predict My Career", key="predict_button", use_container_width=True):
            # Create DataFrame from profile
            profile_df = pd.DataFrame([profile])
            profile_df = profile_df[feature_names]
            
            # Make prediction
            with st.spinner("🔄 Analyzing your profile..."):
                prediction_result = model.predict_career(profile_df)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Top Career", prediction_result['top_career'])
            with col2:
                st.metric("💯 Confidence", f"{prediction_result['confidence']*100:.1f}%")
            with col3:
                st.metric("🏆 Rank", "1st of 10")
            
            st.divider()
            
            # Create visualizations
            create_prediction_visualizations(profile, prediction_result)
            
            st.divider()
            
            # Insights section
            create_insights_section(profile, prediction_result)
            
            st.divider()
            
            # PDF Report Download
            st.markdown("### 📄 Download Report")
            pdf_content = generate_pdf_report(profile, prediction_result, model)
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_content,
                file_name=f"Career_Prediction_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
    
    elif page == "📊 Analytics":
        st.title("📊 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Career Distribution in Dataset")
            
            career_counts = df['target_career'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=career_counts.index,
                    values=career_counts.values,
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])
            
            fig.update_layout(
                height=400,
                title="Students by Career Path",
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Dataset Statistics")
            
            stats_text = f"""
            **Dataset Overview:**
            - Total Students: {len(df)}
            - Total Careers: {df['target_career'].nunique()}
            - Features: 24
            - Time Created: {datetime.now().strftime('%B %d, %Y')}
            
            **Career Breakdown:**
            """
            
            st.markdown(stats_text)
            
            for career, count in career_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"- **{career}**: {count} students ({percentage:.1f}%)")
    
    elif page == "📈 Model Info":
        st.title("📈 Model Information")
        
        st.markdown("### 🤖 Machine Learning Model Details")
        
        model_info = """
        **Model Type:** Gradient Boosting Classifier
        
        **Hyperparameters:**
        - Number of Estimators: 200
        - Learning Rate: 0.1
        - Max Depth: 8
        - Min Samples Split: 5
        - Subsample: 0.8
        
        **Features Used:** 24
        
        **Target Classes:** 10 career categories
        
        **Data Split:** 80% training, 20% testing
        """
        
        st.markdown(model_info)
        
        st.divider()
        
        create_model_metrics_section(model)
    
    elif page == "❓ About":
        st.title("❓ About This System")
        
        st.markdown("""
        ### Project Overview
        
        The **Student Career Prediction System** is an intelligent platform designed to help
        students make informed career decisions by analyzing their academic performance,
        technical skills, soft skills, personality traits, and project experience.
        
        ### How It Works
        
        1. **Data Collection**: Students input their profile information across multiple dimensions
        2. **ML Analysis**: The Gradient Boosting model analyzes patterns and matches careers
        3. **Prediction**: The system provides personalized career recommendations with confidence scores
        4. **Insights**: Detailed recommendations and insights are provided for career development
        5. **Report Generation**: Students can download comprehensive PDF reports
        
        ### Career Paths Supported
        
        1. **Software Engineer** - Full-stack development, backend/frontend specialization
        2. **Data Scientist** - Machine learning, data analysis, statistical modeling
        3. **Product Manager** - Product strategy, roadmapping, team leadership
        4. **DevOps Engineer** - Infrastructure, deployment, cloud management
        5. **UX/UI Designer** - User experience, interface design, prototyping
        6. **Business Analyst** - Requirements gathering, process optimization
        7. **ML Engineer** - Deep learning, model deployment, MLOps
        8. **Cybersecurity Specialist** - Security architecture, penetration testing
        9. **Cloud Architect** - Cloud infrastructure design, scalability
        10. **Technical Writer** - Documentation, technical communication
        
        ### Technologies Used
        
        - **Streamlit**: Interactive web application framework
        - **Scikit-learn**: Machine learning algorithms
        - **Plotly**: Data visualization
        - **Pandas/NumPy**: Data processing
        - **ReportLab**: PDF generation
        
        ### Disclaimer
        
        This system is designed to provide guidance and recommendations based on input data.
        Career selection should also consider personal interests, market trends, and individual circumstances.
        
        ---
        
        **Version:** 1.0.0  
        **Last Updated:** November 2024
        """)


if __name__ == "__main__":
    main()
