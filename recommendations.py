"""
Career Recommendations Engine
=============================
Generates personalized career development recommendations
based on student profiles and prediction results.
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Career-specific recommendations
CAREER_RECOMMENDATIONS = {
    'Software Engineer': {
        'description': 'Develops software applications, systems, and tools using programming languages and frameworks.',
        'skills_to_develop': [
            'Advanced programming in Python, Java, or C++',
            'System design and architecture patterns',
            'Git and version control mastery',
            'RESTful API development',
            'Database optimization'
        ],
        'learning_path': [
            'Complete Leetcode/HackerRank challenges (Easy → Hard)',
            'Build 3-5 complete portfolio projects',
            'Learn microservices architecture',
            'Study design patterns in depth',
            'Contribute to open-source projects'
        ],
        'tools': ['VS Code', 'Git', 'Docker', 'Jenkins', 'AWS/GCP'],
        'certifications': ['AWS Developer Associate', 'Google Cloud Associate Cloud Engineer']
    },
    'Data Scientist': {
        'description': 'Analyzes complex datasets to help organizations make data-driven decisions.',
        'skills_to_develop': [
            'Advanced statistics and probability',
            'Machine learning algorithms',
            'Python/R programming expertise',
            'SQL and database querying',
            'Data visualization'
        ],
        'learning_path': [
            'Master statistics fundamentals',
            'Complete ML specialization (Andrew Ng)',
            'Build 5+ end-to-end ML projects',
            'Learn deep learning frameworks',
            'Participate in Kaggle competitions'
        ],
        'tools': ['Python', 'R', 'SQL', 'Jupyter', 'TensorFlow', 'PyTorch'],
        'certifications': ['Google Cloud Data Engineer', 'Microsoft Azure Data Scientist']
    },
    'Product Manager': {
        'description': 'Guides product vision, strategy, and execution across the product lifecycle.',
        'skills_to_develop': [
            'Stakeholder management',
            'Data analysis and metrics',
            'User research and empathy',
            'Strategic thinking',
            'Communication and presentation'
        ],
        'learning_path': [
            'Study product management frameworks',
            'Conduct user interviews and research',
            'Create product roadmaps and PRDs',
            'Learn analytics and metrics',
            'Lead cross-functional projects'
        ],
        'tools': ['Jira', 'Figma', 'Analytics tools', 'SQL', 'Mixpanel'],
        'certifications': ['PRAGMATIC Marketing', 'Reforge Product Management']
    },
    'DevOps Engineer': {
        'description': 'Manages infrastructure, deployment pipelines, and system reliability.',
        'skills_to_develop': [
            'Cloud platforms (AWS/Azure/GCP)',
            'Infrastructure as Code (Terraform)',
            'Container orchestration (Kubernetes)',
            'CI/CD pipeline management',
            'Linux system administration'
        ],
        'learning_path': [
            'Master Linux and command line',
            'Learn Docker and containerization',
            'Study Kubernetes deeply',
            'Build CI/CD pipelines',
            'Practice infrastructure automation'
        ],
        'tools': ['Docker', 'Kubernetes', 'Terraform', 'Jenkins', 'Git', 'AWS'],
        'certifications': ['AWS Solutions Architect', 'Kubernetes Administrator (CKA)']
    },
    'UX/UI Designer': {
        'description': 'Creates intuitive and beautiful user interfaces and experiences.',
        'skills_to_develop': [
            'Design principles and theory',
            'Prototyping and wireframing',
            'User research methods',
            'Design tools proficiency',
            'Frontend HTML/CSS basics'
        ],
        'learning_path': [
            'Study design fundamentals',
            'Master Figma or Adobe XD',
            'Practice wireframing and prototyping',
            'Conduct user research',
            'Build complete design systems'
        ],
        'tools': ['Figma', 'Adobe XD', 'Sketch', 'Protopie', 'HTML/CSS'],
        'certifications': ['Google UX Design Certificate', 'Interaction Design Foundation']
    },
    'Business Analyst': {
        'description': 'Analyzes business needs and translates them into technical requirements.',
        'skills_to_develop': [
            'Requirements gathering',
            'Data analysis',
            'Process mapping',
            'SQL and databases',
            'Communication skills'
        ],
        'learning_path': [
            'Learn business analysis frameworks',
            'Master SQL and databases',
            'Study process improvement',
            'Practice requirements documentation',
            'Learn analytics tools'
        ],
        'tools': ['SQL', 'Excel/Power BI', 'Visio', 'JIRA', 'Tableau'],
        'certifications': ['CBAP (Certified Business Analyst)', 'Reforge Analytics']
    },
    'ML Engineer': {
        'description': 'Designs, builds, and deploys machine learning solutions at scale.',
        'skills_to_develop': [
            'Deep learning frameworks',
            'ML model optimization',
            'Feature engineering',
            'MLOps and model deployment',
            'Advanced mathematics'
        ],
        'learning_path': [
            'Master statistics and linear algebra',
            'Study deep learning architectures',
            'Learn TensorFlow/PyTorch in depth',
            'Build production ML systems',
            'Study MLOps practices'
        ],
        'tools': ['Python', 'TensorFlow', 'PyTorch', 'Jupyter', 'MLflow', 'Kubernetes'],
        'certifications': ['Fast.ai Deep Learning', 'DeepLearning.AI Specializations']
    },
    'Cybersecurity Specialist': {
        'description': 'Protects organizations from cyber threats and vulnerabilities.',
        'skills_to_develop': [
            'Network security',
            'Penetration testing',
            'Security protocols and standards',
            'Incident response',
            'Compliance and regulations'
        ],
        'learning_path': [
            'Learn networking fundamentals',
            'Study security protocols',
            'Practice penetration testing',
            'Learn incident response',
            'Study compliance frameworks'
        ],
        'tools': ['Linux', 'Wireshark', 'Metasploit', 'Nmap', 'Burp Suite'],
        'certifications': ['CompTIA Security+', 'Certified Ethical Hacker (CEH)']
    },
    'Cloud Architect': {
        'description': 'Designs and implements cloud infrastructure solutions for scalability and reliability.',
        'skills_to_develop': [
            'Cloud platform expertise (AWS/Azure)',
            'Architecture design patterns',
            'Serverless computing',
            'Disaster recovery planning',
            'Cost optimization'
        ],
        'learning_path': [
            'Master cloud platform fundamentals',
            'Study architecture patterns',
            'Design complex cloud systems',
            'Learn disaster recovery',
            'Practice cost optimization'
        ],
        'tools': ['AWS', 'Azure', 'Terraform', 'CloudFormation', 'Kubernetes'],
        'certifications': ['AWS Solutions Architect Professional', 'Azure Solutions Architect']
    },
    'Technical Writer': {
        'description': 'Creates clear and comprehensive technical documentation for products and services.',
        'skills_to_develop': [
            'Technical writing best practices',
            'Documentation tools mastery',
            'Basic technical understanding',
            'API documentation',
            'Information architecture'
        ],
        'learning_path': [
            'Study technical writing principles',
            'Learn documentation tools (Sphinx, MkDocs)',
            'Study API documentation',
            'Practice creating user guides',
            'Learn information design'
        ],
        'tools': ['Markdown', 'Sphinx', 'MkDocs', 'Confluence', 'Git'],
        'certifications': ['Society for Technical Communication (STC)', 'Write the Docs']
    }
}


class RecommendationEngine:
    """
    Generates personalized career recommendations based on student profile
    and prediction results.
    """

    @staticmethod
    def get_career_overview(career_name):
        """
        Get overview information about a specific career.
        
        Args:
            career_name (str): Name of the career
            
        Returns:
            dict: Career information
        """
        return CAREER_RECOMMENDATIONS.get(career_name, {})

    @staticmethod
    def generate_strengths_analysis(student_data, top_career):
        """
        Analyze student's strengths for their predicted career.
        
        Args:
            student_data (dict): Student profile data
            top_career (str): Predicted top career
            
        Returns:
            dict: Strengths analysis
        """
        strengths = []
        
        # Analyze based on career type
        if top_career == 'Software Engineer':
            if student_data.get('programming_skills', 0) > 7:
                strengths.append('Excellent programming foundation')
            if student_data.get('problem_solving', 0) > 7:
                strengths.append('Strong problem-solving ability')
            if student_data.get('projects_completed', 0) > 5:
                strengths.append('Solid project portfolio')
        
        elif top_career == 'Data Scientist':
            if student_data.get('analytical', 0) > 3.5:
                strengths.append('Strong analytical mindset')
            if student_data.get('data_science_skills', 0) > 7:
                strengths.append('Solid data science foundation')
            if student_data.get('problem_solving', 0) > 7:
                strengths.append('Excellent problem decomposition skills')
        
        elif top_career == 'Product Manager':
            if student_data.get('communication', 0) > 7:
                strengths.append('Excellent communication skills')
            if student_data.get('leadership', 0) > 7:
                strengths.append('Strong leadership potential')
            if student_data.get('teamwork', 0) > 7:
                strengths.append('Great collaborative abilities')
        
        elif top_career == 'DevOps Engineer':
            if student_data.get('organized', 0) > 3.5:
                strengths.append('Well-organized and detail-oriented')
            if student_data.get('cloud_skills', 0) > 6:
                strengths.append('Good cloud infrastructure knowledge')
            if student_data.get('problem_solving', 0) > 7:
                strengths.append('Strong troubleshooting skills')
        
        elif top_career == 'UX/UI Designer':
            if student_data.get('creativity', 0) > 3.5:
                strengths.append('Strong creative thinking')
            if student_data.get('social', 0) > 3.5:
                strengths.append('Good user empathy and communication')
            if student_data.get('communication', 0) > 7:
                strengths.append('Clear presentation abilities')
        
        elif top_career == 'ML Engineer':
            if student_data.get('data_science_skills', 0) > 8:
                strengths.append('Advanced ML knowledge')
            if student_data.get('programming_skills', 0) > 8:
                strengths.append('Expert-level programming skills')
            if student_data.get('analytical', 0) > 4:
                strengths.append('Exceptional analytical mindset')
        
        return {
            'strengths': strengths if strengths else ['Building expertise in this field'],
            'count': len(strengths)
        }

    @staticmethod
    def generate_improvement_areas(student_data, top_career):
        """
        Identify areas for improvement.
        
        Args:
            student_data (dict): Student profile data
            top_career (str): Predicted top career
            
        Returns:
            dict: Improvement recommendations
        """
        improvements = []
        
        if top_career == 'Software Engineer':
            if student_data.get('programming_skills', 0) < 7:
                improvements.append('Strengthen core programming skills')
            if student_data.get('projects_completed', 0) < 5:
                improvements.append('Build more portfolio projects')
            if student_data.get('github_contributions', 0) < 100:
                improvements.append('Increase open-source contributions')
        
        elif top_career == 'Data Scientist':
            if student_data.get('data_science_skills', 0) < 7:
                improvements.append('Deepen machine learning expertise')
            if student_data.get('analytical', 0) < 3.5:
                improvements.append('Enhance analytical thinking')
            if student_data.get('gpa', 0) < 3.5:
                improvements.append('Focus on mathematics and statistics coursework')
        
        elif top_career == 'Product Manager':
            if student_data.get('leadership', 0) < 7:
                improvements.append('Develop leadership capabilities')
            if student_data.get('communication', 0) < 7:
                improvements.append('Improve presentation and communication')
            if student_data.get('internships', 0) < 1:
                improvements.append('Seek product management internships')
        
        return {
            'areas': improvements if improvements else ['You\'re well-positioned for this role'],
            'priority': 'High' if len(improvements) >= 2 else 'Medium' if len(improvements) == 1 else 'Low'
        }

    @staticmethod
    def generate_action_plan(student_data, top_career, career_info):
        """
        Generate a concrete action plan.
        
        Args:
            student_data (dict): Student profile data
            top_career (str): Predicted top career
            career_info (dict): Career information from CAREER_RECOMMENDATIONS
            
        Returns:
            dict: Action plan
        """
        return {
            'immediate_actions': [
                'Review the career description and required skills',
                'Identify current skill gaps',
                'Follow the recommended learning path',
                'Start building portfolio projects'
            ],
            'short_term_goals': career_info.get('learning_path', [])[:2],
            'medium_term_goals': career_info.get('learning_path', [])[2:4],
            'long_term_goals': [
                'Earn relevant certification',
                'Build strong portfolio',
                'Network with professionals in the field',
                'Aim for internship or entry-level position'
            ]
        }


<automated_v0_instructions_reminder>

- Context Gathering
  - Use Parallel Tool Calls Where Possible
  - Don't stop at first match - When searching finds multiple files or components, examine ALL of them to ensure you find the right variant/version
  - Understand the full system before changes - Check existing patterns, parent components, utilities, schemas, and architecture to see if solutions already exist
  - Search systematically - Use a broad → specific → verify relationships approach to fully understand how your changes fit into the broader codebase architecture

- You always Edit/Write to these files using ```

```python file="" isHidden
