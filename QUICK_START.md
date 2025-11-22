# Quick Start Guide

Get the Student Career Prediction System running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip

## Installation Steps

### 1. Set Up Environment

\`\`\`bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
\`\`\`

### 2. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Run Application

\`\`\`bash
python run.py
\`\`\`

Or directly:

\`\`\`bash
streamlit run app.py
\`\`\`

## Access the App

Open your browser to: **http://localhost:8501**

## First Run Checklist

- [ ] App loads successfully
- [ ] Home page displays with system stats
- [ ] Can navigate through all pages
- [ ] Predict Career page form loads all fields
- [ ] Can make a prediction
- [ ] Visualizations display correctly
- [ ] PDF download works
- [ ] Model Info shows metrics
- [ ] Analytics shows dataset stats

## Troubleshooting First Run

### "Module not found" error

\`\`\`bash
pip install -r requirements.txt --upgrade
\`\`\`

### Data file not found

The system will auto-generate synthetic data on first run. Wait for the "Loading model and data..." message.

### Slow on first run

Model training happens first time. Subsequent runs use cache (much faster).

### Port 8501 already in use

\`\`\`bash
streamlit run app.py --server.port 8502
\`\`\`

## Next Steps

1. Fill in your profile in "Predict Career"
2. Get your top 5 career matches
3. Download your PDF report
4. Explore "Model Info" to understand predictions
5. Check "Analytics" for dataset insights

## Customization

To modify the system:

1. **Change careers**: Edit `CAREERS` in `config.py`
2. **Add features**: Modify `data_generator.py` and input forms in `app.py`
3. **Tune model**: Adjust `MODEL_PARAMS` in `config.py`
4. **Customize colors**: Update `COLORS` in `config.py` and CSS in `app.py`

## Support

- Check README.md for full documentation
- Review inline code comments
- Run `streamlit run app.py -- --logger.level=debug` for debug info

Happy predicting!
