# Loan Risk Assessment

A full-stack machine learning application designed to predict the probability of loan default. The system provides real-time risk scoring, feature-driven explanations, and financial planning tools to assist in credit evaluation.

## Core Features

- **Risk Prediction**: Probability-based default assessment using an optimized XGBoost classifier.
- **Impact Analysis**: Live feature importance explanations leveraging SHAP to identify key risk drivers.
- **Financial Planning**: Dynamic amortization schedules and total cost of credit calculations.
- **Peer Benchmarking**: Comparative analysis of applicant metrics against historical distributions.
- **Optimization Suggestions**: Automated identification of threshold-based improvements for interest rates and loan amounts.

## Technology Stack

- **Backend**: FastAPI (Python)
- **Machine Learning**: XGBoost, Scikit-learn, SHAP
- **Frontend**: Vanilla JavaScript, CSS3, HTML5, Chart.js
- **Deployment**: Docker, Render

## Project Structure

- `app/`: FastAPI application, static assets, and HTML templates.
- `models/`: Optimized model pipelines and serialized feature preprocessors.
- `data/`: Sample datasets for peer comparison metrics.
- `notebooks/`: Research and development pipelines for model training.

## Installation and Local Development

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Access the interface at `http://127.0.0.1:8000`.


