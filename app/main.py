from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import warnings
import shap

warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI(title="Risk API")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize global components
model_pipeline = None
explainer = None
feature_names = None
df_sample = None

try:
    # Load core model components
    model_pipeline = joblib.load("models/best_model.joblib")
    feature_names = joblib.load("models/processed_feature_names.joblib")
    
    # Load sample for peer comparison and SHAP background
    df_sample = pd.read_csv("data/lending_club_sample.csv")
    
    # Extract pipeline steps for SHAP
    classifier = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # CRITICAL: Initialize TreeExplainer at runtime to avoid deserialization errors in cloud
    # We use a small representative sample as the background distribution
    background_data = preprocessor.transform(df_sample.dropna().head(100))
    explainer = shap.TreeExplainer(classifier, background_data, feature_perturbation="interventional")
    
    print("Services initialized successfully.")
except Exception as e:
    print(f"Startup Error: {e}")

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    dti: float
    int_rate: float = Field(..., gt=0)
    emp_length: float
    credit_history_length: float
    term: Literal[' 36 months', ' 60 months']
    grade: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G']
    home_ownership: Literal['RENT', 'MORTGAGE', 'OWN', 'ANY']
    purpose: Literal['debt_consolidation', 'credit_card', 'home_improvement', 'other']
    sub_grade: str = "C1"
    verification_status: str = "Verified"
    open_acc: float = 10.0
    pub_rec: float = 0.0
    revol_bal: float = 15000.0
    revol_util: float = 50.0
    total_acc: float = 25.0
    initial_list_status: str = "w"
    application_type: str = "Individual"
    mort_acc: float = 2.0
    pub_rec_bankruptcies: float = 0.0

FEATURE_ORDER = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'credit_history_length',
    'loan_to_income_ratio', 'interest_to_income_ratio', 'revol_util_to_open_acc'
]

def calculate_amortization(principal, annual_rate, term_months):
    rate = (annual_rate / 100) / 12
    if rate <= 0: return [], 0, principal
    
    pmt = (principal * rate * (1 + rate)**term_months) / ((1 + rate)**term_months - 1)
    schedule = []
    bal = principal
    total_int = 0
    
    for i in range(1, term_months + 1):
        interest_charge = bal * rate
        total_int += interest_charge
        princ_paid = pmt - interest_charge
        bal -= princ_paid
        schedule.append({
            "month": i, "payment": round(pmt, 2), "principal": round(princ_paid, 2), 
            "interest": round(interest_charge, 2), "balance": max(0, round(bal, 2))
        })
    return schedule, round(total_int, 2), round(pmt * term_months, 2)

def get_comparison_metrics(grade: str):
    if df_sample is None: return None
    
    subset = df_sample[df_sample['grade'] == grade].dropna()
    if len(subset) < 10: subset = df_sample.dropna()
    
    subset['loan_to_income'] = subset['loan_amnt'] / (subset['annual_inc'] + 1)
    subset['credit_history_length'] = (pd.to_datetime(subset['issue_d'], errors='coerce') - pd.to_datetime(subset['earliest_cr_line'], errors='coerce')).dt.days / 365.25
    
    success = subset[subset['loan_status'] == 'Fully Paid']
    failure = subset[subset['loan_status'] == 'Charged Off']
    
    if len(success) == 0 or len(failure) == 0: return None
        
    return {
        "labels": ["DTI", "Loan/Income", "Credit History"],
        "successful_avg": [round(success['dti'].mean(), 2), round(success['loan_to_income'].mean(), 2), round(success['credit_history_length'].mean(), 2)],
        "failed_avg": [round(failure['dti'].mean(), 2), round(failure['loan_to_income'].mean(), 2), round(failure['credit_history_length'].mean(), 2)]
    }

def find_optimization_targets(input_data: dict, threshold: float = 20.0):
    suggestions = []
    
    # Interest rate check
    for i in range(1, 10):
        test_data = input_data.copy()
        test_data['int_rate'] -= (i * 0.5)
        if test_data['int_rate'] < 5: break
        
        r, n = (test_data['int_rate'] / 100) / 12, int(test_data['term'].strip().split(" ")[0])
        test_data['installment'] = (test_data['loan_amnt'] * r * (1+r)**n) / ((1+r)**n - 1) if r > 0 else test_data['loan_amnt']/n
        
        prob = model_pipeline.predict_proba(pd.DataFrame([test_data], columns=FEATURE_ORDER))[0][1]
        if prob * 100 < threshold:
            suggestions.append(f"Lower interest rate to {test_data['int_rate']:.1f}%")
            break
            
    # Loan amount check
    for i in range(1, 10):
        test_data = input_data.copy()
        test_data['loan_amnt'] *= (1 - (i * 0.1))
        if test_data['loan_amnt'] < 1000: break
        
        test_data['loan_to_income_ratio'] = test_data['loan_amnt'] / (test_data['annual_inc'] + 1)
        r, n = (test_data['int_rate'] / 100) / 12, int(test_data['term'].strip().split(" ")[0])
        test_data['installment'] = (test_data['loan_amnt'] * r * (1+r)**n) / ((1+r)**n - 1) if r > 0 else test_data['loan_amnt']/n
        
        prob = model_pipeline.predict_proba(pd.DataFrame([test_data], columns=FEATURE_ORDER))[0][1]
        if prob * 100 < threshold:
            suggestions.append(f"Reduce loan amount to ~${int(test_data['loan_amnt']/1000)*1000:,}")
            break
            
    return suggestions

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict(app_in: LoanApplication):
    if not all([model_pipeline, explainer]):
        return JSONResponse(status_code=503, content={"error": "Prediction service offline"})

    data = app_in.model_dump()
    months = int(app_in.term.strip().split(" ")[0])
    r, p, n = (app_in.int_rate / 100) / 12, app_in.loan_amnt, months
    
    data['installment'] = (p * r * (1+r)**n) / ((1+r)**n - 1) if r > 0 else p/n
    data['loan_to_income_ratio'] = p / (app_in.annual_inc + 1)
    data['interest_to_income_ratio'] = (data['installment'] * 12) / (app_in.annual_inc + 1)
    data['revol_util_to_open_acc'] = app_in.revol_util / (app_in.open_acc + 1)
    
    df_in = pd.DataFrame([data], columns=FEATURE_ORDER)
    proba = model_pipeline.predict_proba(df_in)[0][1]
    
    factors = []
    try:
        processed = model_pipeline.named_steps['preprocessor'].transform(df_in)
        shap_vals = explainer.shap_values(processed)
        # Structural check for SHAP output compatibility
        contribution = shap_vals[1][0] if (isinstance(shap_vals, list) and len(shap_vals) > 1) else (shap_vals[0] if not isinstance(shap_vals, list) else shap_vals[0][0])
        
        impact = pd.DataFrame({'feature': feature_names, 'shap': contribution, 'val': processed[0]})
        top = impact[impact['val'] != 0].copy()
        top['abs'] = np.abs(top['shap'])
        
        for _, row in top.sort_values(by='abs', ascending=False).head(3).iterrows():
            fname = row['feature'].split('__')[1] if '__' in row['feature'] else row['feature']
            factors.append({"feature": fname, "impact": "increases" if row['shap'] > 0 else "decreases"})
    except:
        if data.get('int_rate', 0) > 15: factors.append({"feature": "interest_rate", "impact": "increases"})
        if not factors: factors.append({"feature": "credit_factors", "impact": "neutral"})
    
    schedule, total_int, total_pmt = calculate_amortization(p, app_in.int_rate, months)
    risk_score = proba * 100
    
    if risk_score > 50: label, cls = "High Risk", "high-risk"
    elif risk_score > 20: label, cls = "Medium Risk", "medium-risk"
    else: label, cls = "Low Risk", "low-risk"

    return {
        "risk_score": risk_score, "recommendation": label, "risk_class": cls, "explanation": factors,
        "amortization": {"schedule": schedule[:12], "total_interest": total_int, "total_paid": total_pmt},
        "peer_comparison": get_comparison_metrics(app_in.grade),
        "breakeven_analysis": find_optimization_targets(data) if risk_score >= 20 else []
    }