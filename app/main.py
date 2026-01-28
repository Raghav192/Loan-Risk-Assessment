from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import shap
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- Global Setup ---
app = FastAPI(title="Financial Risk Prediction API")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Load model components and dataset for analysis at startup
try:
    model_pipeline = joblib.load("models/best_model.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    feature_names = joblib.load("models/processed_feature_names.joblib")
    df_sample = pd.read_csv("data/lending_club_accepted.csv", usecols=['grade', 'loan_status', 'dti', 'loan_amnt', 'annual_inc', 'issue_d', 'earliest_cr_line'], nrows=50000)
    print("INFO:     Model components and data sample loaded successfully.")
except Exception as e:
    model_pipeline, explainer, feature_names, df_sample = None, None, None, None
    print(f"FATAL:   Could not load files. Error: {e}")

# --- Pydantic Models ---
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
    # Default values for fields not on the form
    sub_grade: str = "C1"; verification_status: str = "Verified"; open_acc: float = 10.0; pub_rec: float = 0.0; revol_bal: float = 15000.0
    revol_util: float = 50.0; total_acc: float = 25.0; initial_list_status: str = "w"; application_type: str = "Individual"; mort_acc: float = 2.0
    pub_rec_bankruptcies: float = 0.0

FEATURE_ORDER = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies', 'credit_history_length',
    'loan_to_income_ratio', 'interest_to_income_ratio', 'revol_util_to_open_acc'
]

# --- Helper Functions ---
def calculate_amortization(principal, annual_rate, term_months):
    monthly_rate = (annual_rate / 100) / 12
    if monthly_rate == 0 or (1 + monthly_rate)**term_months == 1: return [], 0, principal
    monthly_payment = (principal * monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
    schedule = []; balance = principal; total_interest = 0
    for i in range(1, term_months + 1):
        interest = balance * monthly_rate
        total_interest += interest
        principal_paid = monthly_payment - interest
        balance -= principal_paid
        if balance < 0: balance = 0
        schedule.append({"month": i, "payment": round(monthly_payment, 2), "principal": round(principal_paid, 2), "interest": round(interest, 2), "balance": round(balance, 2)})
    return schedule, round(total_interest, 2), round(monthly_payment * term_months, 2)

def get_peer_comparison(grade: str):
    if df_sample is None: return None
    peer_group = df_sample[df_sample['grade'] == grade].dropna()
    if len(peer_group) < 10: peer_group = df_sample.dropna()
    peer_group['loan_to_income'] = peer_group['loan_amnt'] / (peer_group['annual_inc'] + 1)
    peer_group['credit_history_length'] = (pd.to_datetime(peer_group['issue_d'], errors='coerce') - pd.to_datetime(peer_group['earliest_cr_line'], errors='coerce')).dt.days / 365.25
    successful_peers = peer_group[peer_group['loan_status'] == 'Fully Paid']
    failed_peers = peer_group[peer_group['loan_status'] == 'Charged Off']
    if len(successful_peers) == 0 or len(failed_peers) == 0: return None
    comparison = {
        "labels": ["DTI", "Loan to Income Ratio", "Credit History (yrs)"],
        "successful_avg": [round(successful_peers['dti'].mean(), 2), round(successful_peers['loan_to_income'].mean(), 2), round(successful_peers['credit_history_length'].mean(), 2)],
        "failed_avg": [round(failed_peers['dti'].mean(), 2), round(failed_peers['loan_to_income'].mean(), 2), round(failed_peers['credit_history_length'].mean(), 2)]
    }
    return comparison

def find_breakeven_points(input_dict: dict, threshold: float = 20.0):
    suggestions = []
    
    # Suggestion 1: Lower Interest Rate
    for i in range(1, 20):
        temp_dict = input_dict.copy()
        new_rate = temp_dict['int_rate'] - (i * 0.5)
        if new_rate < 5: break
        
        temp_dict['int_rate'] = new_rate
        term_months = int(temp_dict['term'].strip().split(" ")[0])
        monthly_rate = (new_rate / 100) / 12
        p, n = temp_dict['loan_amnt'], term_months
        if monthly_rate > 0: temp_dict['installment'] = (p * monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
        else: temp_dict['installment'] = p / n
        temp_df = pd.DataFrame([temp_dict], columns=FEATURE_ORDER)
        
        risk_proba = model_pipeline.predict_proba(temp_df)[0][1]
        if float(risk_proba) * 100 < threshold:
            suggestions.append(f"Lower interest rate to {new_rate:.1f}%")
            break
            
    
    for i in range(1, 20):
        temp_dict = input_dict.copy()
        new_amnt = temp_dict['loan_amnt'] * (1 - (i * 0.05))
        if new_amnt < 1000: break

        temp_dict['loan_amnt'] = new_amnt
        temp_dict['loan_to_income_ratio'] = new_amnt / (temp_dict['annual_inc'] + 1)
        term_months = int(temp_dict['term'].strip().split(" ")[0])
        monthly_rate = (temp_dict['int_rate'] / 100) / 12
        p, n = new_amnt, term_months
        if monthly_rate > 0: temp_dict['installment'] = (p * monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
        else: temp_dict['installment'] = p / n
        temp_df = pd.DataFrame([temp_dict], columns=FEATURE_ORDER)

        risk_proba = model_pipeline.predict_proba(temp_df)[0][1]
        if float(risk_proba) * 100 < threshold:
            suggestions.append(f"Reduce loan amount to ~${int(new_amnt/1000)*1000:,}")
            break

    return suggestions

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def api_predict(application: LoanApplication):
    if model_pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded. Please check server logs."})

    input_dict = application.model_dump()
    
    # Calculate derived features
    term_in_months = int(application.term.strip().split(" ")[0])
    monthly_rate = (application.int_rate / 100) / 12
    p, n = application.loan_amnt, term_in_months
    if monthly_rate > 0:
        input_dict['installment'] = (p * monthly_rate * (1 + monthly_rate)**n) / ((1 + monthly_rate)**n - 1)
    else:
        input_dict['installment'] = p / n
    input_dict['loan_to_income_ratio'] = input_dict['loan_amnt'] / (input_dict['annual_inc'] + 1)
    input_dict['interest_to_income_ratio'] = (input_dict['installment'] * 12) / (input_dict['annual_inc'] + 1)
    input_dict['revol_util_to_open_acc'] = input_dict['revol_util'] / (input_dict['open_acc'] + 1)
    
    input_df = pd.DataFrame([input_dict], columns=FEATURE_ORDER)
    risk_proba = model_pipeline.predict_proba(input_df)[0][1]
    
    explanation_data = []
    try:
        processed_input = model_pipeline.named_steps['preprocessor'].transform(input_df)
        shap_values = explainer.shap_values(processed_input)

        # Handle both single output and multi-class output
        if isinstance(shap_values, list):
            # For binary classification, use class 1 (default/risk class)
            shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            shap_vals = shap_values[0]

        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_vals,
            'feature_value': processed_input[0]
        })
        shap_df_filtered = shap_df[shap_df['feature_value'] != 0].copy()
        shap_df_filtered['abs_shap'] = np.abs(shap_df_filtered['shap_value'])
        top_contributors = shap_df_filtered.sort_values(by='abs_shap', ascending=False).head(3)

        for _, row in top_contributors.iterrows():
            feature_name = row['feature']
            # Handle both prefixed and non-prefixed feature names
            if '__' in feature_name:
                feature_name = feature_name.split('__')[1]
            explanation_data.append({
                "feature": feature_name,
                "impact": "increases" if float(row['shap_value']) > 0 else "decreases"
            })
    except Exception as e:
        print(f"--- SHAP Explanation Generation Error: {e} ---")
        # Provide fallback explanations based on input values
        if float(input_dict.get('int_rate', 0)) > 15:
            explanation_data.append({"feature": "interest_rate", "impact": "increases"})
        if float(input_dict.get('dti', 0)) > 20:
            explanation_data.append({"feature": "debt_to_income", "impact": "increases"})
        if float(input_dict.get('loan_to_income_ratio', 0)) > 0.5:
            explanation_data.append({"feature": "loan_to_income", "impact": "increases"})
        if not explanation_data:
            explanation_data.append({"feature": "Unable to determine specific factors", "impact": "neutral"})
    
    amortization_schedule, total_interest, total_paid = calculate_amortization(application.loan_amnt, application.int_rate, term_in_months)
    peer_data = get_peer_comparison(application.grade)
    
    risk_score = float(risk_proba) * 100
    
    breakeven_suggestions = []
    if risk_score >= 20.0:
        breakeven_suggestions = find_breakeven_points(input_dict)

    if risk_score > 50: recommendation, risk_class = "High Risk - Not Recommended", "high-risk"
    elif risk_score > 20: recommendation, risk_class = "Medium Risk - Manual Review", "medium-risk"
    else: recommendation, risk_class = "Low Risk - Recommended", "low-risk"

    return JSONResponse(content={
        "risk_score": risk_score, "recommendation": recommendation, "risk_class": risk_class,
        "explanation": explanation_data,
        "amortization": {"schedule": amortization_schedule[:12], "total_interest": total_interest, "total_paid": total_paid},
        "peer_comparison": peer_data,
        "breakeven_analysis": breakeven_suggestions
    })