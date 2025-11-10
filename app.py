# app.py — MODIFIED FOR FAST FEATURE IMPORTANCE PLOTTING
import streamlit as st
import pandas as pd
import joblib
# NOTE: SHAP is no longer needed but kept in imports if other parts of the app rely on it
import shap 
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

# ---------- CONFIG ----------
MODEL_FILE = "models.pkl"      
TARGET_COL = "loan_status"     
MAX_DISPLAY_ROWS = 500         
# SHAP-related configs are now irrelevant but kept for structure
MAX_SHAP_ROWS = 500            
SHAP_BACKGROUND = 100          

st.set_page_config(page_title="ML Deployment (5 models) w/ Fast Plot", layout="wide")
st.title("🚀 ML Deployment — Loan Prediction")

# ---------- load pipelines dict ----------
@st.cache_resource
def load_models(path: str = MODEL_FILE):
    return joblib.load(path)

try:
    models = load_models(MODEL_FILE)
except Exception as e:
    st.error(f"Failed to load '{MODEL_FILE}': {e}")
    st.stop()

model_names = list(models.keys())

# ---------- sidebar ----------
st.sidebar.header("Model selection & Upload")
# The select box now chooses the model for the FAST feature plot
selected_model_name = st.sidebar.selectbox("Choose model for Feature Plot & detailed view", model_names)
uploaded_file = st.sidebar.file_uploader("Upload CSV (same columns used at training)", type=["csv"])
st.sidebar.markdown("**Loaded models:**")
st.sidebar.write(", ".join(model_names))

# ---------- helpers ----------
def detect_steps(pipe) -> Tuple[object, object]:
    """Return (preprocess_step, clf_step) by looking into pipeline steps."""
    pre = None
    clf = None
    if hasattr(pipe, "named_steps"):
        for name, step in pipe.named_steps.items():
            if hasattr(step, "predict"):
                clf = step
            elif hasattr(step, "transform") and not hasattr(step, "predict"):
                pre = step
    if (pre is None or clf is None) and hasattr(pipe, "steps"):
        for name, step in pipe.steps:
            if hasattr(step, "predict"):
                clf = step
            elif hasattr(step, "transform") and not hasattr(step, "predict"):
                pre = step
    return pre, clf


def safe_preview(df, n=MAX_DISPLAY_ROWS):
    """Display a preview of the DataFrame."""
    st.write(f"Showing first {min(n, len(df))} rows (of {len(df)} total).")
    st.dataframe(df.head(n))


# ---------- main ----------
if uploaded_file is None:
    st.info("📄 Please upload a CSV file to start.")
    st.stop()

# read CSV
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded CSV: {e}")
    st.stop()

st.subheader("📋 Uploaded Data (head)")
st.dataframe(df.head())

# drop target column if present
if TARGET_COL in df.columns:
    st.sidebar.info(f"Dropping target column '{TARGET_COL}' before predictions.")
    df_input = df.drop(columns=[TARGET_COL])
else:
    df_input = df.copy()

# ---------- Predictions for ALL models ----------
st.subheader("🔮 Predictions from All Models")

pred_df = df_input.copy()
errors = {}

# Iterate and predict with all 8 models
for name, pipe in models.items():
    try:
        preds = pipe.predict(df_input) 
        if len(preds) != len(df_input):
            raise ValueError(f"prediction length {len(preds)} != input length {len(df_input)}")
        pred_df[f"pred_{name}"] = preds 
    except Exception as e:
        errors[name] = str(e)
        pred_df[f"pred_{name}"] = [None] * len(df_input)

if errors:
    st.warning("Some models failed to predict. See details in the sidebar.")
    for k, v in errors.items():
        st.sidebar.error(f"{k}: {v}")

# show preview of predictions
safe_preview(pred_df, MAX_DISPLAY_ROWS)

# summary counts for each model
st.subheader("Model Prediction Counts (summary)")
counts = {}
for name in model_names:
    col = f"pred_{name}"
    counts[name] = pred_df[col].value_counts(dropna=True).to_dict()
counts_df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).astype(int)
st.dataframe(counts_df)

# ---------- FAST FEATURE IMPORTANCE PLOT (INSTEAD OF SHAP) ----------
st.subheader(f"🧠 Fast Feature Importance — {selected_model_name}")

selected_pipe = models[selected_model_name]
pre_step, clf_step = detect_steps(selected_pipe)

if pre_step is None or clf_step is None:
    st.error("Could not detect preprocessing or classifier step. Analysis unavailable.")
else:
    try:
        # Use a small sample to get the transformed feature names
        df_sample = df_input.head(10)
        X_trans = pre_step.transform(df_sample)
        
        feature_names = None
        try:
            if hasattr(pre_step, "get_feature_names_out"):
                feature_names = pre_step.get_feature_names_out(df_sample.columns)
        except Exception:
            # Fallback if get_feature_names_out fails
            feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]


        def basic_feature_importance_analysis(model, feature_names):
            """Generates instant feature importance plot based on model type."""
            model_name = type(model).__name__
            st.write(f"🔍 Analyzing Model: `{model_name}`")
            
            importances = None
            title = ""
            
            # For tree models (Random Forest, XGBoost, etc.)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                title = "Feature Importance (Gini/Gain)"
            
            # For linear models (Logistic Regression, SVM)
            elif hasattr(model, 'coef_'):
                coefs_raw = model.coef_
                # Assume binary classification, take absolute value of coefficients for magnitude
                if isinstance(coefs_raw, np.ndarray) and coefs_raw.ndim > 1:
                    # For multi-class (rare for loan status), we would average, but for binary [0], take abs.
                    importances = np.abs(coefs_raw[0]) 
                else:
                    importances = np.abs(coefs_raw)
                title = "Feature Coefficients (Absolute Magnitude)"
            
            if importances is None:
                st.warning(f"Feature importance not directly available for {model_name}. No plot generated.")
                return

            # Plotting the top features
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True) # Ascending for correct bar plot order

            fig, ax = plt.subplots(figsize=(10, 6))
            # Take Top 15 features
            top_15 = feat_imp_df.tail(15)
            ax.barh(top_15['Feature'], top_15['Importance'])
            ax.set_xlabel(title)
            ax.set_title(f"Top 15 Features for {model_name}")
            plt.tight_layout()
            st.pyplot(fig)

        # Call the faster analysis function
        basic_feature_importance_analysis(clf_step, feature_names)

    except Exception as e:
        st.error(f"❌ Fast analysis failed: {e}")

st.write("✅ Done. Using model-specific feature importance for fast results.")