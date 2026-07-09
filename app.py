# ============================================================
# app.py — Render.com Deployment Version
# Explainable Ensemble ML for Child Malnutrition Prediction
# Model: Stacking Ensemble (GB + LightGBM + MLP + CatBoost)
# ============================================================

import os
import gradio as gr
import pickle
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments


# ============================================================
# STEP 1: Load All Saved Files (runs once at startup)
# ============================================================

print("🔄 Loading model and artifacts...")

with open('stacking_model.pkl', 'rb') as f:
    stacking_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('target_encoder.pkl', 'rb') as f:
    le_target = pickle.load(f)

with open('num_imputer.pkl', 'rb') as f:
    num_imputer = pickle.load(f)

with open('feature_metadata.json', 'r') as f:
    metadata = json.load(f)

FEATURES              = metadata['FEATURES']
NUMERIC_FEATURES      = metadata['NUMERIC_FEATURES']
CATEGORICAL_FEATURES  = metadata['CATEGORICAL_FEATURES']
CLASS_NAMES           = metadata['CLASS_NAMES']
CATEGORICAL_OPTIONS   = metadata['CATEGORICAL_OPTIONS']

X_background = pd.read_csv('shap_background.csv')

print("✅ Model and artifacts loaded successfully!")
print(f"   Features: {FEATURES}")
print(f"   Classes: {CLASS_NAMES}")


# ============================================================
# STEP 2: Build SHAP Explainer Once (avoids rebuilding per request)
# ============================================================

def stacking_predict_proba(X):
    if hasattr(X, 'values'):
        X_arr = X.values
    else:
        X_arr = np.array(X)
    X_df = pd.DataFrame(X_arr, columns=FEATURES)
    return stacking_model.predict_proba(X_df)

print("🔄 Building SHAP PermutationExplainer...")
explainer = shap.PermutationExplainer(
    model=stacking_predict_proba,
    masker=X_background,
    output_names=CLASS_NAMES
)

mean_base_values = np.mean(stacking_predict_proba(X_background), axis=0)
print("✅ SHAP explainer ready!")


# ============================================================
# STEP 3: Core Prediction + Explanation Function
# ============================================================

def predict_malnutrition(
    child_age_months, child_sex,
    breastfeeding, mother_age, mother_height_cm, mother_education,
    father_occupation, father_education,
    wealth_index, residence_type, child_weight_kg, child_height_cm
):
    try:
        raw_input = {
            'child_age_months':  child_age_months,
            'child_sex':         child_sex,
            'breastfeeding':     breastfeeding,
            'mother_age':        mother_age,
            'mother_height_cm':  mother_height_cm,
            'mother_education':  mother_education,
            'father_occupation': father_occupation,
            'father_education':  father_education,
            'wealth_index':      wealth_index,
            'residence_type':    residence_type,
            'child_weight_kg':   child_weight_kg,
            'child_height_cm':   child_height_cm,
        }

        input_df = pd.DataFrame([raw_input])

        for col in CATEGORICAL_FEATURES:
            le = label_encoders[col]
            val = str(input_df[col].values[0])
            if val in le.classes_:
                input_df[col] = le.transform([val])[0]
            else:
                input_df[col] = 0

        input_df = input_df[FEATURES].astype(float)

        pred_encoded = stacking_model.predict(input_df)[0]
        pred_proba   = stacking_model.predict_proba(input_df)[0]
        pred_label   = le_target.classes_[pred_encoded]

        proba_dict = {
            cls: float(prob)
            for cls, prob in zip(le_target.classes_, pred_proba)
        }

        shap_result  = explainer(input_df)
        shap_vals_3d = shap_result.values

        if shap_vals_3d.ndim == 3:
            shap_for_pred = shap_vals_3d[0, :, pred_encoded]
        else:
            shap_for_pred = shap_vals_3d[0]

        feature_impact = pd.DataFrame({
            'Feature':     FEATURES,
            'SHAP Value':  shap_for_pred,
            'Input Value': [raw_input[f] for f in FEATURES]
        })
        feature_impact['Abs SHAP'] = feature_impact['SHAP Value'].abs()
        feature_impact = feature_impact.sort_values('Abs SHAP', ascending=False)

        top_drivers   = feature_impact.head(3)
        driver_labels = ['Primary Driver', 'Secondary Driver', 'Tertiary Driver']

        report_lines = [
            f"### 🔍 Explainable AI (XAI) Report\n",
            f"**Prediction: {pred_label}**\n",
            f"The model analyzed your input and found:\n"
        ]

        for i, (_, row) in enumerate(top_drivers.iterrows()):
            direction = "increases" if row['SHAP Value'] > 0 else "decreases"
            report_lines.append(
                f"{i+1}. **{driver_labels[i]}**: `{row['Feature']}` "
                f"(value: {row['Input Value']}) — {direction} risk of {pred_label}"
            )

        top1 = top_drivers.iloc[0]
        top2 = top_drivers.iloc[1]
        report_lines.append(
            f"\n**Scientific Logic:** The combination of "
            f"`{top1['Feature']}={top1['Input Value']}` and "
            f"`{top2['Feature']}={top2['Input Value']}` strongly correlates "
            f"with the model's prediction of **{pred_label}**."
        )

        xai_text = "\n".join(report_lines)

        base_val = float(mean_base_values[pred_encoded])

        fig = plt.figure(figsize=(9, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_for_pred,
                base_values=base_val,
                data=input_df.values[0],
                feature_names=FEATURES
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall — Prediction: {pred_label}', fontsize=11)
        plt.tight_layout()

        return proba_dict, xai_text, fig

    except Exception as e:
        error_dict = {"Error": 1.0}
        error_text = f"❌ Error during prediction: {str(e)}"
        fig = plt.figure()
        return error_dict, error_text, fig


# ============================================================
# STEP 4: Build Gradio Interface
# ============================================================

with gr.Blocks(title="Child Malnutrition Predictor") as demo:

    gr.Markdown("""
    # 🍼 Explainable AI — Child Malnutrition Risk Predictor
    ### Multi-Class Prediction: Normal / Stunted / Acute Malnourished
    Powered by a Stacking Ensemble (Gradient Boosting + LightGBM + MLP + CatBoost)
    trained on BDHS 2022 data, with SHAP-based explainability.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👶 Child Information")
            child_age_months = gr.Number(value=18, label="Child Age (months)", minimum=0, maximum=59)
            child_weight_kg   = gr.Number(value=10.5, label="Child Weight (kg)")
            child_height_cm   = gr.Number(value=78.0, label="Child Height (cm)")
            child_sex         = gr.Dropdown(CATEGORICAL_OPTIONS['child_sex'], value='Male', label="Child Sex")

            gr.Markdown("### 🍼 Feeding")
            breastfeeding = gr.Dropdown(CATEGORICAL_OPTIONS['breastfeeding'], value='Yes', label="Currently Breastfeeding")

            gr.Markdown("### 👩 Maternal Information")
            mother_age        = gr.Number(value=25, label="Mother Age (years)")
            mother_height_cm  = gr.Number(value=152.0, label="Mother Height (cm)")
            mother_education  = gr.Dropdown(CATEGORICAL_OPTIONS['mother_education'], value='Secondary', label="Mother Education")

        with gr.Column():
            gr.Markdown("### 👨 Paternal Information")
            father_occupation = gr.Dropdown(CATEGORICAL_OPTIONS['father_occupation'], value='Manual', label="Father Occupation")
            father_education  = gr.Dropdown(CATEGORICAL_OPTIONS['father_education'], value='Secondary', label="Father Education")

            gr.Markdown("### 🏠 Household & Location")
            wealth_index    = gr.Dropdown(CATEGORICAL_OPTIONS['wealth_index'], value='Middle', label="Wealth Index")
            residence_type  = gr.Dropdown(CATEGORICAL_OPTIONS['residence_type'], value='Rural', label="Residence Type")

            predict_btn = gr.Button("🔍 Predict & Explain", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            output_label = gr.Label(label="Prediction Probabilities", num_top_classes=3)
            output_text  = gr.Markdown(label="XAI Report")
        with gr.Column():
            output_plot = gr.Plot(label="SHAP Waterfall Explanation")

    predict_btn.click(
        fn=predict_malnutrition,
        inputs=[
            child_age_months, child_sex,
            breastfeeding, mother_age, mother_height_cm, mother_education,
            father_occupation, father_education,
            wealth_index, residence_type, child_weight_kg, child_height_cm
        ],
        outputs=[output_label, output_text, output_plot]
    )

    gr.Markdown("""
    ---
    ⚠️ **Disclaimer**: This tool is for research and educational purposes only.
    It is not a substitute for professional medical diagnosis. Please consult
    a qualified healthcare provider for actual nutritional assessment.
    """)


# ============================================================
# STEP 5: Launch — Render.com Specific Configuration
# ============================================================
# CRITICAL DIFFERENCE FROM HUGGING FACE:
# Render assigns a PORT via environment variable — you MUST
# read it dynamically, and bind to 0.0.0.0 (not 127.0.0.1)
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",   # required — listen on all interfaces
        server_port=port,         # required — use Render's assigned port
        share=False
    )
