"""
Parkinson's Disease Detection - Streamlit Web Application
Provides a clinical dashboard for clinicians to:
1. Input 22 continuous voice measurements (via bulk pasting or manual sliders/inputs).
2. Receive a predicted probability score of Parkinson's Disease.
3. Visualize the diagnostic risk with a color-coded indicator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# Set page config
st.set_page_config(
    page_title="Parkinson's Risk Assessment Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #FF4B4B, #FF8383);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-weight: 300;
        font-size: 1.2rem;
        color: #7A808A;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #F8F9FB;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E9EBEF;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 1.5rem;
    }
    
    .dark-card {
        background-color: #0E1117;
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #262730;
        margin-bottom: 1.5rem;
    }
    
    .risk-badge {
        font-weight: 600;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        display: inline-block;
        font-size: 1rem;
    }
    
    .risk-low {
        background-color: #D4EDDA;
        color: #155724;
        border: 1px solid #C3E6CB;
    }
    
    .risk-medium {
        background-color: #FFF3CD;
        color: #856404;
        border: 1px solid #FFEBAA;
    }
    
    .risk-high {
        background-color: #F8D7DA;
        color: #721C24;
        border: 1px solid #F5C6CB;
    }
</style>
""", unsafe_allow_html=True)

# Feature metadata (medians, mins, maxs)
FEATURE_METADATA = {
    'Fo_Hz': {'median': 148.79, 'min': 88.333, 'max': 260.105, 'desc': 'Average vocal fundamental frequency (Hz)'},
    'Fhi_Hz': {'median': 175.829, 'min': 102.145, 'max': 592.03, 'desc': 'Maximum vocal fundamental frequency (Hz)'},
    'Flo_Hz': {'median': 104.315, 'min': 65.476, 'max': 239.17, 'desc': 'Minimum vocal fundamental frequency (Hz)'},
    'Jitter_pct': {'median': 0.00494, 'min': 0.00168, 'max': 0.03316, 'desc': 'Percentage of fundamental frequency variation'},
    'Jitter_Abs': {'median': 0.00003, 'min': 0.000007, 'max': 0.00026, 'desc': 'Absolute fundamental frequency variation in seconds'},
    'RAP': {'median': 0.0025, 'min': 0.00068, 'max': 0.02144, 'desc': 'Relative amplitude perturbation (jitter metric)'},
    'PPQ': {'median': 0.00269, 'min': 0.00092, 'max': 0.01958, 'desc': '5-point period perturbation quotient (jitter metric)'},
    'Jitter_DDP': {'median': 0.00749, 'min': 0.00204, 'max': 0.06433, 'desc': 'Average absolute difference of differences between jitter cycles'},
    'Shimmer': {'median': 0.02297, 'min': 0.00954, 'max': 0.11908, 'desc': 'Vocal amplitude variation'},
    'Shimmer_dB': {'median': 0.221, 'min': 0.085, 'max': 1.302, 'desc': 'Vocal amplitude variation in dB'},
    'Shimmer_APQ3': {'median': 0.01279, 'min': 0.00455, 'max': 0.05647, 'desc': '3-point amplitude perturbation quotient (shimmer metric)'},
    'Shimmer_APQ5': {'median': 0.01347, 'min': 0.0057, 'max': 0.0794, 'desc': '5-point amplitude perturbation quotient (shimmer metric)'},
    'APQ': {'median': 0.01826, 'min': 0.00719, 'max': 0.13778, 'desc': '11-point amplitude perturbation quotient (shimmer metric)'},
    'Shimmer_DDA': {'median': 0.03836, 'min': 0.01364, 'max': 0.16942, 'desc': 'Average absolute difference of differences between shimmer cycles'},
    'NHR': {'median': 0.01166, 'min': 0.00065, 'max': 0.31482, 'desc': 'Ratio of noise to tonal components in the voice'},
    'HNR': {'median': 22.085, 'min': 8.441, 'max': 33.047, 'desc': 'Ratio of harmonic components to noise in the voice'},
    'RPDE': {'median': 0.495954, 'min': 0.25657, 'max': 0.685151, 'desc': 'Recurrence period density entropy (nonlinear measure)'},
    'DFA': {'median': 0.722254, 'min': 0.574282, 'max': 0.825288, 'desc': 'Signal fractal scaling exponent (detrended fluctuation analysis)'},
    'spread1': {'median': -5.720868, 'min': -7.964984, 'max': -2.434031, 'desc': 'Nonlinear measure of fundamental frequency variation (spread 1)'},
    'spread2': {'median': 0.218885, 'min': 0.006274, 'max': 0.450493, 'desc': 'Nonlinear measure of fundamental frequency variation (spread 2)'},
    'D2': {'median': 2.361532, 'min': 1.423287, 'max': 3.671155, 'desc': 'Correlation dimension (nonlinear measure)'},
    'PPE': {'median': 0.194052, 'min': 0.044539, 'max': 0.527367, 'desc': 'Pitch period entropy (nonlinear measure)'}
}

# Ordered list of features the model expects (original features only, before engineering)
RAW_FEATURE_LIST = list(FEATURE_METADATA.keys())

def load_pipeline():
    try:
        return joblib.load("models/best_model_pipeline.joblib")
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}. Make sure `train.py` has been run and models are saved.")
        return None

pipeline_data = load_pipeline()

# Title Section
st.markdown('<div class="main-title">🩺 Parkinson\'s Risk Assessment Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A clinical interface for evaluating Parkinson\'s Disease risk based on 22 acoustic voice features.</div>', unsafe_allow_html=True)

if pipeline_data:
    st.sidebar.markdown("### 📊 Active Model Specifications")
    st.sidebar.info(
        f"**Pipeline Method**: {pipeline_data['pipeline_method'].upper()}\n\n"
        f"**Classifier**: {type(pipeline_data['model']).__name__}"
    )
    
    st.sidebar.markdown("### 📖 Feature Guide")
    for key, val in FEATURE_METADATA.items():
        st.sidebar.caption(f"**{key}**: {val['desc']}")

    # Form to input values
    input_values = {}
    
    input_method = st.radio(
        "Select Input Method", 
        ["📋 Bulk Copy-Paste", "🎛️ Manual Parameters"], 
        horizontal=True
    )
    
    parse_error = False
    
    if input_method == "📋 Bulk Copy-Paste":
        st.markdown("#### Paste Acoustic Measurement Vector")
        st.caption("Paste a row of 22 values directly from a spreadsheet (CSV) or medical record, separated by commas, spaces, or tabs.")
        
        # Prepopulate with a healthy row sample for demo
        demo_str = "119.992,157.302,74.997,0.00784,7e-05,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654"
        paste_input = st.text_area("Acoustic Vector input", value=demo_str, height=100, label_visibility="collapsed")
        
        # Parse pasted text
        parsed_tokens = re.split(r'[,\t\s\n]+', paste_input.strip())
        parsed_tokens = [t for t in parsed_tokens if t]
        
        parsed_values = []
        for token in parsed_tokens:
            try:
                parsed_values.append(float(token))
            except ValueError:
                parse_error = True
        
        if len(parsed_values) == 22:
            st.success("✅ Successfully parsed exactly 22 features!")
            for idx, key in enumerate(RAW_FEATURE_LIST):
                input_values[key] = parsed_values[idx]
        else:
            st.warning(f"⚠️ Vector contains {len(parsed_values)} values. Please provide exactly 22 values.")
            parse_error = True
            
    else:
        st.markdown("#### Adjust Parameters Individually")
        st.caption("Sliding values are initialized to the training set median.")
        
        col1, col2 = st.columns(2)
        
        for idx, (key, meta) in enumerate(FEATURE_METADATA.items()):
            col_target = col1 if idx % 2 == 0 else col2
            
            # Format inputs gracefully (min, max, median)
            step_val = (meta['max'] - meta['min']) / 100.0
            
            # Absolute jitter can have very small step
            if key == 'Jitter_Abs':
                step_val = 0.000001
                
            input_values[key] = col_target.slider(
                label=f"{key} - {meta['desc']}",
                min_value=float(meta['min']),
                max_value=float(meta['max']),
                value=float(meta['median']),
                step=float(step_val),
                format="%.6f" if meta['median'] < 0.01 else "%.3f"
            )

    # 2. Prediction execution
    if st.button("Evaluate Diagnostic Risk", type="primary", use_container_width=True):
        if len(input_values) < 22 or (input_method == "📋 Bulk Copy-Paste" and parse_error):
            st.error("Cannot perform evaluation. Please ensure inputs are valid and contain 22 metrics.")
        else:
            # Create Raw input dataframe
            input_df = pd.DataFrame([input_values])
            
            # 3. Dynamic Feature Engineering
            input_df['frequency_range'] = input_df['Fhi_Hz'] - input_df['Flo_Hz']
            
            # Avg Jitter Calculation
            j_cols_to_avg = [c for c in pipeline_data["jitter_cols"] if c != "jitter_avg"]
            input_df['jitter_avg'] = input_df[j_cols_to_avg].mean(axis=1)
            
            # Avg Shimmer Calculation
            s_cols_to_avg = [c for c in pipeline_data["shimmer_cols"] if c != "shimmer_avg"]
            input_df['shimmer_avg'] = input_df[s_cols_to_avg].mean(axis=1)
            
            # Voice to Noise Ratio
            input_df['voice_to_noise'] = input_df['HNR'] / (input_df['NHR'] + 1e-9)
            
            # Align features with order expected by scaler / transforms
            X_input = input_df[pipeline_data["feature_names"]]
            
            # 4. Apply preprocessing pipeline
            method = pipeline_data["pipeline_method"]
            meta = pipeline_data["meta"]
            model = pipeline_data["model"]
            
            if method == "baseline":
                scaler = meta["scaler"]
                X_trans = scaler.transform(X_input)
            elif method == "pca":
                scaler_coll = meta["scaler_coll"]
                pca = meta["pca"]
                scaler_non_coll = meta["scaler_non_coll"]
                collinear_cols = meta["collinear_cols"]
                non_collinear_cols = meta["non_collinear_cols"]
                
                # Transform collinear features
                X_coll_scaled = scaler_coll.transform(X_input[collinear_cols])
                X_coll_pca = pca.transform(X_coll_scaled)
                
                # Transform non-collinear features
                X_non_coll_scaled = scaler_non_coll.transform(X_input[non_collinear_cols])
                
                # Stack them
                X_trans = np.hstack([X_coll_pca, X_non_coll_scaled])
            elif method == "lasso":
                scaler = meta["scaler"]
                selector = meta["selector"]
                X_scaled = scaler.transform(X_input)
                X_trans = selector.transform(X_scaled)
                
            # Perform prediction
            prob = model.predict_proba(X_trans)[0][1]
            risk_percent = prob * 100
            
            # Display results
            st.markdown("### 📊 Risk Assessment Result")
            
            c_left, c_right = st.columns([1, 2])
            
            with c_left:
                if risk_percent < 30.0:
                    badge_style = "risk-low"
                    badge_text = "LOW RISK"
                    box_color = "#D4EDDA"
                elif risk_percent <= 70.0:
                    badge_style = "risk-medium"
                    badge_text = "MODERATE RISK"
                    box_color = "#FFF3CD"
                else:
                    badge_style = "risk-high"
                    badge_text = "HIGH RISK"
                    box_color = "#F8D7DA"
                    
                st.markdown(
                    f'<div style="text-align: center; background-color: {box_color}; border-radius: 12px; padding: 2rem; border: 1px solid rgba(0,0,0,0.05);">'
                    f'<div style="font-size: 1.2rem; color: #555; margin-bottom: 0.5rem;">Estimated Probability</div>'
                    f'<div style="font-size: 4rem; font-weight: 800; line-height: 1; margin-bottom: 1rem;">{risk_percent:.1f}%</div>'
                    f'<div class="risk-badge {badge_style}">{badge_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with c_right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Clinical Interpretation Note")
                if risk_percent < 30.0:
                    st.write("Acoustic parameters indicate voice characteristics well within normal variations. Probability of Parkinson's Disease is low. Please proceed with standard diagnostic protocol.")
                elif risk_percent <= 70.0:
                    st.write("Acoustic measurements demonstrate moderate deviation from control samples, specifically in frequency/amplitude stability parameters (Jitter/Shimmer metrics). Suggest scheduling standard clinical follow-ups or motor examination.")
                else:
                    st.write("Voice perturbations show high instability. Pitch stability (PPE), recurrence period density (RPDE), and frequency variance are strongly correlated with Parkinsonian dysarthria. Clinical investigation is highly recommended.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature contributions table (Top Features used in prediction)
                if method == "lasso":
                    st.info(f"**Lasso Pipeline Selected Features**: Out of 26 engineered features, Lasso selected {len(meta['selected_features'])} features to evaluate this patient: \n`{meta['selected_features']}`")
                elif method == "pca":
                    st.info(f"**PCA Pipeline Dimensionality Reduction**: The 13 collinear Jitter & Shimmer features were successfully compressed into 2 uncorrelated principal components, preventing collinearity leakage.")
else:
    st.warning("Model file not found. Please verify that training pipeline `train.py` has completed successfully.")
