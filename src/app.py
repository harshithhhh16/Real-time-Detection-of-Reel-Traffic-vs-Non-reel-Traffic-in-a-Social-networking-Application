import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

st.set_page_config(
    page_title="Video Traffic Classifier",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video vs Non-Video Traffic Classifier")
st.markdown("**Real-time network traffic classification for SNS applications**")
st.markdown("---")

st.sidebar.header("📁 Model Selection")

@st.cache_resource
def load_model_list():
    if not os.path.exists(MODELS_DIR):
        return []
    models = []
    for file in os.listdir(MODELS_DIR):
        if file.startswith('video_classifier_') and file.endswith('.pkl'):
            timestamp = file.replace('video_classifier_', '').replace('.pkl', '')
            models.append(timestamp)
    return sorted(models, reverse=True)

@st.cache_resource
def load_classifier(timestamp):
    try:
        model_path = os.path.join(MODELS_DIR, f'video_classifier_{timestamp}.pkl')
        scaler_path = os.path.join(MODELS_DIR, f'scaler_{timestamp}.pkl')
        features_path = os.path.join(MODELS_DIR, f'features_{timestamp}.pkl')
        metadata_path = os.path.join(MODELS_DIR, f'metadata_{timestamp}.pkl')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        metadata = joblib.load(metadata_path)
        return model, scaler, features, metadata, None
    except Exception as e:
        return None, None, None, None, str(e)

available_models = load_model_list()

if not available_models:
    st.error("❌ No trained models found! Please train and save a model first.")
    st.info("Expected files in 'src/models/' directory:")
    st.code("video_classifier_TIMESTAMP.pkl\nscaler_TIMESTAMP.pkl\nfeatures_TIMESTAMP.pkl\nmetadata_TIMESTAMP.pkl")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model Version:",
    available_models,
    format_func=lambda x: f"Model {x[:8]}..."
)

model, scaler, features, metadata, error = load_classifier(selected_model)

if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()

st.sidebar.subheader("📊 Model Info")
if metadata:
    st.sidebar.write(f"**Training Date:** {metadata['training_date'][:10]}")
    st.sidebar.write(f"**Test Accuracy:** {metadata['test_accuracy']:.2%}")
    st.sidebar.write(f"**CV Score:** {metadata.get('cv_mean_score', 0):.3f}")
    st.sidebar.write(f"**Features:** {len(features)}")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔧 Input Traffic Features")
    with st.form("prediction_form"):
        input_col1, input_col2 = st.columns(2)
        with input_col1:
            flow_duration = st.number_input(
                "Flow Duration (ms)", 
                min_value=0.0, 
                value=3000.0,
                help="Total duration of the network flow"
            )
            total_fwd_packets = st.number_input(
                "Total Forward Packets", 
                min_value=1, 
                value=45,
                help="Number of packets sent from client to server"
            )
            total_bwd_packets = st.number_input(
                "Total Backward Packets", 
                min_value=1, 
                value=30,
                help="Number of packets sent from server to client"
            )
            total_length_fwd_packets = st.number_input(
                "Forward Packets Length", 
                min_value=0.0, 
                value=6000.0,
                help="Total bytes in forward direction"
            )
            total_length_bwd_packets = st.number_input(
                "Backward Packets Length", 
                min_value=0.0, 
                value=3500.0,
                help="Total bytes in backward direction"
            )
            fwd_packet_length_mean = st.number_input(
                "Forward Packet Mean Size", 
                min_value=0.0, 
                value=900.0,
                help="Average size of forward packets"
            )
        with input_col2:
            bwd_packet_length_mean = st.number_input(
                "Backward Packet Mean Size", 
                min_value=0.0, 
                value=650.0,
                help="Average size of backward packets"
            )
            flow_bytes_per_sec = st.number_input(
                "Flow Bytes per Second", 
                min_value=0.0, 
                value=85000.0,
                help="Data transfer rate (bytes/sec)"
            )
            flow_packets_per_sec = st.number_input(
                "Flow Packets per Second", 
                min_value=0.0, 
                value=25.0,
                help="Packet rate (packets/sec)"
            )
            flow_iat_mean = st.number_input(
                "Flow Inter-arrival Time", 
                min_value=0.0, 
                value=120.0,
                help="Average time between packets (ms)"
            )
            fwd_iat_mean = st.number_input(
                "Forward Inter-arrival Time", 
                min_value=0.0, 
                value=100.0,
                help="Average time between forward packets"
            )
            bwd_iat_mean = st.number_input(
                "Backward Inter-arrival Time", 
                min_value=0.0, 
                value=150.0,
                help="Average time between backward packets"
            )
        predict_button = st.form_submit_button("🔍 Classify Traffic", use_container_width=True)

with col2:
    st.header("📊 Prediction Result")
    result_placeholder = st.empty()
    confidence_placeholder = st.empty()
    st.subheader("🚀 Quick Tests")
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        if st.button("📺 Test Video Pattern", use_container_width=True):
            video_data = {
                'flow_duration': 8000,
                'total_fwd_packets': 120,
                'total_bwd_packets': 80,
                'total_length_fwd_packets': 15000,
                'total_length_bwd_packets': 8000,
                'fwd_packet_length_mean': 1200,
                'bwd_packet_length_mean': 800,
                'flow_bytes_per_sec': 150000,
                'flow_packets_per_sec': 50,
                'flow_iat_mean': 80,
                'fwd_iat_mean': 60,
                'bwd_iat_mean': 100
            }
            input_df = pd.DataFrame([video_data])
            input_scaled = scaler.transform(input_df[features])
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            prediction_label = 'STREAMING' if prediction == 1 else 'NON-STREAMING'
            confidence = probabilities.max()
            with result_placeholder.container():
                if prediction_label == 'STREAMING':
                    st.success(f"🎥 **{prediction_label}**")
                else:
                    st.info(f"📄 **{prediction_label}**")
            with confidence_placeholder.container():
                st.metric("Confidence", f"{confidence:.1%}")
                st.write("**Probabilities:**")
                st.write(f"Non-Video: {probabilities[0]:.1%}")
                st.progress(probabilities[0])
                st.write(f"Video: {probabilities[1]:.1%}")
                st.progress(probabilities[1])
    with col_test2:
        if st.button("📄 Test Web Pattern", use_container_width=True):
            web_data = {
                'flow_duration': 1200,
                'total_fwd_packets': 25,
                'total_bwd_packets': 15,
                'total_length_fwd_packets': 3000,
                'total_length_bwd_packets': 1800,
                'fwd_packet_length_mean': 600,
                'bwd_packet_length_mean': 500,
                'flow_bytes_per_sec': 25000,
                'flow_packets_per_sec': 15,
                'flow_iat_mean': 200,
                'fwd_iat_mean': 180,
                'bwd_iat_mean': 250
            }
            input_df = pd.DataFrame([web_data])
            input_scaled = scaler.transform(input_df[features])
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            prediction_label = 'STREAMING' if prediction == 1 else 'NON-STREAMING'
            confidence = probabilities.max()
            with result_placeholder.container():
                if prediction_label == 'STREAMING':
                    st.success(f"🎥 **{prediction_label}**")
                else:
                    st.info(f"📄 **{prediction_label}**")
            with confidence_placeholder.container():
                st.metric("Confidence", f"{confidence:.1%}")
                st.write("**Probabilities:**")
                st.write(f"Non-Video: {probabilities[0]:.1%}")
                st.progress(probabilities[0])
                st.write(f"Video: {probabilities[1]:.1%}")
                st.progress(probabilities[1])

if predict_button:
    input_data = {
        'flow_duration': flow_duration,
        'total_fwd_packets': total_fwd_packets,
        'total_bwd_packets': total_bwd_packets,
        'total_length_fwd_packets': total_length_fwd_packets,
        'total_length_bwd_packets': total_length_bwd_packets,
        'fwd_packet_length_mean': fwd_packet_length_mean,
        'bwd_packet_length_mean': bwd_packet_length_mean,
        'flow_bytes_per_sec': flow_bytes_per_sec,
        'flow_packets_per_sec': flow_packets_per_sec,
        'flow_iat_mean': flow_iat_mean,
        'fwd_iat_mean': fwd_iat_mean,
        'bwd_iat_mean': bwd_iat_mean
    }
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df[features])
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        prediction_label = 'STREAMING' if prediction == 1 else 'NON-STREAMING'
        confidence = probabilities.max()
        with result_placeholder.container():
            if prediction_label == 'STREAMING':
                st.success(f"🎥 **{prediction_label}**")
                st.write("This traffic pattern indicates **video streaming** (like YouTube, TikTok, Instagram Reels)")
            else:
                st.info(f"📄 **{prediction_label}**")
                st.write("This traffic pattern indicates **non-video content** (like text posts, comments, web browsing)")
        with confidence_placeholder.container():
            st.metric("Confidence", f"{confidence:.1%}")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': "Confidence %"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Detailed Probabilities:**")
            prob_df = pd.DataFrame({
                'Traffic Type': ['Non-Video', 'Video'],
                'Probability': [probabilities[0], probabilities[1]]
            })
            fig_bar = px.bar(
                prob_df, 
                x='Traffic Type', 
                y='Probability',
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚀 Video Traffic Classifier v1.0 | Built for SNS Applications | Real-time Network Analysis</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 How to Run")
st.sidebar.markdown("""
1. **Save this as `app.py`**
2. **Install requirements:**
3. **Run the app:**
4. **Open browser:** http://localhost:8501
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use Quick Tests to see typical video")