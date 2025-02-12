import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Health Classifier",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Health Classification System")
st.write("Upload your health data to check health status")

# Sidebar with status
with st.sidebar:
    st.header("Status")
    if 'client_id' in st.session_state:
        st.success(f"Client ID: {st.session_state['client_id']}")
        if 'model_trained' in st.session_state:
            st.success("Local Model: Trained")
        else:
            st.info("Local Model: Not trained")
        
        # Add global model status
        try:
            response = requests.get("http://localhost:8000/client/global-model-status")
            if response.status_code == 200:
                result = response.json()
                st.metric("Total Clients", result['num_clients'])
        except:
            st.warning("Global status unavailable")
    else:
        st.warning("Not connected")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Upload Data", "Train Model", "View Results", "Global Model"])

# Tab 1: Data Upload
with tab1:
    st.header("1. Upload Health Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        # Read data
        df = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape:", df.shape)
            st.write("Missing values:", df.isnull().sum().sum())
        with col2:
            st.write("Columns:", list(df.columns))
        
        # Process button
        if st.button("Process Data"):
            with st.spinner("Processing..."):
                try:
                    # Reset file pointer and prepare file
                    uploaded_file.seek(0)
                    files = {'file': ('data.csv', uploaded_file.getvalue(), 'text/csv')}
                    
                    response = requests.post(
                        "http://localhost:8000/client/upload-data/",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['client_id'] = result['client_id']
                        st.session_state['data'] = df
                        
                        # Show success message and details
                        st.success(f"✅ Data processed successfully! Client ID: {result['client_id']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Samples", result['train_shape'][0])
                            st.metric("Features", result['train_shape'][1])
                        with col2:
                            st.metric("Test Samples", result['test_shape'][0])
                            st.metric("Healthy Samples", result['healthy_count'])
                        
                        # Show data summary
                        st.subheader("Data Summary")
                        for col in ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2']:
                            st.metric(
                                label=col.replace('_', ' ').title(),
                                value=f"{df[col].mean():.1f}",
                                delta=f"±{df[col].std():.1f}"
                            )
                    else:
                        st.error(f"Failed to process data: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 2: Model Training
with tab2:
    st.header("2. Model Training")
    if 'client_id' not in st.session_state:
        st.warning("Please upload and process data first")
    else:
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                try:
                    response = requests.post(
                        f"http://localhost:8000/client/train/{st.session_state['client_id']}"
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['model_trained'] = True
                        st.success("✅ Model trained successfully!")
                        
                        # Show training curves
                        history = result['training_history']
                        metrics_df = pd.DataFrame({
                            'Epoch': range(len(history['accuracy'])),
                            'Training Accuracy': history['accuracy'],
                            'Validation Accuracy': history['val_accuracy'],
                            'Loss': history['loss']
                        })
                        
                        fig = px.line(metrics_df, x='Epoch', y=['Training Accuracy', 'Validation Accuracy', 'Loss'])
                        st.plotly_chart(fig)
                        
                        # Show best metrics
                        st.subheader("Best Model Metrics")
                        final_metrics = history['metrics'][-1]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{final_metrics['accuracy']*100:.2f}%")
                        with col2:
                            st.metric("Precision", f"{final_metrics['precision']*100:.2f}%")
                        with col3:
                            st.metric("Recall", f"{final_metrics['recall']*100:.2f}%")
                        
                        # Show model info
                        st.subheader("Model Information")
                        if 'model_path' in result:
                            st.write(f"Model saved at: {result['model_path']}")
                        
                    else:
                        st.error("Training failed")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

# Tab 3: Results
with tab3:
    st.header("3. Results")
    if 'model_trained' not in st.session_state:
        st.warning("Please train the model first")
    else:
        if st.button("Get Predictions"):
            with st.spinner("Getting predictions..."):
                try:
                    response = requests.get(
                        f"http://localhost:8000/client/predict/{st.session_state['client_id']}/test"
                    )
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Show accuracy
                        st.metric("Model Accuracy", f"{result['accuracy']*100:.2f}%")
                        
                        # Show predictions vs actual values
                        df_results = pd.DataFrame({
                            'Actual': result['actual_values'],
                            'Predicted': result['predictions'],
                            'Probability': result['probabilities']
                        })
                        st.write("Prediction Results:")
                        st.dataframe(df_results)
                except Exception as e:
                    st.error(f"Error getting predictions: {str(e)}")

# Tab 4: Global Model
with tab4:
    st.header("4. Global Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Update Global Model"):
            with st.spinner("Aggregating models..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/client/aggregate/"
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Global model updated successfully!")
                        
                        # Show overall metrics
                        st.metric("Number of Clients", result['num_clients'])
                        st.metric("Global Accuracy", f"{result['global_accuracy']*100:.2f}%")
                        
                        # Show per-client results
                        st.subheader("Client Results")
                        df_results = pd.DataFrame(result['client_results'])
                        df_results['accuracy'] = df_results['accuracy'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(df_results)
                    else:
                        st.error("Failed to update global model")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("View Detailed Metrics"):
            try:
                response = requests.get(
                    "http://localhost:8000/client/global-model-metrics"
                )
                if response.status_code == 200:
                    result = response.json()
                    
                    st.metric("Total Clients", result['num_clients'])
                    st.metric("Average Global Accuracy", 
                             f"{result['average_global_accuracy']*100:.2f}%")
                    
                    # Show comparison table
                    st.subheader("Client Comparison")
                    df_metrics = pd.DataFrame(result['client_metrics'])
                    df_metrics['local_accuracy'] = df_metrics['local_accuracy'].apply(
                        lambda x: f"{x*100:.2f}%"
                    )
                    df_metrics['global_accuracy'] = df_metrics['global_accuracy'].apply(
                        lambda x: f"{x*100:.2f}%"
                    )
                    st.dataframe(df_metrics)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Federated Force") 