import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Title and warning
st.title("🏥 AI-Based Disease Prediction System")
st.warning("⚠️ This system is for educational purposes only, not a medical diagnosis. Please consult a healthcare professional for medical advice.")
@st.cache_resource
def load_model_components():
    """Load the trained model and preprocessing components"""
   @st.cache_resource
def load_model_components():
    # Load dataset
    df = pd.read_csv("final_clean_disease_dataset.csv")

    # Features = all columns except last
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model directly
    model = LogisticRegression(max_iter=2000)
    model.fit(X_scaled, y_encoded)

    symptom_columns = X.columns.tolist()

    return model, scaler, label_encoder, symptom_columns

# Load components
model, scaler, label_encoder, symptom_columns = load_model_components()

# Main interface
st.header("Select Your Symptoms")

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Multi-select for symptoms
    selected_symptoms = st.multiselect(
        "Choose symptoms you are experiencing:",
        options=symptom_columns,
        help="Select all symptoms that apply to you"
    )
    
    # Alternative: Checkbox interface (commented out)
    # st.subheader("Or select using checkboxes:")
    # selected_symptoms_checkbox = []
    # for i, symptom in enumerate(symptom_columns):
    #     if st.checkbox(symptom, key=f"checkbox_{i}"):
    #         selected_symptoms_checkbox.append(symptom)

with col2:
    st.subheader("Selected Symptoms")
    if selected_symptoms:
        for symptom in selected_symptoms:
            st.write(f"✓ {symptom}")
    else:
        st.write("No symptoms selected")

# Prediction section
st.header("Prediction")

if st.button("🔍 Predict Disease", type="primary"):
    if not selected_symptoms:
        st.error("Please select at least one symptom before predicting.")
    else:
        try:
            # Create feature vector
            feature_vector = np.zeros(len(symptom_columns))
            for symptom in selected_symptoms:
                if symptom in symptom_columns:
                    feature_vector[symptom_columns.index(symptom)] = 1
            
            # Reshape for prediction
            feature_vector = feature_vector.reshape(1, -1)
            
            # Apply scaling
            scaled_features = scaler.transform(feature_vector)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]
            
            # Decode prediction
            predicted_disease = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100
            
            # Display results
            st.success(f"**Predicted Disease:** {predicted_disease}")
            st.info(f"**Confidence:** {confidence:.1f}%")
            
            # Show probability distribution for top 3 diseases
            if len(prediction_proba) > 1:
                st.subheader("Top Predictions:")
                
                # Get top 3 predictions
                top_indices = np.argsort(prediction_proba)[-3:][::-1]
                
                for i, idx in enumerate(top_indices):
                    disease_name = label_encoder.inverse_transform([idx])[0]
                    probability = prediction_proba[idx] * 100
                    
                    if i == 0:
                        st.write(f"🥇 **{disease_name}**: {probability:.1f}%")
                    elif i == 1:
                        st.write(f"🥈 {disease_name}: {probability:.1f}%")
                    else:
                        st.write(f"🥉 {disease_name}: {probability:.1f}%")
                        
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("*Developed for educational purposes. Always consult healthcare professionals for medical advice.*")
