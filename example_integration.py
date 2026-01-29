"""
Example: How to modify your existing training script to save model components
for the web application.

Add this code at the end of your existing training script.
"""

# Example of your existing training code structure:
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv('final_clean_disease_dataset.csv')
X = df.drop('Disease', axis=1)
y = df['Disease']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model (your existing evaluation code)
# ...
"""

# ADD THIS CODE AT THE END OF YOUR TRAINING SCRIPT:
import pickle

def save_components_for_webapp(model, scaler, label_encoder):
    """Save trained components for the web application"""
    
    # Save the trained model
    with open('logistic_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved successfully!")
    
    # Save the fitted scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved successfully!")
    
    # Save the fitted label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("✅ Label encoder saved successfully!")
    
    print("\n🎉 All components saved! You can now run the web application.")
    print("Run: streamlit run app.py")

# Call this function with your trained components
# save_components_for_webapp(model, scaler, label_encoder)

print("Add the save_components_for_webapp() function call to your training script!")