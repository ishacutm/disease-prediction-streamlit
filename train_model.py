import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("final_clean_disease_dataset.csv")

# Separate features and target
X = df.drop("Disease", axis=1)
y = df["Disease"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y_encoded)

# Save model and preprocessors
pickle.dump(model, open("logistic_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Model trained and saved successfully")
