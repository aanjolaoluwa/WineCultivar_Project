# model_development.py

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["cultivar"] = wine.target

# Select 6 features
features = [
    "alcohol",
    "malic_acid",
    "ash",
    "magnesium",
    "flavanoids",
    "proline"
]

X = df[features]
y = df["cultivar"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/wine_cultivar_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")

