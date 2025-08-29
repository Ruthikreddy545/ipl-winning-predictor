import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ===============================
# Load dataset
# ===============================
data = pd.read_csv("dataset.csv")  

print("Dataset Columns:", data.columns)
print("Sample Rows:")
print(data.head())

# ===============================
# Features and target
# ===============================
# Use "results" column as target
X = data.drop("results", axis=1)
y = data["results"]

# ===============================
# One-hot encode categorical columns
# ===============================
categorical_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train model
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Evaluate model
# ===============================
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# ===============================
# Save model and columns
# ===============================
joblib.dump(model, "ipl_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
print("Model and columns saved successfully!")
