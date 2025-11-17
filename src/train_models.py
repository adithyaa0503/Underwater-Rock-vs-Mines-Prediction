import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Create models folder if not exists
os.makedirs("../models", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/sonar_data.csv")

# Features and labels
X = df.drop("Label", axis=1)
y = df["Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")

# Initialize models
models = {
    "knn_model.pkl": KNeighborsClassifier(n_neighbors=5),
    "logistic_regression_model.pkl": LogisticRegression(max_iter=1000),
    "svm_model.pkl": SVC(kernel="rbf", probability=True)
}

# Train & save models
for filename, model in models.items():
    print(f"Training {filename} ...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Save
    joblib.dump(model, f"../models/{filename}")

print("\nAll models saved in /models folder successfully!")
