# STEP 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Load the dataset
df = pd.read_csv('data/listings.csv')

# STEP 3: Select relevant columns
selected_columns = [
    'availability_365',
    'reviews_per_month',
    'number_of_reviews',
    'minimum_nights',
    'calculated_host_listings_count',
    'room_type',
    'latitude',
    'longitude'
]
df_model = df[selected_columns].copy()

# STEP 4: Create 'churned' target column
df_model['reviews_per_month'] = df_model['reviews_per_month'].fillna(0)
df_model['churned'] = np.where((df_model['availability_365'] == 0) | 
                               (df_model['reviews_per_month'] == 0), 1, 0)

# STEP 5: Drop rows with missing values
df_model.dropna(inplace=True)

# STEP 6: One-hot encode 'room_type'
df_model = pd.get_dummies(df_model, columns=['room_type'], drop_first=True)

# STEP 7: Define features and target
X = df_model.drop('churned', axis=1)
y = df_model['churned']

# STEP 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 9: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 10: Train Logistic Regression with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# STEP 11: Predict
y_pred = model.predict(X_test_scaled)

# STEP 12: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 13: AUC Score and ROC Curve
y_probs = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_probs)
print(f"AUC Score: {auc:.2f}")

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_model['churn_predicted'] = model.predict(scaler.transform(X))
df_model.to_csv("airbnb_churn_output.csv", index=False)
print("\nExported to 'airbnb_churn_output.csv'")
    