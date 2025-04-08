import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv("creditcard_2023.csv")

# Step 3: Data Exploration
print("First 5 rows of the dataset:")
print(df.head())
print("\nData types of the dataset:")
print(df.dtypes)
print("\nStatistical data of the dataset:")
print(df.describe())
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Step 4: Data Preprocessing
print("\nClass Distribution Before SMOTE:")
print(df['Class'].value_counts())

# Check for missing values
print("\nMissing values in dataset:\n", df.isnull().sum().sum())

# Split data into X (features) and y (target)
X = df.drop(columns=['Class'])
y = df['Class']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Apply SMOTE for Balancing
print("\nClass Distribution Before Applying SMOTE:")
print(pd.Series(y_train).value_counts()) #use pandas series to display value counts.

try:
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("\nClass Distribution After SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

except ValueError as e:
    print("\nSMOTE Failed:", e)
    print("Trying with lower sampling ratio...")
    smote = SMOTE(sampling_strategy=0.2, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("\nClass Distribution After Adjusted SMOTE:")
    print(pd.Series(y_train_smote).value_counts())
except Exception as e:
    print(f"An unexpected error occured: {e}")

# Step 6: Train Random Forest Model with Hyperparameter Tuning
clf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train_smote, y_train_smote)
clf_best = grid_search.best_estimator_

# Step 7: Evaluate Model Performance
y_pred = clf_best.predict(X_test)
print("\nModel Performance After SMOTE:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
plt.title("Confusion Matrix After SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 8: Compare with Model Performance Before SMOTE
clf_no_smote = RandomForestClassifier(random_state=42)
clf_no_smote.fit(X_train, y_train)
y_pred_no_smote = clf_no_smote.predict(X_test)

print("\nModel Performance Before SMOTE:")
print(classification_report(y_test, y_pred_no_smote))

# Confusion Matrix Before SMOTE
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
sns.heatmap(cm_no_smote, annot=True, fmt="d", cmap="coolwarm", xticklabels=["No Fraud", "Fraud"], yticklabels=["No Fraud", "Fraud"])
plt.title("Confusion Matrix Before SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Visualizing Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette="coolwarm")
plt.title("Class Distribution Before SMOTE")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote, palette="coolwarm")
plt.title("Class Distribution After SMOTE")
plt.show()