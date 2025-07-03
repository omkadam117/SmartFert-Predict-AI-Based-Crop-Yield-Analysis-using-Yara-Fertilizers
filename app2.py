pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 2: Load Dataset
df = pd.read_csv(r"C:\Users\MANAKARI SIR\Downloads\yara_fertilizer_project\large_crop_fertilizer_data.csv")  # replace with your path
print(df.head())

# STEP 3: Exploratory Data Analysis
print(df.describe())
print(df['Crop'].value_counts())

# Plot Yield vs Fertilizer Used
sns.boxplot(x='Crop', y='Crop_Yield_ton_per_ha', hue='Fertilizer_Type', data=df)
plt.title("Crop Yield by Fertilizer Type")
plt.show()

# Correlation heatmap
numerical_cols = df.select_dtypes(include=np.number)
sns.heatmap(numerical_cols.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()

# STEP 4: Preprocess Data
df_encoded = pd.get_dummies(df, columns=["State", "Crop", "Fertilizer_Type"], drop_first=True)

X = df_encoded.drop("Crop_Yield_ton_per_ha", axis=1)
y = df_encoded["Crop_Yield_ton_per_ha"]

# STEP 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# STEP 7: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# STEP 8: Evaluation Function
def evaluate(model_name, y_true, y_pred):
    print(f"\n🔍 {model_name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.4f}")

# STEP 9: Model Evaluation
evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)

# STEP 10 (Optional): Save models for Streamlit
import joblib
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(lr, "linear_regression_model.pkl")
