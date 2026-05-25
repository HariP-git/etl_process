import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set(style="whitegrid")

df = pd.read_csv('/content/used_cars_data.csv')
print("Data Loaded Successfully")
print(df.head())

# Remove units from columns
df['Mileage'] = df['Mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
df['Engine'] = df['Engine'].str.extract(r'(\d+)').astype(float)
df['Power'] = df['Power'].str.extract(r'(\d+\.?\d*)').astype(float)
df['New_Price'] = df['New_Price'].str.extract(r'(\d+\.?\d*)').astype(float)

# Drop unnecessary column
df.drop(['S.No.'], axis=1, inplace=True)

# Missing values
df = df.dropna()
df['Car_Age'] = 2025 - df['Year']

le = LabelEncoder()
for col in ['Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']:
    df[col] = le.fit_transform(df[col])

print("\nCleaned Data:")
print(df.head())

# 1. Price Distribution
plt.figure()
sns.histplot(df['Price'], kde=True)
plt.title("Price Distribution")
plt.show()

# 2. Boxplot (Outliers)
plt.figure()
sns.boxplot(x=df['Price'])
plt.title("Price Outliers")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. Scatter (Car Age vs Price)
plt.figure()
sns.scatterplot(x=df['Car_Age'], y=df['Price'])
plt.title("Car Age vs Price")
plt.show()

# 5. Fuel Type vs Price
plt.figure()
sns.barplot(x='Fuel_Type', y='Price', data=df)
plt.title("Fuel Type vs Price")
plt.show()

# 6. Count Plot (Fuel Type)
plt.figure()
sns.countplot(x='Fuel_Type', data=df)
plt.title("Fuel Type Distribution")
plt.show()

# 7. Violin Plot (Transmission vs Price)
plt.figure()
sns.violinplot(x='Transmission', y='Price', data=df)
plt.title("Transmission vs Price Distribution")
plt.show()

# 8. KDE Plot (Price Density)
plt.figure()
sns.kdeplot(df['Price'], fill=True)
plt.title("Price Density")
plt.show()

# 9. Regression Plot (Mileage vs Price)
plt.figure()
sns.regplot(x=df['Mileage'], y=df['Price'])
plt.title("Mileage vs Price")
plt.show()

# 10. Missing Value Heatmap
plt.figure()
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values")
plt.show()

# 11. Pairplot (Important features only)
sns.pairplot(df[['Price','Mileage','Engine','Power','Car_Age']])
plt.show()

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLinear Regression")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2:", r2_score(y_test, lr_pred))

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))

# 12. Actual vs Predicted
plt.figure()
sns.scatterplot(x=y_test, y=rf_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

# 13. Residuals
residuals = y_test - rf_pred
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution")
plt.show()

# 14. Feature Importance
importances = rf.feature_importances_
features = X.columns

plt.figure()
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

print("\n✅ Pipeline Completed Successfully")
