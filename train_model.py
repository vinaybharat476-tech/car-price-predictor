import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚗 CAR PRICE PREDICTION - MODEL TRAINING")
print("=" * 60)

# ============== LOAD DATA ==============
print("\n📂 Loading dataset...")

# Load your CSV
df = pd.read_csv('data.csv')

# Display info
print(f"\n✓ Dataset loaded: {df.shape[0]} cars, {df.shape[1]} features")
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# ============== DATA CLEANING ==============
print("\n🧹 Cleaning data...")

# The target column is MSRP (not price)
target_col = 'MSRP'

# Drop rows with missing target
df = df.dropna(subset=[target_col])
y = df[target_col].copy()

# Remove $0 prices (invalid data)
df = df[y > 0]
y = y[y > 0]

# Drop price from features
X = df.drop(columns=[target_col])

print(f"Dataset size: {len(X)} cars")
print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")

# Handle missing values
print(f"Missing values before: {X.isnull().sum().sum()}")
X = X.fillna(X.mean(numeric_only=True))  # Numerical columns
X = X.fillna(X.mode().iloc[0])  # Categorical columns
print(f"Missing values after: {X.isnull().sum().sum()}")

# ============== FEATURE ENGINEERING ==============
print("\n⚙️ Feature engineering...")

X_processed = X.copy()

# Encode categorical variables
label_encoders = {}
categorical_cols = X_processed.select_dtypes(include=['object']).columns

print(f"Categorical columns found: {list(categorical_cols)}")

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded: {col} ({len(le.classes_)} categories)")

# Remove extreme outliers in price
Q1 = y.quantile(0.05)
Q3 = y.quantile(0.95)
mask = (y >= Q1) & (y <= Q3)
X_processed = X_processed[mask]
y = y[mask]

print(f"\n✓ After outlier removal: {len(y)} cars")
print(f"  Price range: ${y.min():,.0f} - ${y.max():,.0f}")

# ============== TRAIN TEST SPLIT ==============
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"  Training set: {len(X_train)} cars")
print(f"  Testing set: {len(X_test)} cars")

# ============== FEATURE SCALING ==============
print("\n📊 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============== TRAIN XGBOOST MODEL ==============
print("\n🚀 Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

# ============== EVALUATE ==============
print("\n📈 Model Performance:")
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  Training R² Score: {train_score:.4f}")
print(f"  Testing R² Score: {test_score:.4f}")
print(f"  Mean Absolute Error: ${mae:,.0f}")
print(f"  Root Mean Squared Error: ${rmse:,.0f}")

# ============== FEATURE IMPORTANCE ==============
print("\n🎯 Top 10 Important Features:")
feature_importance = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:25s} : {row['Importance']:.4f}")

# ============== SAVE MODEL ==============
print("\n💾 Saving model...")
pickle.dump(model, open('car_price_model.pkl', 'wb'))
pickle.dump(scaler, open('car_scaler.pkl', 'wb'))
pickle.dump(X_processed.columns, open('car_features.pkl', 'wb'))
pickle.dump(label_encoders, open('car_encoders.pkl', 'wb'))

print("""
✅ Model saved successfully!
✓ car_price_model.pkl
✓ car_scaler.pkl
✓ car_features.pkl
✓ car_encoders.pkl

🎉 Ready for deployment!
""")