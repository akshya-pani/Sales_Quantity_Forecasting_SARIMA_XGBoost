# XGB_P1.py (Phase 1 Final Model Using Best Parameters)

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load cleaned data
cleaned_path = r'P:\2105520\4th YEAR\Sem8\Project\Output\cleaned_data.csv'
df = pd.read_csv(cleaned_path)
df['C_DATE'] = pd.to_datetime(df['C_DATE'], dayfirst=True)
df = df.sort_values('C_DATE')

# Encode categorical features
le_pname = LabelEncoder()
le_name = LabelEncoder()
le_itemcode = LabelEncoder()
le_cmonth = LabelEncoder()
df['PNAME'] = le_pname.fit_transform(df['PNAME'])
df['NAME'] = le_name.fit_transform(df['NAME'])
df['ITEMCODE'] = le_itemcode.fit_transform(df['ITEMCODE'])
df['C_MONTH'] = le_cmonth.fit_transform(df['C_MONTH'])

# Scale FREE and FREEAMT
scaler = StandardScaler()
df[['FREE', 'FREEAMT']] = scaler.fit_transform(df[['FREE', 'FREEAMT']])

# Drop missing values
df.dropna(inplace=True)

# Split features and target
X = df.drop(['QTY', 'C_DATE', 'EXPDT', 'AMOUNT'], axis=1)
y = df['QTY']

# Final best parameters
best_params = {
    'n_estimators': 200,
    'learning_rate': 0.09,
    'max_depth': 8,
    'subsample': 1.0,
    'colsample_bytree': 0.9,
    'gamma': 0.15,
    'min_child_weight': 3,
    'reg_alpha': 0.25,
    'reg_lambda': 1.5,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'verbosity': 0
}

# Train model
model = XGBRegressor(**best_params)
model.fit(X, y)

# Save the model and encoders/scaler
joblib.dump(model, 'xgboost_phase1_model.pkl')
joblib.dump(le_pname, 'le_pname.pkl')
joblib.dump(le_name, 'le_name.pkl')
joblib.dump(le_itemcode, 'le_itemcode.pkl')
joblib.dump(le_cmonth, 'le_cmonth.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load future features
future_path = r'P:\2105520\4th YEAR\Sem8\Project\Output\future_features_90.csv'
future_df = pd.read_csv(future_path)

# Apply same preprocessing
future_df['PNAME'] = le_pname.transform(future_df['PNAME'])
future_df['NAME'] = le_name.transform(future_df['NAME'])
future_df['ITEMCODE'] = le_itemcode.transform(future_df['ITEMCODE'])
future_df['C_MONTH'] = le_cmonth.transform(future_df['C_MONTH'])
future_df[['FREE', 'FREEAMT']] = scaler.transform(future_df[['FREE', 'FREEAMT']])

# Ensure feature order matches training data
X_future = future_df[X.columns]

# Predict future QTY
future_df['XGB_Predicted_QTY'] = model.predict(X_future)

# Save predictions
output_path = r'P:\2105520\4th YEAR\Sem8\Project\Output\xgboost_output_90.csv'
future_df.to_csv(output_path, index=False)

print(f"Phase 1 XGBoost model completed and predictions saved to xgboost_output_90.csv")
