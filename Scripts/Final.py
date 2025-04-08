import pandas as pd

# Load CSVs
sarima_df = pd.read_csv(r'P:\2105520\4th YEAR\Sem8\Project\Output\sarima_output_90.csv')
xgb_df = pd.read_csv(r'P:\2105520\4th YEAR\Sem8\Project\Output\xgboost_output_90.csv')

# Rename 'Date' column to 'C_DATE'
sarima_df = sarima_df.rename(columns={'Date': 'C_DATE'})

# Force datetime parsing, handle errors, format both to DD-MM-YYYY
sarima_df['C_DATE'] = pd.to_datetime(sarima_df['C_DATE'], errors='coerce', dayfirst=True).dt.strftime('%d-%m-%Y')
xgb_df['C_DATE'] = pd.to_datetime(xgb_df['C_DATE'], errors='coerce').dt.strftime('%d-%m-%Y')

# Drop rows with NaT dates just in case
sarima_df = sarima_df.dropna(subset=['C_DATE'])
xgb_df = xgb_df.dropna(subset=['C_DATE'])

# Merge on C_DATE
merged_df = pd.merge(xgb_df, sarima_df, on='C_DATE', how='inner')
print("Merged DataFrame shape:", merged_df.shape)

# Final prediction: average of SARIMA and XGB
if not merged_df.empty:
    merged_df['Final_Predicted_QTY'] = (
        merged_df['SARIMA_Predicted_QTY'] + merged_df['XGB_Predicted_QTY']
    ) / 2

    # Save result
    merged_df.to_csv(r'P:\2105520\4th YEAR\Sem8\Project\Output\final_90.csv', index=False)
    print("✅ final_90.csv saved with averaged predictions.")
else:
    print("❌ No matching rows found. Merge resulted in an empty DataFrame.")
