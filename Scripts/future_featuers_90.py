import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load encoders and scaler
le_pname = joblib.load(r'P:\2105520\4th YEAR\Sem8\Project\Scripts')
le_name = joblib.load(r'P:\2105520\4th YEAR\Sem8\Project\Scripts')
le_itemcode = joblib.load(r'P:\2105520\4th YEAR\Sem8\Project\Scripts')
le_cmonth = joblib.load(r'P:\2105520\4th YEAR\Sem8\Project\Scripts')
scaler = joblib.load(r'P:\2105520\4th YEAR\Sem8\Project\Scripts')

# Load cleaned data
df = pd.read_csv(r'P:\2105520\4th YEAR\Sem8\Project\Output\cleaned_data.csv')
df['C_DATE'] = pd.to_datetime(df['C_DATE'], dayfirst=True)

# Step 1: Top 50 customers and products
top_pnames = df['PNAME'].value_counts().head(50).index
top_itemcodes = df['ITEMCODE'].value_counts().head(50).index
filtered_df = df[df['PNAME'].isin(top_pnames) & df['ITEMCODE'].isin(top_itemcodes)]

# Step 2: Top 1000 highest selling (PNAME, ITEMCODE)
top_pairs = (
    filtered_df.groupby(['PNAME', 'ITEMCODE'])['QTY']
    .sum()
    .sort_values(ascending=False)
    .head(1000)
    .reset_index()[['PNAME', 'ITEMCODE']]
)

# Merge product name
top_pairs = pd.merge(top_pairs, df[['PNAME', 'ITEMCODE', 'NAME']].drop_duplicates(), on=['PNAME', 'ITEMCODE'])

# Map NAME for each ITEMCODE
item_name_map = top_pairs.set_index(['PNAME', 'ITEMCODE'])['NAME'].to_dict()

# Generate dates
start_date = datetime(2024, 4, 1)
dates = [start_date + timedelta(days=i) for i in range(90)]

# Generate rows
rows = []
total = len(dates) * len(top_pairs)
count = 0

for date in dates:
    c_month_encoded = le_cmonth.transform([date.month])[0]
    for _, row in top_pairs.iterrows():
        pname, itemcode = row['PNAME'], row['ITEMCODE']
        name = item_name_map[(pname, itemcode)]
        rows.append([
            date, c_month_encoded, pname, name, itemcode, 0.0, 0.0  # FREE, FREEAMT
        ])
        count += 1
        if count % 100000 == 0:
            print(f"Progress: {count}/{total} rows generated...")

# DataFrame + encoding
future_df = pd.DataFrame(rows, columns=['C_DATE', 'C_MONTH', 'PNAME', 'NAME', 'ITEMCODE', 'FREE', 'FREEAMT'])

# Label encode
future_df['PNAME'] = le_pname.transform(future_df['PNAME'])
future_df['NAME'] = le_name.transform(future_df['NAME'])
future_df['ITEMCODE'] = le_itemcode.transform(future_df['ITEMCODE'])

# Scale FREE, FREEAMT
future_df[['FREE', 'FREEAMT']] = scaler.transform(future_df[['FREE', 'FREEAMT']])

# Save
save_path = r'P:\2105520\4th YEAR\Sem8\Project\Output\future_features_90.csv'
future_df.to_csv(save_path, index=False)
print(f"✅ Saved optimized future features to:\n{save_path}")
