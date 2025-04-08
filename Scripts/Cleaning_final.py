import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the file as a raw string
file_path = r"P:\2105520\4th YEAR\Sem8\Project\date_wise_sale_analysis 23-24.csv"  # update with your actual file path
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Split into lines and manually split the header
lines = content.splitlines()
header = lines[0].split(",")  # Assuming the header is comma-separated
data = "\n".join(lines[1:])

# Read the CSV using the manually split header
df = pd.read_csv(StringIO(data), names=header)

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

print("Columns in original data:")
print(df.columns.tolist())

# Convert date columns
# Convert C_DATE: format "01-Apr-23" -> "%d-%b-%y"
df["C_DATE"] = pd.to_datetime(df["C_DATE"].str.strip(), format="%d-%b-%y", errors="coerce")

# Convert EXPDT: format "01-04-2030" -> "%d-%m-%Y"
df["EXPDT"] = pd.to_datetime(df["EXPDT"].str.strip(), format="%d-%m-%Y", errors="coerce")

# Check date conversion issues
print("C_DATE conversion issues:", df["C_DATE"].isnull().sum())
print("EXPDT conversion issues:", df["EXPDT"].isnull().sum())

# Create additional feature: extract month-year from C_DATE for seasonality
df["C_MONTH"] = df["C_DATE"].dt.strftime("%b-%Y")

# Encode categorical columns: NAME, PNAME, ITEMCODE, C_MONTH
categorical_cols = ["NAME", "PNAME", "ITEMCODE", "C_MONTH"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.strip())
    label_encoders[col] = le

# Scale numerical columns (excluding QTY, our target)
numerical_cols = ["FREE", "FREEAMT", "AMOUNT"]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Drop any remaining rows with missing values
df.dropna(inplace=True)

# Display cleaned data info and sample
print("\nCleaned Data Info:")
print(df.info())
print("\nCleaned Data Sample:")
print(df.head())

# Optionally, save the cleaned data for further processing
df.to_csv(r"P:\2105520\4th YEAR\Sem8\Project\Output\cleaned_data.csv", index=False)
