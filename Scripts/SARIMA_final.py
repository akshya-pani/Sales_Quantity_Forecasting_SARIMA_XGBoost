import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Start timing
start_time = time.time()

# -------------------------
# Data Loading and Aggregation
# -------------------------
file_path = r"P:\\2105520\\4th YEAR\\Sem8\\Project\\Output\\cleaned_data.csv"
df = pd.read_csv(file_path, parse_dates=["C_DATE"])
df.set_index("C_DATE", inplace=True)

daily_qty = df["QTY"].resample("D").sum().asfreq("D").fillna(0)

# -------------------------
# ADF Test (Stationarity Check)
# -------------------------
def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] < 0.05:
        print("The series is stationary (p-value < 0.05).")
    else:
        print("The series is not stationary (p-value >= 0.05).")
        
adf_test(daily_qty)

# -------------------------
# ACF and PACF Plots
# -------------------------
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(daily_qty, lags=40, ax=plt.gca())
plt.subplot(122)
plot_pacf(daily_qty, lags=40, ax=plt.gca())
plt.show()

# -------------------------
# Model Training (Using Best Parameters Found Earlier)
# -------------------------
best_order = (0, 1, 3)
best_seasonal_order = (0, 2, 2, 30)

model = SARIMAX(daily_qty, order=best_order,
                 seasonal_order=best_seasonal_order,
                 enforce_stationarity=False, enforce_invertibility=False)
best_model = model.fit(disp=False, maxiter=500)
print(f"\nBest SARIMA model: {best_order} x {best_seasonal_order} - AIC: {best_model.aic:.2f}")

# -------------------------
# Forecasting with the Best Model
# -------------------------
forecast_steps = 90
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Save SARIMA forecast output
forecast_df = pd.DataFrame({
    "Date": forecast.predicted_mean.index,
    "SARIMA_Predicted_QTY": forecast.predicted_mean.values
})
forecast_df.to_csv(r"P:\\2105520\\4th YEAR\\Sem8\\Project\\Output\\sarima_output_90.csv", index=False)
print("SARIMA forecast saved to sarima_output.csv")

# Plot results
plt.figure(figsize=(14, 7))
ax = daily_qty.plot(label="Observed")
forecast.predicted_mean.plot(ax=ax, label="Forecast", color="red")
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color="pink", alpha=0.3)
ax.set_xlabel("Date")
ax.set_ylabel("QTY")
plt.legend()
plt.title(f"SARIMA Forecast of Daily QTY (Best Model: {best_order} x {best_seasonal_order})")
plt.show()

# Stop timing and print elapsed time
elapsed_time = time.time() - start_time
print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
