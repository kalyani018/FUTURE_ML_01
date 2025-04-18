import pandas as pd
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Enable interactive plotting
plt.ion()

# Load sales data
df = pd.read_csv("sales_data_samples.csv")

# Format date and prepare data
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], format="%d-%m-%Y")
daily_sales = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
daily_sales.rename(columns={'ORDERDATE': 'ds', 'SALES': 'y'}, inplace=True)

# Resample to monthly sales for decomposition (you can use 'W' for weekly if preferred)
monthly_sales = daily_sales.set_index('ds').resample('M').sum()

# Decompose the time series
decomposition = seasonal_decompose(monthly_sales['y'], model='additive', period=12)

# Plot decomposition components
fig_decomp = decomposition.plot()
fig_decomp.set_size_inches(12, 8)
plt.suptitle("Time Series Decomposition", fontsize=16)
plt.tight_layout()
plt.draw()
plt.pause(2)


# Build and train model
model = Prophet()
model.fit(daily_sales)

# Future prediction
future = model.make_future_dataframe(periods=100)
forecast = model.predict(future)

# Plot 1: Forecast
fig1 = model.plot(forecast)
plt.title("Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.draw()
plt.pause(2)  # Pause to show the figure for a few seconds

# Plot 2: Actual vs Predicted
fig2 = plt.figure(figsize=(10, 6))
plt.plot(daily_sales['ds'], daily_sales['y'], label='Actual Sales', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Sales', color='orange')
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.draw()
plt.pause(2)



# Plot 3: Trend and Seasonality
fig3 = model.plot_components(forecast)
plt.draw()
plt.pause(2)

# Keep plots open until manually closed
input("Press Enter to close all plots...")
plt.ioff()
plt.close('all')
