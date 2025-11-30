import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import statsmodels.api as sm

# --- 1. Adatok betöltése ---
# Az AirPassengers adatbázis betöltése a statsmodels beépített könyvtárából
# Ez biztosítja, hogy külső fájl nélkül is lefusson a kód.
dataset = sm.datasets.get_rdataset('AirPassengers')
df = dataset.data

# Dátum index létrehozása (1949-től kezdődően, havi adatok)
df['Date'] = pd.date_range(start='1949-01-01', periods=len(df), freq='MS')
df.set_index('Date', inplace=True)
df.rename(columns={'value': 'Utasok'}, inplace=True)

print("Az adatok első 5 sora:")
print(df.head())

# Tanító és teszt adatok szétválasztása (utolsó 12 hónap a teszt)
test_horizon = 12
train = df.iloc[:-test_horizon]
test = df.iloc[-test_horizon:]

# Hány hónapra előre szeretnénk látni a predikciót (pl. 60 = 5 év)
FUTURE_MONTHS = 60

# ---------------------------------------------------------
# 2. ARIMA Modell (2,1,2)
# ---------------------------------------------------------
print("\n--- ARIMA Modell futtatása ---")

# ARIMA modell illesztése (p=2, d=1, q=2)
# Megjegyzés: A sima ARIMA nem kezeli a szezonalitást, ezt fogjuk látni az eredményen.
arima_model = ARIMA(train['Utasok'], order=(2, 1, 2))
arima_fit = arima_model.fit()

# Előrejelzés a teszt időszakra (12 lépés)
arima_forecast = arima_fit.forecast(steps=test_horizon)
# Az előrejelzés indexét hozzáigazítjuk a teszt adatok dátumaihoz
arima_forecast.index = test.index

# --- Kiegészítő: hosszabb távú előrejelzés (modellezés az egész adathalmazon)
arima_full = ARIMA(df['Utasok'], order=(2, 1, 2)).fit()
last_date = df.index[-1]
arima_full_index = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=FUTURE_MONTHS, freq='MS')
arima_full_forecast = arima_full.forecast(steps=FUTURE_MONTHS)
arima_full_forecast.index = arima_full_index

# ---------------------------------------------------------
# 3. Prophet Modell
# ---------------------------------------------------------
print("\n--- Prophet Modell futtatása ---")

# A Prophet speciális formátumot vár: 'ds' (dátum) és 'y' (érték) oszlopok
prophet_df = train.reset_index()[['Date', 'Utasok']]
prophet_df.columns = ['ds', 'y']

# Modell inicializálása és illesztése
# A Prophet automatikusan detektálja a szezonalitást
m = Prophet()
m.fit(prophet_df)

# Jövőbeli dataframe készítése (a tesztre vonatkozóan)
future = m.make_future_dataframe(periods=test_horizon, freq='MS')
prophet_forecast_full = m.predict(future)

# Csak a teszt időszakra vonatkozó előrejelzés kinyerése
prophet_forecast = prophet_forecast_full.iloc[-test_horizon:]['yhat']
prophet_forecast.index = test.index

# --- Prophet hosszabb távra, a teljes adathalmazon illesztve ---
prophet_full_df = df.reset_index()[['Date', 'Utasok']]
prophet_full_df.columns = ['ds', 'y']
m_full = Prophet()
m_full.fit(prophet_full_df)
future_full = m_full.make_future_dataframe(periods=FUTURE_MONTHS, freq='MS')
prophet_full_forecast = m_full.predict(future_full)
prophet_full_yhat = prophet_full_forecast.set_index('ds')['yhat']


# ---------------------------------------------------------
# 4. Eredmények Vizualizálása és Összehasonlítása
# ---------------------------------------------------------
plt.figure(figsize=(14, 7))

# Tényleges adatok (Train + Test)
plt.plot(train.index, train['Utasok'], label='Tanító adatok (Múlt)', color='black')
plt.plot(test.index, test['Utasok'], label='Tényleges adat (Teszt)', color='green', linewidth=2)

# ARIMA Előrejelzés (teszt)
plt.plot(arima_forecast.index, arima_forecast, label='ARIMA (teszt, 12)', color='red', linestyle='--')

# Prophet Előrejelzés (teszt)
plt.plot(prophet_forecast.index, prophet_forecast, label='Prophet (teszt, 12)', color='blue', linestyle='--')

# Hosszabb távú előrejelzések (az egész adathalmazon illesztve)
plt.plot(arima_full_forecast.index, arima_full_forecast, label=f'ARIMA (extended {FUTURE_MONTHS}m)', color='orangered', linestyle=':')
plt.plot(prophet_full_yhat.index, prophet_full_yhat, label=f'Prophet (extended {FUTURE_MONTHS}m)', color='royalblue', linestyle=':')

# Jelölés az adathalmaz végétől (utolsó ismert pont)
plt.axvline(df.index[-1], color='grey', linestyle='--', linewidth=1)

plt.title('ARIMA vs Prophet Előrejelzés Összehasonlítása (AirPassengers)', fontsize=16)
plt.xlabel('Dátum')
plt.ylabel('Utasok száma')
plt.legend()
plt.grid(True)

# X tengely korlát beállítása, hogy a kiterjesztett idősort is lássuk
plt.xlim(left=df.index[0], right=prophet_full_yhat.index[-1])

# Kép mentése (opcionális)
# plt.savefig('arima_vs_prophet_plot.png')

plt.show()