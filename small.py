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
train = df.iloc[:-12]
test = df.iloc[-12:]

# ---------------------------------------------------------
# 2. ARIMA Modell (2,1,2)
# ---------------------------------------------------------
print("\n--- ARIMA Modell futtatása ---")

# ARIMA modell illesztése (p=2, d=1, q=2)
# Megjegyzés: A sima ARIMA nem kezeli a szezonalitást, ezt fogjuk látni az eredményen.
arima_model = ARIMA(train['Utasok'], order=(2, 1, 2))
arima_fit = arima_model.fit()

# Előrejelzés a teszt időszakra (12 lépés)
arima_forecast = arima_fit.forecast(steps=12)
# Az előrejelzés indexét hozzáigazítjuk a teszt adatok dátumaihoz
arima_forecast.index = test.index

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

# Jövőbeli dataframe készítése (12 hónapra előre)
future = m.make_future_dataframe(periods=12, freq='MS')
prophet_forecast_full = m.predict(future)

# Csak a teszt időszakra vonatkozó előrejelzés kinyerése
prophet_forecast = prophet_forecast_full.iloc[-12:]['yhat']
prophet_forecast.index = test.index

# ---------------------------------------------------------
# 4. Eredmények Vizualizálása és Összehasonlítása
# ---------------------------------------------------------
plt.figure(figsize=(14, 7))

# Tényleges adatok (Train + Test)
plt.plot(train.index, train['Utasok'], label='Tanító adatok (Múlt)', color='black')
plt.plot(test.index, test['Utasok'], label='Tényleges adat (Teszt)', color='green', linewidth=2)

# ARIMA Előrejelzés
plt.plot(arima_forecast.index, arima_forecast, label='ARIMA (2,1,2) Előrejelzés', color='red', linestyle='--')

# Prophet Előrejelzés
plt.plot(prophet_forecast.index, prophet_forecast, label='Prophet Előrejelzés', color='blue', linestyle='--')

plt.title('ARIMA vs Prophet Előrejelzés Összehasonlítása (AirPassengers)', fontsize=16)
plt.xlabel('Dátum')
plt.ylabel('Utasok száma')
plt.legend()
plt.grid(True)

# Kép mentése (opcionális)
# plt.savefig('arima_vs_prophet_plot.png')

plt.show()