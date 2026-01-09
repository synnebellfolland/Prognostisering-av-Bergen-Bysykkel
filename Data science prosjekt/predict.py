import pickle 
import pandas as pd
from train import pipeline

model = pickle.load(open("model.pkl", "rb"))

stations_raw = pd.read_csv("raw_data/stations.csv")

# fjerner hullet i stations
stations_raw["timestamp"] = pd.to_datetime(stations_raw["timestamp"], errors="coerce", utc=True)
stations_sorted = stations_raw.sort_values("timestamp")
stations_sorted["gap"] = stations_sorted["timestamp"].diff()
largest_gap = stations_sorted["gap"].max()
after_gap = stations_sorted.loc[stations_sorted["gap"] == largest_gap, "timestamp"].iloc[0]
stations_after_gap = stations_sorted[stations_sorted["timestamp"] > after_gap].copy()

# tar inn hele modellen: trian, val og test
df = pipeline(stations_raw)


def predict(): 
    last_timestamp = df["timestamp"].max()
    last_observation = df["last_obs"].max()

    prediction_time = last_timestamp.ceil("h") + pd.DateOffset(hours=1)

    latest_per_station = df.sort_values("timestamp").groupby("station").last().reset_index()

    X_pred = latest_per_station.drop(columns=["timestamp", "last_obs", "target", "station"])

    latest_per_station["predicted"] = model.predict(X_pred)

    print(f"Siste tisstempel i data: {last_observation}")
    print(f"Neste hele klokketime: {last_timestamp}")
    print(f"Predikerer for tidsstempel: {prediction_time}")

    # brukte KI til å hjelpe med denne utskriften
    print("Stasjon\t\t\tNåværende sykler\tPredikerte sykler")
    for _, row in latest_per_station.iterrows():
        print(f"{row['station']:25s} {int(row['free_bikes']):>5} {int(round(row['predicted'])):>15}")

predict()