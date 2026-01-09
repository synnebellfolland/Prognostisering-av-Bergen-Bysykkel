import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import random
import numpy as np
random.seed(42)
np.random.seed(42)

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# er disse nødvendig å ha med?:
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# mini pre-prosessering først
stations_raw = pd.read_csv("raw_data/stations.csv")
trips_raw = pd.read_csv("raw_data/trips.csv")
weather_raw = pd.read_csv("raw_data/weather.csv")

# fjerner hullet i stations
stations_raw["timestamp"] = pd.to_datetime(stations_raw["timestamp"], errors="coerce", utc=True)
stations_sorted = stations_raw.sort_values("timestamp")
stations_sorted["gap"] = stations_sorted["timestamp"].diff()
largest_gap = stations_sorted["gap"].max()
after_gap = stations_sorted.loc[stations_sorted["gap"] == largest_gap, "timestamp"].iloc[0]
stations_after_gap = stations_sorted[stations_sorted["timestamp"] > after_gap].copy()

# deler i train, val og test på BARE stations
stations_train, stations_split = train_test_split(stations_after_gap, test_size=0.3, shuffle=False)
stations_val, stations_test = train_test_split(stations_after_gap, test_size=0.5, shuffle=False)


def pipeline(stations_data):
    # se om noe av dette må endre seg
    stations = stations_data
    trips = trips_raw
    weather = weather_raw
    
    # gjør om til dato og tidspunkt
    stations["timestamp"] = pd.to_datetime(stations["timestamp"], errors="coerce", utc=True)
    trips["started_at"] = pd.to_datetime(trips["started_at"], errors="coerce", utc=True)
    trips["ended_at"] = pd.to_datetime(trips["ended_at"], errors="coerce", utc=True)
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce", utc=True)

    # lager total_trips per time
    trips["hour"] = trips["started_at"].dt.ceil("h")
    trips_per_hour = trips.groupby("hour").size().reset_index(name="total_trips")

    # gjør weather om til hver time
    # bruker ikke floor her
    weather["hour"] = weather["timestamp"].dt.ceil("h")
    weather_per_hour = (
        weather
        .groupby("hour")[["temperature", "precipitation", "wind_speed"]]
        .agg({
            "temperature" : "mean",
            "precipitation" : "sum",
            "wind_speed" : "mean"
        })
        .reset_index()
    )

    # bruker bare de utvalgte stasjonene 
    utvalgte_stasjoner = [
        "Møllendalsplass",
        "Torgallmenningen",
        "Grieghallen",
        "Høyteknologisenteret",
        "Studentboligene",
        "Akvariet",
        "Damsgårdsveien 71",
        "Dreggsallmenningen Sør",
        "Florida Bybanestopp"
    ]

    stations_filtered = stations[stations["station"].isin(utvalgte_stasjoner)].copy()

    # lager en "hour"-kolonne
    stations_filtered["hour"] = stations_filtered["timestamp"].dt.ceil("h")

    # lager en kolonne som tar vare på klokkeslettet til sist målte observasjon
    last_obs = stations_filtered.sort_values(by=["timestamp", "station"]).groupby("hour")["timestamp"].last().reset_index(name="last_obs")

    # gjør hver stasjon til egne kolonner med free bikes
    stations_per_hour = (
        stations_filtered
        .sort_values("timestamp")
        .groupby(["hour", "station"])["free_bikes"]
        .last()
        .unstack(fill_value=0)
        .add_suffix("_free_bikes")
        .reset_index()
    )

    # hver rad tilsvarer en stasjon per time
    stations = stations_per_hour.melt(id_vars="hour", var_name="station", value_name="free_bikes")

    # fjerner "_free_bikes"
    stations["station"] = stations["station"].str.replace("_free_bikes", "", regex=False)

    # bruker one-hot encoding for hver stasjon
    stations_dummies = pd.get_dummies(stations["station"])

    # slår sammen hour, station og free_bikes sammen med stations_dummies
    stations = pd.concat([stations[["hour", "station", "free_bikes"]], stations_dummies], axis=1).sort_values(["hour", "station"]).reset_index(drop=True)

    # lager full tidsramme
    full_range = pd.date_range(start=stations["hour"].min(), end=stations["hour"].max(), freq="h")
    stations_list = stations["station"].unique()
    full_grid = pd.DataFrame([(station, hour) for station in stations_list for hour in full_range], columns=["station", "hour"])
    stations_expanded = full_grid.merge(stations, on=["station", "hour"], how="left")

    # fyller inn manglende verdier
    cols = stations_expanded.columns.difference(["station", "hour"])
    stations_expanded[cols] = stations_expanded.groupby("station")[cols].ffill()
    stations_expanded = stations_expanded.sort_values(["hour", "station"])

    # legger til last_obs i datasettet
    final_stations = stations_expanded.merge(last_obs, on="hour", how="left")

    # slår sammen datasettene
    stations_and_trips = final_stations.merge(trips_per_hour[["hour", "total_trips"]], on="hour", how="left").copy()
    stations_and_trips["total_trips"] = stations_and_trips["total_trips"].astype("Int64")
    merged = stations_and_trips.merge(weather_per_hour, on="hour", how="left").copy()

    merged["target"] = merged.groupby("station", observed=True)["free_bikes"].shift(-1)

    # endrer først navnet på "hour" kolonnen tilbake til "timestamp"
    merged.rename(columns={"hour": "timestamp"}, inplace=True)

    # forskyver temperatur for å få ikke-negative verdier (til modellering senere)
    merged["temperature_shifted"] = merged["temperature"] - merged["temperature"].min()

    # manglende verdier
    merged["total_trips"] = merged["total_trips"].fillna(0)
    merged["last_obs"] = merged["last_obs"].ffill()
    merged["precipitation"] = merged["precipitation"].fillna(0)
    merged["wind_speed"] = merged["wind_speed"].ffill()
    merged["temperature_shifted"] = merged["temperature_shifted"].ffill()

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True).dt.tz_convert("Europe/Oslo")
    merged["last_obs"] = pd.to_datetime(merged["last_obs"], utc=True).dt.tz_convert("Europe/Oslo")

    # legger til ekstra kolonner som kan være nyttige
    merged["hour"] = merged["timestamp"].dt.hour
    merged["day_of_week"] = merged["timestamp"].dt.day_of_week
    merged["month"] = merged["timestamp"].dt.month

    # fikser rekkefølge på kolonner og tar vekk unødvendige
    df = merged[[
        "timestamp",
        "last_obs",
        "hour", 
        "day_of_week",
        "month", 
        "station",
        "free_bikes",
        "Akvariet", 
        "Damsgårdsveien 71", 
        "Dreggsallmenningen Sør", 
        "Florida Bybanestopp", 
        "Grieghallen", 
        "Høyteknologisenteret", 
        "Møllendalsplass", 
        "Studentboligene", 
        "Torgallmenningen", 
        "precipitation", 
        "wind_speed", 
        "temperature_shifted", 
        "total_trips", 
        "target"
    ]].copy()

    return df


def modellering():

    # må finne den beste måten å importere datasettene på
    train = pipeline(stations_train)
    val = pipeline(stations_val)
    test = pipeline(stations_test)

    # fjerner manglende verdier før fit
    train = train.dropna(subset=["target"])
    val = val.dropna(subset=["target"])
    test = test.dropna(subset=["target"])

    # drppper tekstkolonner og deler i X og y
    X_train, y_train = train.drop(columns=(["timestamp", "last_obs", "target", "station"])), train["target"]
    X_val, y_val = val.drop(columns=(["timestamp", "last_obs", "target", "station"])), val["target"]
    X_test, y_test = test.drop(columns=(["timestamp", "last_obs", "target", "station"])), test["target"]
    
    # modellene som skal brukes
    models = {
        "Baseline" : DummyRegressor(strategy="mean"),
        "LinearRegression" : LinearRegression(),
        "kNN-3" : KNeighborsRegressor(n_neighbors=3),
        "kNN-10" : KNeighborsRegressor(n_neighbors=10), 
        "kNN-50" : KNeighborsRegressor(n_neighbors=50),
        "DecisionTree" : DecisionTreeRegressor(random_state=42),
        "RandomForest" : RandomForestRegressor(random_state=42),
        "Lasso" : make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=0.01, max_iter=5000)),
        "SVR" : make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    }

    # trener modellene
    for modelname, model in models.items():
        model.fit(X_train, y_train)

    # finner RMSE for de ulike modellene
    rmse_scores = {name: {"train": root_mean_squared_error(y_train, model.predict(X_train)),
                          "val": root_mean_squared_error(y_val, model.predict(X_val))} for name, model in models.items()}
    
    rmse_df = pd.DataFrame.from_dict(rmse_scores, orient="index")
    rmse_df.sort_values(by="val", inplace=True, ascending=True)

    # henter ut den modellen med lavest RMSE 
    best_model_name = rmse_df.index[0]
    best_model = models[best_model_name]

    # tester med test-data
    forventet_rmse = root_mean_squared_error(y_test, best_model.predict(X_test))

    print(f"Beste modell: {best_model_name}")
    print(f"Forventet RMSE: {forventet_rmse:.2f}")

    # lagrer modellen slik at man kan bruke den i predict.py
    pickle.dump(best_model, open("model.pkl", "wb"))

#train = pipeline(stations_train)
#val = pipeline(stations_val)
#test = pipeline(stations_test)

if __name__ == "__main__":
    modellering()