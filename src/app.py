import pandas as pd
import streamlit as st
import joblib  # Wird verwendet, um die Modelle zu laden
from data_loader import load_data, prepare_features, prepare_output

@st.cache_data
def predict_and_merge(model_path):
    # Modell laden
    try:
        model = joblib.load(model_path)  # Modelle mit joblib laden
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return pd.DataFrame()  # Rückgabe eines leeren DataFrames im Fehlerfall
    df = load_data()
    df_output = prepare_output(df)
    df_merged = prepare_features(df)

    # Filter the data for the 2024 season
    df_test = df_merged[df_merged['season'] == 2024]
    df_output = df_output[df_output['season'] == 2024]

    # Prepare the data for prediction
    X_test = df_test.drop(columns=['fantasy_points'])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create the output dataframe
    df_output['predicted_fantasy_points'] = y_pred

    return df_output


# Hauptfunktion der Streamlit-App
def main():
    st.title("Fantasy Points Prediction")
    st.sidebar.header("Model Selection")

    # Auswahl der Modelle
    model_options = {
        "XGBoost": "models/XGBoost_model.pkl",
        "Linear Regression": "models/Linear Regression_model.pkl"
    }
    selected_model = st.sidebar.selectbox("Select a model:", list(model_options.keys()))

    # Daten vorbereiten
    with st.spinner("Daten werden geladen und Vorhersagen erstellt..."):
        data = predict_and_merge(model_options[selected_model])

    if data.empty:
        return  # Beenden, falls beim Laden der Daten ein Fehler aufgetreten ist

    # Filteroptionen
    st.sidebar.subheader("Filter")

    season_filter = st.sidebar.multiselect("Season:", options=data['season'].unique())
    if season_filter:
        data = data[data['season'].isin(season_filter)]

    week_filter = st.sidebar.multiselect("Week:", options=data['week'].unique())
    if week_filter:
        data = data[data['week'].isin(week_filter)]
    
    player_filter = st.sidebar.multiselect("Player:", options=data['player_display_name'].unique())
    if player_filter:
        data = data[data['player_display_name'].isin(player_filter)]

    position_filter = st.sidebar.multiselect("Position:", options=data['position'].unique())
    if position_filter:
        data = data[data['position'].isin(position_filter)]

    # Tabelle anzeigen (nach season, week und predicted_fantasy_points sortiert)
    data = data.sort_values(by=["season", "week", "predicted_fantasy_points"], ascending=[False, False, False])

    data['season'] = data['season'].astype(str)  # Sicherstellen, dass die Season als ganze Zahl dargestellt wird (ohne , als Seperator)
    data['player_id'] = data['player_id'].astype(str) # Sicherstellen, dass die player_id als ganze Zahl dargestellt wird (ohne , als Seperator)

    st.write("Gefilterte Daten")
    st.dataframe(data.reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    main()
