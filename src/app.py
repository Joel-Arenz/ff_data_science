import pandas as pd
import streamlit as st
import joblib

from pipeline.O1_data_preparation import load_data, prepare_features, prepare_output


# run 'streamlit run src/app.py' to start the app

@st.cache_data
def predict_and_merge(model_path):
    # Load models via joblib
    try:
        model = joblib.load(model_path)  
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return pd.DataFrame()  
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
    df_output = df_output[(df_output['depth_team'] == 1) & (df_output['status'] == 'ACT')]
    df_output = df_output[['season', 'week', 'player_display_name', 'position', 'recent_team', 'opponent_team', 'predicted_fantasy_points', 'fantasy_points']]

    return df_output


# Main Function of Streamlit-App
def main():

    st.title("Fantasy Points Prediction")
    st.sidebar.header("Model Selection")

    model_options = {
        "XGBoost": "models/XGBoost_model.pkl",
        "Linear Regression": "models/Linear Regression_model.pkl"
    }
    selected_model = st.sidebar.selectbox("Select a model:", list(model_options.keys()))

    with st.spinner("Loading data and generating predictions..."):
        data = predict_and_merge(model_options[selected_model])

    if data.empty:
        return 

    # Filteroptions
    st.sidebar.subheader("Filter")

    season_filter = st.sidebar.multiselect("Season:", options=data['season'].unique())
    if season_filter:
        data = data[data['season'].isin(season_filter)]

    week_filter = st.sidebar.multiselect("Week:", options=sorted(data['week'].unique(), reverse=True))
    if week_filter:
        data = data[data['week'].isin(week_filter)]
    
    player_filter = st.sidebar.multiselect("Player:", options=data['player_display_name'].unique())
    if player_filter:
        data = data[data['player_display_name'].isin(player_filter)]

    position_filter = st.sidebar.multiselect("Position:", options=data['position'].unique())
    if position_filter:
        data = data[data['position'].isin(position_filter)]

    data = data.sort_values(by=["season", "week", "predicted_fantasy_points"], ascending=[False, False, False])
    
    data['season'] = data['season'].astype(str)  

    data.columns = [
        "Season", "Week", "Player Name", "Position", "Recent Team", "Opponent Team", "Actual Fantasy Points", "Predicted Fantasy Points"
    ]

    st.write("Note: Only players who have played three consecutive games in the last three weeks are included in the predictions.")
    st.dataframe(data.reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    main()
