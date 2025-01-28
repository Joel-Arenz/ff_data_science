import pandas as pd
import streamlit as st
import joblib

from pipeline.O1_data_preparation import load_ind_data, prepare_ind_features, prepare_ind_output, load_sys_data, prepare_sys_features, prepare_sys_output


# run 'streamlit run src/app.py' to start the app

@st.cache_data
def predict_and_merge(model_path, approach):
    # Load models via joblib
    try:
        model = joblib.load(model_path)  
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return pd.DataFrame()  
    
    if approach == "Individual":
        df = load_ind_data()
        df_output = prepare_ind_output(df)
        df_merged = prepare_ind_features(df)
    else:
        df = load_sys_data()
        df_output = prepare_sys_output(df)
        df_merged = prepare_sys_features(df)

    # Filter the data for the 2024 season
    df_test = df_merged[df_merged['time_index'] > 202318]
    df_output = df_output[df_output['season'] == 2024]

    # Prepare the data for prediction
    X_test = df_test.drop(columns=['fantasy_points'])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create the output dataframe
    df_output['predicted_fantasy_points'] = y_pred
    
    if approach == "Individual":
        df_output = df_output[(df_output['depth_team'] == 1) & (df_output['status'] == 'ACT')]
        df_output = df_output[['season', 'week', 'player_display_name', 'position', 'recent_team', 'opponent_team', 'predicted_fantasy_points', 'fantasy_points']]
    else: 
        df_output = df_output[['season', 'week', 'role', 'recent_team', 'opponent_team', 'predicted_fantasy_points', 'fantasy_points']]
    return df_output


# Main Function of Streamlit-App
def main():
    st.title("Fantasy Points Prediction")
    st.sidebar.header("Model and Approach Selection")

    approach_options = ["Individual", "Systematic"]
    selected_approach = st.sidebar.selectbox("Select an approach:", approach_options)

    model_options = {
        "XGBoost": f"models/{selected_approach.lower()}_xgb_approach_model.pkl",
        "Linear Regression": f"models/{selected_approach.lower()}_lr_approach_model.pkl"
    }
    selected_model = st.sidebar.selectbox("Select a model:", list(model_options.keys()))

    with st.spinner("Loading data and generating predictions..."):
        data = predict_and_merge(model_options[selected_model], selected_approach)

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
    
    if selected_approach == "Individual":
        player_filter = st.sidebar.multiselect("Player:", options=data['player_display_name'].unique())
        if player_filter:
            data = data[data['player_display_name'].isin(player_filter)]

        position_filter = st.sidebar.multiselect("Position:", options=data['position'].unique())
        if position_filter:
            data = data[data['position'].isin(position_filter)]

        data.columns = [
            "Season", "Week", "Player Name", "Position", "Recent Team", "Opponent Team", "Predicted Fantasy Points", "Actual Fantasy Points"
        ]
    else:
        role_filter = st.sidebar.multiselect("Role:", options=data['role'].unique())
        if role_filter:
            data = data[data['role'].isin(role_filter)]

        data.columns = [
            "Season", "Week", "Role", "Recent Team", "Opponent Team", "Predicted Fantasy Points", "Actual Fantasy Points"
        ]

    # Sort the data by Season, Week, and Predicted Fantasy Points
    data = data.sort_values(by=["Season", "Week", "Predicted Fantasy Points"], ascending=[False, False, False])

    # Remove the comma separator from the Season column
    data['Season'] = data['Season'].astype(str).str.replace(",", "")

    st.write("Note: Only players who have played three consecutive games in the last three weeks are included in the predictions.")
    st.dataframe(data.to_dict(orient="records"))

if __name__ == "__main__":
    main()
