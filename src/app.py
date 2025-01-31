import pandas as pd
import streamlit as st
import joblib

from pipeline.O1_data_loading import load_ind_data, load_sys_data
from pipeline.O2_data_preparation import create_ind_train_and_test_data_for_lstm, create_sys_train_and_test_data_for_lstm, prepare_ind_data_for_lstm_prediction_and_outcome, prepare_ind_features_for_lr_and_xgb, prepare_ind_output_for_lr_and_xgb, prepare_sys_data_for_lstm_prediction_and_outcome, prepare_sys_features_for_lr_and_xgb, prepare_sys_output_for_lr_and_xgb, split_data_for_lr_and_xgb, prepare_ind_data_for_lstm_training, split_data_for_lstm
from pipeline.O6_predict_functions import predict_2024_season_for_ind_lstm, predict_2024_season_for_sys_lstm
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

# run 'streamlit run src/app.py' to start the app

@st.cache_data
def predictions(model_path, approach):

    # Prepare Data
    if approach == "Individual":
        df = load_ind_data()
        if model_path.endswith('.h5'):
            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            label_encoder = joblib.load("../models/individual_label_encoder.joblib")
            scaler = joblib.load("../models/individual_scaler.joblib")
            _, df_seq = create_ind_train_and_test_data_for_lstm(df)
            _, df_pred, _, _ = prepare_ind_data_for_lstm_prediction_and_outcome(df_seq)
            df_output = predict_2024_season_for_ind_lstm(model, df_pred, scaler, label_encoder)
        else:
            model = joblib.load(model_path)
            df_pred = prepare_ind_features_for_lr_and_xgb(df)
            df_output = prepare_ind_output_for_lr_and_xgb(df)
            df_pred = df_pred[df_pred['time_index'] > 202318]
            df_output = df_output[df_output['season'] == 2024]
            X_test = df_pred.drop(columns=['fantasy_points'])
            y_pred = model.predict(X_test)
            df_output['predicted_fantasy_points'] = y_pred

    elif approach == 'Systematic':
        df = load_sys_data()
        if model_path.endswith('.h5'):
            model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
            label_encoder = joblib.load("../models/systematic_label_encoder.joblib")
            scaler = joblib.load("../models/systematic_scaler.joblib")
            _, df_seq = create_sys_train_and_test_data_for_lstm(df)
            _, df_pred, _, _ = prepare_sys_data_for_lstm_prediction_and_outcome(df_seq)
            df_output = predict_2024_season_for_sys_lstm(model, df_pred, scaler, label_encoder)            
        else:
            model = joblib.load(model_path)
            df_pred = prepare_sys_features_for_lr_and_xgb(df)
            df_output = prepare_sys_output_for_lr_and_xgb(df)
            df_pred = df_pred[df_pred['time_index'] > 202318]
            df_output = df_output[df_output['season'] == 2024]
            X_test = df_pred.drop(columns=['fantasy_points'])
            y_pred = model.predict(X_test)
            df_output['predicted_fantasy_points'] = y_pred

    # Filter and sort output
    if approach == "Individual":
        df_output = df_output[(df_output['depth_team'] == 1) & (df_output['status'] == 'ACT')]
        df_output = df_output[['season', 'week', 'player_display_name', 'position', 'recent_team', 'opponent_team', 'predicted_fantasy_points', 'fantasy_points']]
    elif approach == "Systematic":
        df_output = df_output[['season', 'week', 'role', 'recent_team', 'opponent_team', 'predicted_fantasy_points', 'fantasy_points']]
        
    return df_output


# Main Function of Streamlit-App
def main():
    st.title("Fantasy Points Prediction")
    st.sidebar.header("Model and Approach Selection")

    approach_options = ["Individual", "Systematic"]
    selected_approach = st.sidebar.selectbox("Select an approach:", approach_options)

    model_options = {
        "XGBoost": f"../models/{selected_approach.lower()}_xgb_approach_model.pkl",
        "Linear Regression": f"../models/{selected_approach.lower()}_lr_approach_model.pkl",
        "LSTM": f"../models/{selected_approach.lower()}_lstm_approach_model.h5"
    }
    selected_model = st.sidebar.selectbox("Select a model:", list(model_options.keys()))

    with st.spinner("Loading data and generating predictions..."):
        data = predictions(model_options[selected_model], selected_approach)

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
    elif selected_approach == "Systematic":
        role_filter = st.sidebar.multiselect("Role:", options=data['role'].unique())
        if role_filter:
            data = data[data['role'].isin(role_filter)]

        data.columns = [
            "Season", "Week", "Role", "Recent Team", "Opponent Team", "Predicted Fantasy Points", "Actual Fantasy Points"
        ]
    else:
        data.columns = [
            "Player ID", "Week", "Predicted Fantasy Points"
        ]

    # Sort the data by Season, Week, and Predicted Fantasy Points
    data = data.sort_values(by=["Season", "Week", "Predicted Fantasy Points"], ascending=[False, False, False])

    # Remove the comma separator from the Season column
    data['Season'] = data['Season'].astype(str).str.replace(",", "")

    st.write("Note: Only players who have played three consecutive games in the last three weeks are included in the predictions.")
    st.dataframe(data.to_dict(orient="records"))

if __name__ == "__main__":
    main()
