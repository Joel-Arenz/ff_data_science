from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pipeline.O1_data_loading import load_ind_data, load_sys_data
from pipeline.O2_data_preparation import create_ind_train_and_test_data_for_lstm, create_sequences_for_lstm, create_sys_train_and_test_data_for_lstm, prepare_ind_data_for_lstm_prediction_and_outcome, prepare_ind_features_for_lr_and_xgb, prepare_sys_data_for_lstm_prediction_and_outcome, prepare_sys_data_for_lstm_training, prepare_sys_features_for_lr_and_xgb, split_data_for_lr_and_xgb, prepare_ind_data_for_lstm_training, split_data_for_lstm
from pipeline.O3_pipeline_implementation import create_model_for_lstm, create_preprocessor_for_lr_and_xgb, optimize_hyperparameters_for_lr_and_xgb, train_model_for_lstm
from pipeline.O5_model_deployment import save_model_artifacts_for_lstm, save_model_for_lr_and_xgb
from pipeline.O6_predict_functions import predict_2024_season_for_ind_lstm, predict_2024_season_for_sys_lstm

def main():
    # Prepare individual features for model training and testing
    df_ind = load_ind_data()
    df_ind_prepared = prepare_ind_features_for_lr_and_xgb(df_ind)

    df_ind_lstm_train, df_ind_lstm_seq = create_ind_train_and_test_data_for_lstm(df_ind)
    scaled_ind_lstm_features, target_ind_lstm, scaler_ind_lstm_train, le_ind_lstm_train = prepare_ind_data_for_lstm_training(df_ind_lstm_train)
    feature_names_ind_lstm, df_ind_lstm_test, scaler_ind_lstm_pred, le_ind_lstm_pred = prepare_ind_data_for_lstm_prediction_and_outcome(df_ind_lstm_seq)
    X_ind_lstm, y_ind_lstm = create_sequences_for_lstm(scaled_ind_lstm_features, target_ind_lstm)

    # Prepare systematic features for model training and testing
    df_sys = load_sys_data()
    df_sys_prepared = prepare_sys_features_for_lr_and_xgb(df_sys)

    df_sys_lstm_train, df_sys_lstm_seq = create_sys_train_and_test_data_for_lstm(df_sys)
    scaled_sys_lstm_features, target_sys_lstm, scaler_sys_lstm_train, le_sys_lstm_train = prepare_sys_data_for_lstm_training(df_sys_lstm_train)
    feature_names_sys_lstm, df_sys_lstm_test, scaler_sys_lstm_pred, le_sys_lstm_pred = prepare_sys_data_for_lstm_prediction_and_outcome(df_sys_lstm_seq)
    X_sys_lstm, y_sys_lstm = create_sequences_for_lstm(scaled_sys_lstm_features, target_sys_lstm)

    # Split data for training and testing
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = split_data_for_lr_and_xgb(df_ind_prepared)
    X_train_sys, X_test_sys, y_train_sys, y_test_sys = split_data_for_lr_and_xgb(df_sys_prepared)

    X_train_ind_lstm, X_test_ind_lstm, y_train_ind_lstm, y_test_ind_lstm = split_data_for_lstm(X_ind_lstm, y_ind_lstm)
    X_train_sys_lstm, X_test_sys_lstm, y_train_sys_lstm, y_test_sys_lstm = split_data_for_lstm(X_sys_lstm, y_sys_lstm)

    preprocessor_ind = create_preprocessor_for_lr_and_xgb(df_ind_prepared)
    preprocessor_sys = create_preprocessor_for_lr_and_xgb(df_sys_prepared)

    # Define the approach
    approaches = {
        'individual_lr_approach': (LinearRegression(), X_train_ind, X_test_ind, y_train_ind, y_test_ind, preprocessor_ind, {
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, -1]
        }),
        'individual_xgb_approach': (XGBRegressor(), X_train_ind, X_test_ind, y_train_ind, y_test_ind, preprocessor_ind,  {
            'model__n_estimators': [100, 500],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1],
            'model__alpha': [0, 0.1],
            'model__gamma': [0, 0.1]
        }),
        'systematic_lr_approach': (LinearRegression(), X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys, {
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, -1]
        }),
        'systematic_xgb_approach': (XGBRegressor(), X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys, {
            'model__n_estimators': [100, 500],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1],
            'model__alpha': [0, 0.1],
            'model__gamma': [0, 0.1]
        })
    }

    for approach_name, (model, X_train, X_test, y_train, y_test, preprocessor, param_grid) in approaches.items():
        # Fit the preprocessor
        preprocessor.fit(X_train)
        print(f"Performing Grid Search for {approach_name}...")
        best_model = optimize_hyperparameters_for_lr_and_xgb(model, param_grid, X_train, y_train, preprocessor)
        save_model_for_lr_and_xgb(best_model, approach_name)

    lstm_approaches = {
        'individual': (feature_names_ind_lstm, X_ind_lstm, y_ind_lstm, X_train_ind_lstm, X_test_ind_lstm, y_train_ind_lstm, y_test_ind_lstm, scaler_ind_lstm_train, le_ind_lstm_train, df_ind_lstm_test, scaler_ind_lstm_pred, le_ind_lstm_pred, predict_2024_season_for_ind_lstm),
        'systematic': (feature_names_sys_lstm, X_sys_lstm, y_sys_lstm, X_train_sys_lstm, X_test_sys_lstm, y_train_sys_lstm, y_test_sys_lstm, scaler_sys_lstm_train, le_sys_lstm_train, df_sys_lstm_test, scaler_sys_lstm_pred, le_sys_lstm_pred, predict_2024_season_for_sys_lstm)
    }

    for approach_name, (feature_names, X_lstm, y_lstm, X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm_train, le_lstm_train, df_lstm_test, scaler_lstm_pred, le_lstm_pred, predict_2024_season) in lstm_approaches.items():
        print(f"Training LSTM model for {approach_name}_lstm_approach...")
        model_lstm = create_model_for_lstm((X_lstm.shape[1], X_lstm.shape[2]))
        model_lstm, history_lstm = train_model_for_lstm(model_lstm, X_train_lstm, y_train_lstm, epochs=50, batch_size=32)
        save_model_artifacts_for_lstm(approach_name, model_lstm, scaler_lstm_train, le_lstm_train)

    print("Analyse abgeschlossen!")


if __name__ == "__main__":
    main()