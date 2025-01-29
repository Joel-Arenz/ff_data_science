from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pipeline.O1_data_loading import load_ind_data, load_sys_data
from pipeline.O2_data_preparation import create_ind_train_and_test_data_for_lstm, create_sequences_for_lstm, prepare_ind_data_for_lstm_prediction_and_outcome, prepare_ind_features_for_lr_and_xgb, prepare_sys_features_for_lr_and_xgb, split_data_for_lr_and_xgb, prepare_ind_data_for_lstm_training, split_data_for_lstm
from pipeline.O3_pipeline_implementation import create_model_for_lstm, create_preprocessor_for_lr_and_xgb, optimize_hyperparameters_for_lr_and_xgb, train_model_for_lstm
from pipeline.O4_model_evaluation import analyze_model_performance_for_lstm, analyze_predictions_for_lstm, evaluate_models_for_lr_and_xgb, plot_feature_importances_for_lr_and_xgb
from pipeline.O5_model_deployment import save_model_artifacts_for_lstm, save_model_for_lr_and_xgb
from pipeline.O6_predict_functions import predict_2024_season_for_lstm

def main():
    # Prepare individual features for model training and testing
    df_ind = load_ind_data()
    df_ind_prepared = prepare_ind_features_for_lr_and_xgb(df_ind)

    df_ind_lstm_train, df_seq = create_ind_train_and_test_data_for_lstm(df_ind)
    scaled_ind_lstm_features, target, scaler_train, le_train = prepare_ind_data_for_lstm_training(df_ind_lstm_train)
    df_ind_lstm_test, scaler_pred, le_pred = prepare_ind_data_for_lstm_prediction_and_outcome(df_seq)
    X, y = create_sequences_for_lstm(scaled_ind_lstm_features, target)


    # Prepare systematic features for model training and testing
    df_sys = load_sys_data()
    df_sys_prepared = prepare_sys_features_for_lr_and_xgb(df_sys)

    # Split data for training and testing
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = split_data_for_lr_and_xgb(df_ind_prepared)
    X_train_sys, X_test_sys, y_train_sys, y_test_sys = split_data_for_lr_and_xgb(df_sys_prepared)

    X_train_ind_lstm, X_test_ind_lstm, y_train_ind_lstm, y_test_ind_lstm = split_data_for_lstm(X, y)


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
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.05, 0.1]
        }),
        # 'individual_lstm_approach': (X_train_ind, X_test_ind, y_train_ind, y_test_ind, preprocessor_ind),
        'systematic_lr_approach': (LinearRegression(), X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys, {
            'model__fit_intercept': [True, False],
            'model__n_jobs': [1, -1]
        }),
        'systematic_xgb_approach': (XGBRegressor(), X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys, {
            'model__n_estimators': [100, 500],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.05, 0.1]
        })
        # 'systematic_lstm_approach': (X_train_sys, X_test_sys, y_train_sys, y_test_sys, preprocessor_sys)
    }


    for approach_name, (model, X_train, X_test, y_train, y_test, preprocessor, param_grid) in approaches.items():
        # Fit the preprocessor
        preprocessor.fit(X_train)
   
        print(f"Performing Grid Search for {approach_name}...")
        best_model = optimize_hyperparameters_for_lr_and_xgb(model, param_grid, X_train, y_train, preprocessor)
        print(f"Evaluating {approach_name}...")
        evaluate_models_for_lr_and_xgb(best_model, X_test, y_test)
        X_train_transformed = preprocessor.transform(X_train)
        
        # Get feature names after preprocessing
        feature_names = preprocessor.get_feature_names_out()
        plot_feature_importances_for_lr_and_xgb(best_model, X_train_transformed, feature_names)
        
        save_model_for_lr_and_xgb(best_model, approach_name)

    # LSTM
    model = create_model_for_lstm((X.shape[1], X.shape[2]))
    model, history = train_model_for_lstm(model, X_train_ind_lstm, y_train_ind_lstm, epochs=50, batch_size=32)
    analyze_model_performance_for_lstm(history)
    save_model_artifacts_for_lstm(model, scaler_train, le_train)
    predictions_2024 = predict_2024_season_for_lstm(model, df_ind_lstm_test, scaler_pred, le_pred)
    analyze_predictions_for_lstm(predictions_2024)
    
    print("Analyse abgeschlossen!")


if __name__ == "__main__":
    main()