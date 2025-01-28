from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lstm_data_loader import load_data, edit_data, create_train_and_test_data, prepare_data_for_training, prepare_data_for_prediction_and_outcome, create_sequences, create_model, train_model, analyze_model_performance, save_model_artifacts, predict_2024_season, analyze_predictions
from tensorflow.keras.callbacks import EarlyStopping


def main():
    # Lade die Trainingsdaten
    df = load_data()
    df = edit_data(df)
    df_train, df_seq = create_train_and_test_data(df)
    
    # Daten vorbereiten
    scaled_features, target, scaler_train, le_train = prepare_data_for_training(df_train)
    df_pred, scaler_pred, le_pred = prepare_data_for_prediction_and_outcome(df_seq)
    X, y = create_sequences(scaled_features, target)
    
    # Train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:] 

    # Modell erstellen
    model = create_model((X.shape[1], X.shape[2]))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Modell trainieren
    model, history = train_model(model, X_train, y_train, epochs=50, batch_size=32)

    analyze_model_performance(history)

    # Modell speichern
    save_model_artifacts(model, scaler_train, le_train)
    
    # Vorhersagen für 2024 treffen
    predictions_2024 = predict_2024_season(model, df_pred, scaler_pred, le_pred)
    
    # Ergebnisse analysieren
    analyze_predictions(predictions_2024, df_seq)
    
    print("Analyse abgeschlossen!")
    

if __name__ == "__main__":
    main()
