import joblib
import os
from tensorflow.keras.models import save_model


def save_model_for_lr_and_xgb(model, approach_name):
    # Speicherpfad definieren
    file_path = f"models/{approach_name}_model.pkl"
    joblib.dump(model, file_path)
    print(f"Modell '{approach_name}' wurde unter '{file_path}' gespeichert.")

def save_model_artifacts_for_lstm(model, scaler, le, models_path='models'):
    """
    Speichert das Modell und die Skalierungsobjekte.
    """
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    save_model(model, os.path.join(models_path, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(models_path, 'scaler.joblib'))
    joblib.dump(le, os.path.join(models_path, 'label_encoder.joblib'))