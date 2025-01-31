import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def create_preprocessor_for_lr_and_xgb(df_merged):

    X = df_merged.drop(columns=['fantasy_points'])

    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

def create_pipeline_for_lr_and_xgb(model, preprocessor):

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    return pipeline

def optimize_hyperparameters_for_lr_and_xgb(model, param_grid, X_train, y_train, preprocessor):

    pipeline = create_pipeline_for_lr_and_xgb(model, preprocessor)
    pipeline.fit(X_train, y_train)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3), verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")    
    return grid_search.best_estimator_

def create_model_for_lstm(input_shape):
    """
    Erstellt ein LSTM-Modell.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model_for_lstm(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Trainiert das LSTM-Modell.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history