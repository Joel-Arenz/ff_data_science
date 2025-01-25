import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



def create_preprocessor(df_merged):

    X = df_merged.drop(columns=['fantasy_points'])

    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor



def create_pipeline(model, preprocessor):

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    return pipeline



def optimize_hyperparameters(model, param_grid, X_train, y_train, preprocessor):

    pipeline = create_pipeline(model, preprocessor)
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=TimeSeriesSplit(n_splits=3), verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_



def save_model(model, model_name):
    # Speicherpfad definieren
    file_path = f"models/{model_name}_model.pkl"
    joblib.dump(model, file_path)
    print(f"Modell '{model_name}' wurde unter '{file_path}' gespeichert.")

