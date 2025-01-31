import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_ind_features_for_lr_and_xgb(df):
    # Only keep rows with valid data (player played in the last 3 consecutive games)
    df = df[df['streak']>3]

    df = df.drop(columns=['completions', 'attempts', 'passing_yards', 'sacks', 'depth_team', 'sack_yards', 'player_display_name',
                          'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'receiving_yards', 
                          'racr', 'wopr', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 
                          'draft_year', 'home_team', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'game_id', 
                          'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'rushing_tds', 
                          'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'status',
                          'passing_2pt_conversions', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position_encoded', 'acr_total',
                          'home_score', 'away_score', 'position', 'opponent_team', 'recent_team', 'recent_team_points_scored',
                          'turnover', 'points_total', 'yards_total', 'epa_total', 'volume_total', 'opponent_team_points_allowed',
                          'player_rating_total', 'did_play', 'season', 'week'])
    return df

def prepare_ind_output_for_lr_and_xgb(df):
    # Only keep rows with valid data (player played in the last 3 consecutive games)
    df = df[df['streak']>3]

    df = df[['season', 'week', 'player_display_name', 'position', 'depth_team', 'status', 'recent_team', 'opponent_team', 'fantasy_points']].copy()
    return df

def prepare_sys_features_for_lr_and_xgb(df):
    #drop the first row for every position in df, because of missing value for avg_fantasy_points
    df = df.dropna() 

    df = df[['time_index', 'recent_team', 'position', 'ranked_position', 'opponent_team', 'spread_line', 'roof', 'home', 'ewm_recent_team_points_scored_l5w', 'min_recent_team_points_scored_l5w', 'max_recent_team_points_scored_l5w', 'ewm_opponent_team_points_allowed_l5w', 'min_opponent_team_points_allowed_l5w', 'max_opponent_team_points_allowed_l5w', 'mean_fantasy_points_l5w','fantasy_points']]
    return df

def prepare_sys_output_for_lr_and_xgb(df):
    #drop the first row for every position in df, because of missing value for avg_fantasy_points
    df = df.dropna() 

    df = df[['season', 'week', 'role', 'recent_team', 'opponent_team', 'fantasy_points']].copy()
    return df

def split_data_for_lr_and_xgb(df_merged):

    X_train = df_merged[df_merged['time_index']<202401].drop(columns=['fantasy_points'])
    y_train = df_merged[df_merged['time_index']<202401]['fantasy_points']

    X_test = df_merged[df_merged['time_index']>202318].drop(columns=['fantasy_points'])
    y_test = df_merged[df_merged['time_index']>202318]['fantasy_points']

    return X_train, X_test, y_train, y_test

def create_ind_train_and_test_data_for_lstm(df):
    df = df.fillna(0)

    df = df.drop(columns=['completions', 'attempts', 'passing_yards', 'sacks', 'sack_yards', 'passing_air_yards', 'pacr', 'carries', 
                          'rushing_yards', 'receptions', 'targets', 'receiving_yards', 'racr', 'wopr', 'passing_bad_throws', 
                          'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 'draft_year', 'home_team', 'passer_rating', 
                          'aggressiveness', 'efficiency', 'offense_snaps', 'game_id', 'interceptions', 'sack_fumbles_lost', 
                          'rushing_fumbles_lost', 'receiving_fumbles_lost', 'rushing_tds', 'rushing_2pt_conversions', 'receiving_tds', 
                          'receiving_2pt_conversions', 'passing_tds', 'passing_2pt_conversions', 'passing_epa', 'rushing_epa', 
                          'receiving_epa', 'position_encoded', 'acr_total','home_score', 'away_score', 'recent_team_points_scored',
                          'turnover', 'points_total', 'yards_total', 'epa_total', 'volume_total', 'opponent_team_points_allowed',
                          'player_rating_total', 'did_play'])
    
    df = df.sort_values(['player_id', 'season', 'week'])

    df_train = df[df['time_index'] < 202401]
    df_seq = df.copy()
    return df_train, df_seq

def prepare_ind_data_for_lstm_training(df):
    """
    Bereitet die Daten für das LSTM-Modell vor.
    """
    df = df.sort_values(['player_id', 'season', 'week'])
    
    le = LabelEncoder()
    player_id_encoded = le.fit_transform(df['player_id']).reshape(-1, 1)
    
    features = df.drop(['player_display_name', 'position', 'recent_team', 'opponent_team', 'depth_team', 'status', 'fantasy_points', 'player_id', 'week', 'season'], axis=1)
    target = df['fantasy_points']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    final_features = np.hstack([player_id_encoded, scaled_features])
    feature_names = ['player_id_encoded'] + list(features.columns)
    final_df = pd.DataFrame(final_features, columns=feature_names, index=df.index)   
    return final_df.values, target.values, scaler, le

def prepare_ind_data_for_lstm_prediction_and_outcome(df_seq):
    """
    Prepare all data (2018-2024) for sequence prediction
    """
    # Sort chronologically
    df_sorted = df_seq.sort_values(['player_id', 'season', 'week'])
    
    # Encode player_ids
    le = LabelEncoder()
    player_id_encoded = le.fit_transform(df_sorted['player_id']).reshape(-1, 1)
    
    # Scale features
    features = df_sorted.drop(['player_display_name', 'position', 'recent_team', 'opponent_team', 'depth_team', 'status', 'fantasy_points', 'player_id', 'week', 'season'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Combine encoded IDs with scaled features
    final_features = np.hstack([player_id_encoded, scaled_features])
    
    # Create final DataFrame with metadata
    feature_names = ['player_id_encoded'] + list(features.columns)
    prepared_df = pd.DataFrame(
        final_features,
        columns=feature_names,
        index=df_sorted.index
    )
    
    # Add back metadata columns
    prepared_df['original_player_id'] = df_sorted['player_id']
    prepared_df['player_display_name'] = df_sorted['player_display_name']
    prepared_df['position'] = df_sorted['position']
    prepared_df['recent_team'] = df_sorted['recent_team']
    prepared_df['opponent_team'] = df_sorted['opponent_team']
    prepared_df['depth_team'] = df_sorted['depth_team']
    prepared_df['status'] = df_sorted['status']
    prepared_df['season'] = df_sorted['season']
    prepared_df['week'] = df_sorted['week']
    prepared_df['fantasy_points'] = df_sorted['fantasy_points']

    prepared_df = prepared_df[prepared_df['player_display_name'] != 0]
    return feature_names, prepared_df, scaler, le

def create_sys_train_and_test_data_for_lstm(df):
    #drop the first row for every position in df, because of missing value for avg_fantasy_points
    df = df.dropna() 

    df = df[['time_index', 'season', 'week', 'recent_team', 'position', 'ranked_position', 'opponent_team', 'spread_line', 'roof', 
             'home', 'ewm_recent_team_points_scored_l5w', 'min_recent_team_points_scored_l5w', 'max_recent_team_points_scored_l5w', 
             'ewm_opponent_team_points_allowed_l5w', 'min_opponent_team_points_allowed_l5w', 'max_opponent_team_points_allowed_l5w', 
             'mean_fantasy_points_l5w', 'fantasy_points', 'role', 'role_id']]
   
    df = df.sort_values(['role_id', 'season', 'week'])

    df_train = df[df['time_index'] < 202401]
    df_seq = df.copy()
    return df_train, df_seq

def prepare_sys_data_for_lstm_training(df):
    """
    Prepare the data for the LSTM model.
    """
    df = df.sort_values(['role_id', 'season', 'week'])
    
    le = LabelEncoder()
    role_id_encoded = le.fit_transform(df['role_id']).reshape(-1, 1)
    recent_team_encoded = le.fit_transform(df['recent_team']).reshape(-1, 1)
    opponent_team_encoded = le.fit_transform(df['opponent_team']).reshape(-1, 1)
    position_encoded = le.fit_transform(df['position']).reshape(-1, 1)
    roof_encoded = le.fit_transform(df['roof']).reshape(-1, 1)
    
    encoded_features = np.hstack([role_id_encoded, recent_team_encoded, opponent_team_encoded, position_encoded, roof_encoded])
    
    features = df.drop(['season', 'week', 'recent_team', 'position', 'opponent_team', 'roof', 'fantasy_points', 'role_id', 'role'], axis=1)
    target = df['fantasy_points']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    final_features = np.hstack([encoded_features, scaled_features])
    feature_names = ['role_id_encoded', 'recent_team_encoded', 'opponent_team_encoded', 'position_encoded', 'roof_encoded'] + list(features.columns)
    final_df = pd.DataFrame(final_features, columns=feature_names, index=df.index)   
    return final_df.values, target.values, scaler, le

def prepare_sys_data_for_lstm_prediction_and_outcome(df_seq):
    """
    Prepare all data (2018-2024) for sequence prediction
    """
    # Sort chronologically
    df_sorted = df_seq.sort_values(['role_id', 'season', 'week'])
    
    le = LabelEncoder()
    role_id_encoded = le.fit_transform(df_seq['role_id']).reshape(-1, 1)
    recent_team_encoded = le.fit_transform(df_seq['recent_team']).reshape(-1, 1)
    opponent_team_encoded = le.fit_transform(df_seq['opponent_team']).reshape(-1, 1)
    position_encoded = le.fit_transform(df_seq['position']).reshape(-1, 1)
    roof_encoded = le.fit_transform(df_seq['roof']).reshape(-1, 1)
    
    encoded_features = np.hstack([role_id_encoded, recent_team_encoded, opponent_team_encoded, position_encoded, roof_encoded])
    
    features = df_seq.drop(['season', 'week', 'recent_team', 'position', 'opponent_team', 'roof', 'fantasy_points', 'role_id', 'role'], axis=1)
    target = df_seq['fantasy_points']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Combine encoded IDs with scaled features
    final_features = np.hstack([encoded_features, scaled_features])
    
    # Create final DataFrame with metadata
    feature_names = ['role_id_encoded', 'recent_team_encoded', 'opponent_team_encoded', 'position_encoded', 'roof_encoded'] + list(features.columns)
    prepared_df = pd.DataFrame(
        final_features,
        columns=feature_names,
        index=df_sorted.index
    )
    
    # Add back metadata columns
    prepared_df['original_role_id'] = df_sorted['role_id']
    prepared_df['role'] = df_sorted['role']
    prepared_df['recent_team'] = df_sorted['recent_team']
    prepared_df['opponent_team'] = df_sorted['opponent_team']
    prepared_df['season'] = df_sorted['season']
    prepared_df['week'] = df_sorted['week']
    prepared_df['fantasy_points'] = df_sorted['fantasy_points']
    return feature_names, prepared_df, scaler, le

def create_sequences_for_lstm(data, target, seq_length=4):
    """
    Erstellt Sequenzen für das LSTM-Modell.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def split_data_for_lstm(X, y):
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:] 
    return X_train, X_test, y_train, y_test



