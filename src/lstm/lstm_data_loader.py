import os
import numpy as np
import pandas as pd
import joblib
import nfl_data_py as nfl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def load_data():

    df_ids = nfl.import_ids()
    df_roster = nfl.import_weekly_rosters(list(range(2018, 2025)))
    df_depth = nfl.import_depth_charts(list(range(2018, 2025)))
    df_weekly = nfl.import_weekly_data(list(range(2018, 2025)))
    df_seasonal = nfl.import_seasonal_data(list(range(2017,2024)))
    df_schedule = nfl.import_schedules(list(range(2018, 2025)))
    df_pass_pfr = nfl.import_weekly_pfr('pass', list(range(2018, 2025)))
    df_rush_pfr = nfl.import_weekly_pfr('rush', list(range(2018, 2025)))
    df_rec_pfr = nfl.import_weekly_pfr('rec', list(range(2018, 2025)))
    df_pass_ngs = nfl.import_ngs_data('passing',list(range(2018, 2025)))
    df_rush_ngs = nfl.import_ngs_data('rushing',list(range(2018, 2025)))
    df_snap_counts = nfl.import_snap_counts(list(range(2018, 2025)))

    # Only regular season data for QB, WR, RB, TE
    df_weekly = df_weekly[(df_weekly['season_type'] == 'REG') & (df_weekly['position'].isin(['QB', 'WR', 'RB', 'TE']))].reset_index()
    df_depth = df_depth[(df_depth['game_type'] == 'REG') & (df_depth['depth_position'].isin(['QB', 'WR', 'RB', 'TE']))].reset_index()
    df_schedule = df_schedule[df_schedule['game_type']=='REG'].reset_index()

    # Dom rating from last season as feature for next season prediction
    df_seasonal['season'] = df_seasonal['season'] + 1 

    # Clean df_schedule: A few duplicate rows and Oakland Raiders moved to Las Vegas in 2020
    df_schedule = df_schedule[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
    df_schedule['game_id'] = df_schedule['game_id'].str.replace('OAK', 'LV', regex=False) # Umzug der Oakland Raiders nach Las Vegas in der Saison 2020
    df_schedule['home_team'] = df_schedule['home_team'].str.replace('OAK', 'LV', regex=False) # Umzug der Oakland Raiders nach Las Vegas in der Saison 2020
    df_schedule['away_team'] = df_schedule['away_team'].str.replace('OAK', 'LV', regex=False) # Umzug der Oakland Raiders nach Las Vegas in der Saison 2020
    
    # To figure out whether recent_team is home or away team
    df_weekly['game_id_home_away'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['recent_team']+'_'+df_weekly['opponent_team']
    df_weekly['game_id_away_home'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['opponent_team']+'_'+df_weekly['recent_team']

    # Melt both options home and away game resulting in double amount of rows
    df_merged = pd.melt(
        df_weekly,
        id_vars=['player_id', 'player_display_name', 'sack_yards','position', 'season', 'week', 'recent_team', 'opponent_team', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_2pt_conversions', 'interceptions', 'sack_fumbles_lost', 'sacks', 'passing_air_yards', 'passing_epa', 'pacr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions', 'rushing_fumbles_lost', 'rushing_epa', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_2pt_conversions', 'receiving_fumbles_lost', 'racr', 'wopr', 'receiving_epa', 'fantasy_points'],
        value_vars=['game_id_home_away', 'game_id_away_home'],
        var_name='game_id_type',
        value_name='game_id'
    )

    # Reducing the amount of rows by filtering out game_ids that don't exist; Some games: recent_team = opponent_team
    df_merged = pd.merge(df_merged, df_schedule[['game_id', 'home_team', 'home_score', 'away_score']], on='game_id', how='inner') 

    df_merged = df_merged.sort_values(['player_id', 'season', 'week'])
    
    # Existing entry means player did play in that season and week
    df_merged['did_play'] = 1

    # In order to fill all gaps where players did not play (because of injuries, bye weeks, etc.)
    player_seasons = df_merged[['player_id', 'recent_team', 'season']].drop_duplicates()
    all_weeks = []

    for _, row in player_seasons.iterrows():
        season_weeks = 18 if row['season'] >= 2021 else 17
        weeks = pd.DataFrame({
            'player_id': row['player_id'],
            'recent_team': row['recent_team'],
            'season': row['season'],
            'week': range(1, season_weeks + 1),
        })
        all_weeks.append(weeks)
        
    complete_weeks = pd.concat(all_weeks, ignore_index=True)

    df_merged = pd.merge(complete_weeks, df_merged, on=['player_id', 'recent_team', 'season', 'week'], how='left')

    df_merged['did_play'] = df_merged['did_play'].fillna(0).astype(bool)

    # List of all unique teams
    all_teams = pd.concat([df_schedule['home_team'], df_schedule['away_team']]).unique()

    # In order to create a list containing all bye weeks for each team; Handle bye weeks
    bye_weeks = []

    for season in df_schedule['season'].unique():
        # 18 weeks instead of 17 since season 2021
        season_weeks = 18 if season >= 2021 else 17
        for week in range(1, season_weeks + 1):
            games_in_week = df_schedule[(df_schedule['season'] == season) & (df_schedule['week'] == week)]
            teams_playing = set(games_in_week['home_team']).union(set(games_in_week['away_team']))
            teams_with_bye_week = set(all_teams) - teams_playing
            for team in teams_with_bye_week:
                bye_weeks.append({'recent_team': team, 'season': season, 'week': week, 'opponent_team': 'BYE'})
            for _, game in games_in_week.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']   
                bye_weeks.append({'recent_team': home_team, 'season': season, 'week': week, 'opponent_team': away_team})
                bye_weeks.append({'recent_team': away_team, 'season': season, 'week': week, 'opponent_team': home_team})

    # DataFrame for bye weeks
    df_bye = pd.DataFrame(bye_weeks)

    df_merged = pd.merge(df_merged, df_bye, on=['recent_team', 'season', 'week'], how= 'left')
    df_merged = df_merged.rename(columns={'opponent_team_y': 'opponent_team'})
    df_merged = df_merged.drop(columns=['opponent_team_x'])
    df_merged = df_merged[df_merged['opponent_team'] != 'BYE']

    # Renaming all columns uniformly
    df_ids = df_ids.rename(columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})
    df_pass_ngs = df_pass_ngs.rename(columns={'player_gsis_id': 'player_id'})
    df_rush_ngs = df_rush_ngs.rename(columns={'player_gsis_id': 'player_id'})
    df_depth = df_depth.rename(columns={'gsis_id': 'player_id', 'club_code': 'recent_team'})

    # Merging all dataframes together
    df_merged = pd.merge(df_merged, df_ids[['player_id', 'pfr_player_id', 'draft_ovr', 'draft_year']], on = 'player_id', how = 'inner') # Ein paar Spieler ohne draft_year
    df_merged = pd.merge(df_merged, df_depth[['player_id', 'season', 'week', 'recent_team', 'position', 'depth_team']], on = ['player_id', 'season', 'week', 'recent_team', 'position'], how='left') # Leider ein paar unsaubere Zeilen, deshalb auch merge auf position und recent_team
    df_merged = pd.merge(df_merged, df_roster[['player_id', 'season', 'week', 'status']], on = ['player_id', 'season', 'week'], how= 'left')
    df_merged = pd.merge(df_merged, df_seasonal[['player_id', 'season', 'dom']], on = ['player_id', 'season'], how = 'left')
    df_merged = pd.merge(df_merged, df_pass_pfr[['pfr_player_id', 'season', 'week', 'passing_bad_throws', 'times_pressured']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rec_pfr[['pfr_player_id', 'season', 'week', 'receiving_rat']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rush_pfr[['pfr_player_id', 'season', 'week', 'rushing_broken_tackles']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_pass_ngs[['player_id', 'season', 'week', 'passer_rating', 'aggressiveness']], on = ['player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rush_ngs[['player_id', 'season', 'week', 'efficiency']], on = ['player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_snap_counts[['pfr_player_id', 'season', 'week', 'offense_snaps']], on = ['pfr_player_id', 'season', 'week'], how = 'left')

    # Dropping unnercessairy columns
    df_merged = df_merged.drop(columns=['game_id_type', 'pfr_player_id'])

    # Filling and imputing missing data
    df_merged['depth_team'] = df_merged['depth_team'].fillna(1)
    df_merged['draft_ovr'] = df_merged['draft_ovr'].fillna(260)
    df_merged = df_merged.fillna(0)

    # Dropping duplicate rows where players are listed with two different depth_teams keeping the higher listed depth_team
    df_merged['depth_team'] = df_merged['depth_team'].astype(int)
    df_merged = df_merged.sort_values(by='depth_team').drop_duplicates(subset=['player_id', 'season', 'week'], keep='first')

    # Creating important features
    df_merged['rookie_flag'] = (df_merged['season'] == df_merged['draft_year']).astype(int)
    df_merged['home'] = (df_merged['home_team'] == df_merged['recent_team']).astype(int)
    # df_merged['player_id'] = df_merged['player_id'].str.replace('00-00', '').astype(int)

    # volume_total stats
    df_merged['volume_total'] = (
        df_merged['attempts'] +
        df_merged['carries'] +
        df_merged['targets']
    )

    # aggregate interceptions and fumbles as turnover
    df_merged['turnover'] = (
        df_merged['interceptions'] +
        df_merged['sack_fumbles_lost'] +
        df_merged['rushing_fumbles_lost'] +
        df_merged['receiving_fumbles_lost']
    )

    # aggregate passing, rushing and receiving epa
    df_merged['epa_total'] = (
        df_merged['passing_epa'] + 
        df_merged['rushing_epa'] + 
        df_merged['receiving_epa']
    )

    # aggregate total points scored by a player
    df_merged['points_total'] = (
        (df_merged['rushing_tds'] * 6) + 
        (df_merged['rushing_2pt_conversions'] * 2) + 
        (df_merged['receiving_tds'] * 6) + 
        (df_merged['receiving_2pt_conversions'] * 2) + 
        (df_merged['passing_tds'] * 4) + 
        (df_merged['passing_2pt_conversions'] * 2)
    )

    # aggregate passing, rushing and receiving yards
    df_merged['yards_total'] = (
        df_merged['passing_yards'] +
        df_merged['rushing_yards'] +
        df_merged['receiving_yards']
    )

    # aggregate total player_rating
    df_merged['player_rating_total'] = (
        df_merged['receiving_rat'] +
        df_merged['passer_rating']
    )

    # aggregate air conversion ratio
    df_merged['acr_total'] = (
        df_merged['racr'] +
        df_merged['pacr']
    )

    # target-encode feature position
    position_means = df_merged.groupby(['position', 'season', 'week'])['fantasy_points'].mean().reset_index()
    position_means.rename(columns={'fantasy_points': 'position_encoded'}, inplace=True)
    df_merged = pd.merge(df_merged, position_means, on=['position', 'season', 'week'], how='left')

    # points_scored and points_allowed as metric for how strong recent_team and how weak opponent_team performed recently
    df_merged['recent_team_points_scored'] = df_merged.apply(lambda row: row['home_score'] if row['home'] == 1 else row['away_score'], axis=1)
    df_merged['opponent_team_points_allowed'] = df_merged['recent_team_points_scored']

    df_unique_opponent_team_points_allowed = df_merged.drop_duplicates(subset=['game_id', 'opponent_team', 'opponent_team_points_allowed'])
    df_unique_recent_team_points_scored = df_merged.drop_duplicates(subset=['game_id', 'recent_team', 'recent_team_points_scored'])

    df_unique_opponent_team_points_allowed = df_unique_opponent_team_points_allowed.sort_values(by=['opponent_team', 'season', 'week']).reset_index(drop=True)
    df_unique_recent_team_points_scored = df_unique_recent_team_points_scored.sort_values(by=['recent_team', 'season', 'week']).reset_index(drop=True)

    df_unique_opponent_team_points_allowed['ewm_opponent_team_points_allowed_l3w'] = (
        df_unique_opponent_team_points_allowed.groupby('opponent_team')['opponent_team_points_allowed']
        .apply(lambda x: x.shift(1).ewm(span=3, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )

    for metric in ['min', 'max']:
            df_unique_opponent_team_points_allowed[f"{metric}_opponent_team_points_allowed_l3w"] = (
                df_unique_opponent_team_points_allowed.groupby('opponent_team')['opponent_team_points_allowed']
                .apply(lambda x: x.shift(1).rolling(window=3, min_periods=3).agg(metric))  # shift(1) schließt aktuelle Woche aus
                .reset_index(level=0, drop=True)  # Index zurücksetzen
        )

    df_unique_opponent_team_points_allowed = df_unique_opponent_team_points_allowed.drop(columns=['player_id', 'did_play', 'volume_total', 'player_rating_total', 'acr_total',  'depth_team', 'status', 'sack_yards', 'player_display_name', 'draft_year', 'turnover', 'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'points_total', 'rushing_tds', 'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'passing_2pt_conversions', 'epa_total', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position', 'season', 'week', 'recent_team', 'home_team', 'completions', 'attempts', 'passing_yards', 'sacks', 'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'yards_total', 'receiving_yards', 'racr', 'wopr', 'fantasy_points', 'home_score', 'away_score', 'draft_ovr', 'dom', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'rookie_flag', 'home', 'position_encoded', 'recent_team_points_scored', 'opponent_team_points_allowed'])
    df_merged = pd.merge(df_merged, df_unique_opponent_team_points_allowed, on=['game_id','opponent_team'], how='inner')

    df_unique_recent_team_points_scored['ewm_recent_team_points_scored_l3w'] = (
        df_unique_recent_team_points_scored.groupby('recent_team')['recent_team_points_scored']
        .apply(lambda x: x.shift(1).ewm(span=3, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )

    for metric in ['min', 'max']:
        df_unique_recent_team_points_scored[f"{metric}_recent_team_points_scored_l3w"] = (
            df_unique_recent_team_points_scored.groupby('recent_team')['recent_team_points_scored']
            .apply(lambda x: x.shift(1).rolling(window=3, min_periods=3).agg(metric))  # shift(1) schließt aktuelle Woche aus
            .reset_index(level=0, drop=True)  # Index zurücksetzen
        )

    df_unique_recent_team_points_scored = df_unique_recent_team_points_scored.drop(columns=['player_id', 'did_play', 'volume_total', 'player_rating_total', 'acr_total', 'depth_team', 'status', 'sack_yards', 'player_display_name', 'draft_year', 'turnover', 'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'points_total', 'rushing_tds', 'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'passing_2pt_conversions', 'epa_total', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position', 'season', 'week', 'opponent_team', 'home_team', 'completions', 'attempts', 'yards_total', 'passing_yards', 'sacks', 'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'receiving_yards', 'racr', 'wopr', 'fantasy_points', 'home_score', 'away_score', 'draft_ovr', 'dom', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'rookie_flag', 'home', 'position_encoded', 'recent_team_points_scored', 'opponent_team_points_allowed'])
    df_merged = pd.merge(df_merged, df_unique_recent_team_points_scored, on=['game_id','recent_team'], how='inner')


    # Assure that df_merged is sorted correctly
    df_merged = df_merged.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    # Calculating streak that a player played in consecutive games ignoring bye weeks
    def mark_streaks_with_bye(group):
        streak = 0
        streak_list = []
        for _, row in group.iterrows():
            if row['did_play']:
                streak += 1
            else:
                streak = 0
            streak_list.append(streak)
        group['streak'] = streak_list
        return group

    df_merged = df_merged.groupby('player_id').apply(mark_streaks_with_bye)

    # list of all relevant stats / columns that need to be considered as rolling features
    columns_to_roll = ['volume_total', 'player_rating_total', 'aggressiveness', 'efficiency', 'acr_total', 'offense_snaps', 
                        'yards_total', 'fantasy_points', 'passing_bad_throws', 'times_pressured', 'position_encoded', 
                        'rushing_broken_tackles', 'turnover', 'points_total', 'epa_total']

    # Again: Sort correctly
    df_merged = df_merged.sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)

    # cnt_games_over_20ffpts_l4w
    df_merged['cnt_games_over_20ffpts_l3w'] = (
        df_merged
        .groupby('player_id')['fantasy_points']
        .apply(lambda x: x.where(df_merged.loc[x.index, 'did_play']) 
            .shift(1)  # Shifting the feature to next week in order to avoid leakage (don't know the stats of the week we want to predict)
            .rolling(window=3, min_periods=3)
            .apply(lambda y: (y > 20).sum()))
    )

    for col in columns_to_roll:
    
        feature_name_1 = f"ewm_{col}_l3w"
        df_merged[feature_name_1] = (
            df_merged
            .groupby('player_id')[col]
            .apply(lambda x: x.where(df_merged.loc[x.index, 'did_play'])
                .shift(1)
                .ewm(span=3, min_periods=3)
                .mean())
        )

        for metric in ['max', 'min']:
            feature_name_3 = f"{metric}_{col}_l3w"
            df_merged[feature_name_3] = (
                df_merged
                .groupby('player_id')[col]
                .apply(lambda x: x.where(df_merged.loc[x.index, 'did_play'])
                    .shift(1)
                    .rolling(window=3, min_periods=3)
                    .agg(metric))
            )
    
    df_merged['time_index'] = df_merged['season'] * 100 + df_merged['week']
    df_merged = df_merged.fillna(0)

    return df_merged

def edit_data(df):

    df = df.drop(columns=['completions', 'attempts', 'passing_yards', 'sacks', 'depth_team', 'sack_yards', 'player_display_name',
                          'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'receiving_yards', 
                          'racr', 'wopr', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 
                          'draft_year', 'home_team', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'game_id', 
                          'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'rushing_tds', 
                          'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'status',
                          'passing_2pt_conversions', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position_encoded', 'acr_total',
                          'home_score', 'away_score', 'position', 'opponent_team', 'recent_team', 'recent_team_points_scored',
                          'turnover', 'points_total', 'yards_total', 'epa_total', 'volume_total', 'opponent_team_points_allowed',
                          'player_rating_total', 'did_play'])
    
    df = df.sort_values(['player_id', 'season', 'week'])

    return df

def create_train_and_test_data(df):
    df_train = df[df['time_index'] < 202401]
    df_seq = df.copy()
    return df_train, df_seq

def prepare_data_for_training(df):
    """
    Bereitet die Daten für das LSTM-Modell vor.
    """
    df = df.sort_values(['player_id', 'season', 'week'])
    
    le = LabelEncoder()
    player_id_encoded = le.fit_transform(df['player_id']).reshape(-1, 1)
    
    features = df.drop(['fantasy_points', 'player_id', 'week', 'season'], axis=1)
    target = df['fantasy_points']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    final_features = np.hstack([player_id_encoded, scaled_features])
    feature_names = ['player_id_encoded'] + list(features.columns)
    final_df = pd.DataFrame(final_features, columns=feature_names, index=df.index)
    
    return final_df.values, target.values, scaler, le

def prepare_data_for_prediction_and_outcome(df_seq):
    """
    Prepare all data (2018-2024) for sequence prediction
    """
    # Sort chronologically
    df_sorted = df_seq.sort_values(['player_id', 'season', 'week'])
    
    # Encode player_ids
    le = LabelEncoder()
    player_id_encoded = le.fit_transform(df_sorted['player_id']).reshape(-1, 1)
    
    # Scale features
    features = df_sorted.drop(['fantasy_points', 'player_id', 'week', 'season'], axis=1)
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
    prepared_df['season'] = df_sorted['season']
    prepared_df['week'] = df_sorted['week']
    prepared_df['fantasy_points'] = df_sorted['fantasy_points']
    
    return prepared_df, scaler, le

def create_sequences(data, target, seq_length=4):
    """
    Erstellt Sequenzen für das LSTM-Modell.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def create_model(input_shape):
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

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
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

def save_model_artifacts(model, scaler, le, models_path='models'):
    """
    Speichert das Modell und die Skalierungsobjekte.
    """
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    save_model(model, os.path.join(models_path, 'lstm_model.h5'))
    joblib.dump(scaler, os.path.join(models_path, 'scaler.joblib'))
    joblib.dump(le, os.path.join(models_path, 'label_encoder.joblib'))

def load_model_artifacts(models_path='models'):
    """
    Lädt das gespeicherte Modell und die Skalierungsobjekte.
    """
    model = load_model(os.path.join(models_path, 'lstm_model.h5'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
    le = joblib.load(os.path.join(models_path, 'label_encoder.joblib'))
    return model, scaler, le

def predict_2024_season(model, prepared_df, scaler, le, sequence_length=6):
    """
    Predict 2024 season using prepared data
    """
    predictions = []
    
    # Get 2024 weeks
    weeks_2024 = sorted(prepared_df[prepared_df['season'] == 2024]['week'].unique())
    
    for week in weeks_2024:
        # Get active players in current week
        current_players = prepared_df[
            (prepared_df['season'] == 2024) & 
            (prepared_df['week'] == week)
        ]['original_player_id'].unique()
        
        for player in current_players:
            # Get historical sequence
            player_history = prepared_df[
                (prepared_df['original_player_id'] == player) & 
                ((prepared_df['season'] < 2024) | 
                 ((prepared_df['season'] == 2024) & (prepared_df['week'] < week)))
            ].tail(sequence_length-1)
            
            if len(player_history) < sequence_length-1:
                continue
                
            # Get current week features
            current_week = prepared_df[
                (prepared_df['original_player_id'] == player) & 
                (prepared_df['season'] == 2024) & 
                (prepared_df['week'] == week)
            ]
            
            # Combine sequence
            sequence = pd.concat([player_history, current_week])
            sequence_features = sequence.drop(['original_player_id', 'season', 'week', 'fantasy_points'], axis=1)
            
            # Reshape for LSTM
            X = sequence_features.values.reshape(1, sequence_length, sequence_features.shape[1])
            
            # Predict
            pred = model.predict(X, verbose=0)
            
            # Store prediction
            predictions.append({
                'player_id': player,
                'week': week,
                'predicted_fantasy_points': float(pred[0][0])
            })
    
    return pd.DataFrame(predictions)

def predict(model, prepared_df, sequence_length=6):
    """
    Macht Vorhersagen für die Saison 2024.
    """
    predictions = []
    weeks_2024 = sorted(prepared_df[prepared_df['season'] == 2024]['week'].unique())
    
    for week in weeks_2024:
        current_players = prepared_df[(prepared_df['season'] == 2024) & (prepared_df['week'] == week)]['original_player_id'].unique()
        
        for player in current_players:
            player_history = prepared_df[
                (prepared_df['original_player_id'] == player) & 
                ((prepared_df['season'] < 2024) | ((prepared_df['season'] == 2024) & (prepared_df['week'] < week)))
            ].tail(sequence_length-1)
            
            if len(player_history) < sequence_length-1:
                continue
            
            current_week = prepared_df[
                (prepared_df['original_player_id'] == player) & 
                (prepared_df['season'] == 2024) & 
                (prepared_df['week'] == week)
            ]
            
            sequence = pd.concat([player_history, current_week])
            sequence_features = sequence.drop(['original_player_id', 'season', 'week', 'fantasy_points'], axis=1)
            X = sequence_features.values.reshape(1, sequence_length, sequence_features.shape[1])
            pred = model.predict(X, verbose=0)
            predictions.append({'player_id': player, 'week': week, 'predicted_fantasy_points': float(pred[0][0])})
    
    return pd.DataFrame(predictions)

def analyze_model_performance(history):
    """
    Analysiert die Modellperformance.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_predictions(predictions_df, df_seq):
    """
    Vergleicht Vorhersagen mit den echten Werten.
    """
    analysis_df = predictions_df.merge(df_seq[df_seq['season'] == 2024][['player_id', 'week', 'fantasy_points']], on=['player_id', 'week'])
    mae = mean_absolute_error(analysis_df['fantasy_points'], analysis_df['predicted_fantasy_points'])
    mse = mean_squared_error(analysis_df['fantasy_points'], analysis_df['predicted_fantasy_points'])
    r2 = r2_score(analysis_df['fantasy_points'], analysis_df['predicted_fantasy_points'])
    
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    return analysis_df
