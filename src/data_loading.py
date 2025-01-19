import nfl_data_py as nfl
import pandas as pd

# volume, acr (passing & receiving) und player_rating (passing & receiving) statistiken zusammengefasst, trotzdem noch passing, receiving und rushing einzelne features
# Totalling 60 features (columns_to_roll_v2 - racr, wopr, pacr, passer_rating, receiving_rat, sacks + acr_total, player_rating_total)

# Evaluation results for model: LinearRegression
# mean_absolute_error: 4.051465824664367
# mean_squared_error: 31.197646889300138
# root_mean_squared_error: 5.585485376339297
# r2_score: 0.39497913312881416
# spearman_rank_correlation: 0.6189654922154307

# Evaluation results for model: XGBRegressor
# mean_absolute_error: 4.061278343200684
# mean_squared_error: 30.90032386779785
# root_mean_squared_error: 5.5588059425354
# r2_score: 0.4007452130317688
# spearman_rank_correlation: 0.6245976453862347



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

    df_weekly = df_weekly[(df_weekly['season_type'] == 'REG') & (df_weekly['position'].isin(['QB', 'WR', 'RB', 'TE']))].reset_index()
    df_depth = df_depth[(df_depth['game_type'] == 'REG') & (df_depth['depth_position'].isin(['QB', 'WR', 'RB', 'TE']))].reset_index()

    df_seasonal['season'] = df_seasonal['season'] + 1

    df_schedule = df_schedule[['game_id', 'home_team', 'home_score', 'away_score']].drop_duplicates()
    df_schedule['game_id'] = df_schedule['game_id'].str.replace('OAK', 'LV', regex=False) # Umzug der Oakland Raiders nach Las Vegas in der Saison 2020
    df_schedule['home_team'] = df_schedule['home_team'].str.replace('OAK', 'LV', regex=False) # Umzug der Oakland Raiders nach Las Vegas in der Saison 2020

    df_weekly['game_id_home_away'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['recent_team']+'_'+df_weekly['opponent_team']
    df_weekly['game_id_away_home'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['opponent_team']+'_'+df_weekly['recent_team']

    df_merged = pd.melt(
        df_weekly,
        id_vars=['player_id', 'player_display_name', 'sack_yards','position', 'season', 'week', 'recent_team', 'opponent_team', 'completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_2pt_conversions', 'interceptions', 'sack_fumbles_lost', 'sacks', 'passing_air_yards', 'passing_epa', 'pacr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_2pt_conversions', 'rushing_fumbles_lost', 'rushing_epa', 'receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_2pt_conversions', 'receiving_fumbles_lost', 'racr', 'wopr', 'receiving_epa', 'fantasy_points'],
        value_vars=['game_id_home_away', 'game_id_away_home'],
        var_name='game_id_type',
        value_name='game_id'
    )

    df_ids = df_ids.rename(columns={'gsis_id': 'player_id', 'pfr_id': 'pfr_player_id'})
    df_pass_ngs = df_pass_ngs.rename(columns={'player_gsis_id': 'player_id'})
    df_rush_ngs = df_rush_ngs.rename(columns={'player_gsis_id': 'player_id'})
    df_depth = df_depth.rename(columns={'gsis_id': 'player_id', 'club_code': 'recent_team'})

    df_merged = pd.merge(df_merged, df_schedule, on='game_id', how='inner') # Bei ein paar Spielen: recent_team = opponent_team
    df_merged = pd.merge(df_merged, df_ids[['player_id', 'pfr_player_id', 'draft_pick', 'draft_year']], on = 'player_id', how = 'inner') # Ein paar Spieler ohne draft_year
    df_merged = pd.merge(df_merged, df_depth[['player_id', 'season', 'week', 'recent_team', 'position', 'depth_team']], on = ['player_id', 'season', 'week', 'recent_team', 'position'], how='left') # Leider ein paar unsaubere Zeilen, deshalb auch merge auf position und recent_team
    df_merged = pd.merge(df_merged, df_roster[['player_id', 'season', 'week', 'status']], on = ['player_id', 'season', 'week'], how= 'left')
    df_merged = pd.merge(df_merged, df_seasonal[['player_id', 'season', 'dom']], on = ['player_id', 'season'], how = 'left')
    df_merged = pd.merge(df_merged, df_pass_pfr[['pfr_player_id', 'season', 'week', 'passing_bad_throws', 'times_pressured']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rec_pfr[['pfr_player_id', 'season', 'week', 'receiving_rat']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rush_pfr[['pfr_player_id', 'season', 'week', 'rushing_broken_tackles']], on = ['pfr_player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_pass_ngs[['player_id', 'season', 'week', 'passer_rating', 'aggressiveness']], on = ['player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_rush_ngs[['player_id', 'season', 'week', 'efficiency']], on = ['player_id', 'season', 'week'], how = 'left')
    df_merged = pd.merge(df_merged, df_snap_counts[['pfr_player_id', 'season', 'week', 'offense_snaps']], on = ['pfr_player_id', 'season', 'week'], how = 'left')

    df_merged = df_merged.drop(columns=['game_id_type', 'pfr_player_id'])

    df_merged['depth_team'] = df_merged['depth_team'].fillna(1)
    df_merged['draft_pick'] = df_merged['draft_pick'].fillna(260)
    df_merged = df_merged.fillna(0)

    df_merged['depth_team'] = df_merged['depth_team'].astype(int)
    df_merged = df_merged.sort_values(by='depth_team').drop_duplicates(subset=['player_id', 'season', 'week'], keep='first')

    df_merged['rookie_flag'] = (df_merged['season'] == df_merged['draft_year']).astype(int)
    # df_merged['last_season_data_flag'] = (df_merged['week'] < 6).astype(int)
    df_merged['home'] = (df_merged['home_team'] == df_merged['recent_team']).astype(int)
    df_merged['player_id'] = df_merged['player_id'].str.replace('00-00', '').astype(int)

     # volume_total statistiken
    df_merged['volume_total'] = (
        df_merged['attempts'] +
        df_merged['carries'] +
        df_merged['targets']
    )

    # interceptions und fumbles aggregiert als turnover
    df_merged['turnover'] = (
        df_merged['interceptions'] +
        df_merged['sack_fumbles_lost'] +
        df_merged['rushing_fumbles_lost'] +
        df_merged['receiving_fumbles_lost']
    )

    # total epa aggregiert statt passing, rushing und receiving einzeln
    df_merged['epa_total'] = (
        df_merged['passing_epa'] + 
        df_merged['rushing_epa'] + 
        df_merged['receiving_epa']
    )

    # total points aggregiert statt passing, rushing und receiving tds und 2pt conversions einzeln
    df_merged['points_total'] = (
        (df_merged['rushing_tds'] * 6) + 
        (df_merged['rushing_2pt_conversions'] * 2) + 
        (df_merged['receiving_tds'] * 6) + 
        (df_merged['receiving_2pt_conversions'] * 2) + 
        (df_merged['passing_tds'] * 4) + 
        (df_merged['passing_2pt_conversions'] * 2)
    )

    # total yards aggregiert statt passing, rushing und receiving einzeln
    df_merged['yards_total'] = (
        df_merged['passing_yards'] +
        df_merged['rushing_yards'] +
        df_merged['receiving_yards']
    )

    # player rating total aggregiert statt passing, rushing und receiving einzeln
    df_merged['player_rating_total'] = (
        df_merged['receiving_rat'] +
        df_merged['passer_rating']
    )

    # total yards aggregiert statt passing, rushing und receiving einzeln
    df_merged['acr_total'] = (
        df_merged['racr'] +
        df_merged['pacr']
    )

    # position target-encoded
    position_means = df_merged.groupby(['position', 'season', 'week'])['fantasy_points'].mean().reset_index()
    position_means.rename(columns={'fantasy_points': 'position_encoded'}, inplace=True)
    df_merged = pd.merge(df_merged, position_means, on=['position', 'season', 'week'], how='left')

    # points_scored und points_allowed als Maß für Stärke eines Teams
    df_merged['recent_team_points_scored'] = df_merged.apply(lambda row: row['home_score'] if row['home'] == 1 else row['away_score'], axis=1)
    df_merged['opponent_team_points_allowed'] = df_merged['recent_team_points_scored']

    df_unique_opponent_team_points_allowed = df_merged.drop_duplicates(subset=['game_id', 'opponent_team', 'opponent_team_points_allowed'])
    df_unique_recent_team_points_scored = df_merged.drop_duplicates(subset=['game_id', 'recent_team', 'recent_team_points_scored'])

    df_unique_opponent_team_points_allowed = df_unique_opponent_team_points_allowed.sort_values(by=['opponent_team', 'season', 'week']).reset_index(drop=True)
    df_unique_recent_team_points_scored = df_unique_recent_team_points_scored.sort_values(by=['recent_team', 'season', 'week']).reset_index(drop=True)

    df_unique_opponent_team_points_allowed['ewm_opponent_team_points_allowed_l5w'] = (
        df_unique_opponent_team_points_allowed.groupby('opponent_team')['opponent_team_points_allowed']
        .apply(lambda x: x.shift(1).ewm(span=5, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )

    # for metric in ['median', 'mean', 'std']:
    #         df_unique_opponent_team_points_allowed[f"{metric}_opponent_team_points_allowed_l5w"] = (
    #             df_unique_opponent_team_points_allowed.groupby('opponent_team')['opponent_team_points_allowed']
    #             .apply(lambda x: x.shift(1).rolling(window=5, min_periods=5).agg(metric))  # shift(1) schließt aktuelle Woche aus
    #             .reset_index(level=0, drop=True)  # Index zurücksetzen
    #     )

    for metric in ['min', 'max']:
            df_unique_opponent_team_points_allowed[f"{metric}_opponent_team_points_allowed_l3w"] = (
                df_unique_opponent_team_points_allowed.groupby('opponent_team')['opponent_team_points_allowed']
                .apply(lambda x: x.shift(1).rolling(window=3, min_periods=3).agg(metric))  # shift(1) schließt aktuelle Woche aus
                .reset_index(level=0, drop=True)  # Index zurücksetzen
        )

    df_unique_opponent_team_points_allowed = df_unique_opponent_team_points_allowed.drop(columns=['player_id', 'volume_total', 'player_rating_total', 'acr_total',  'depth_team', 'status', 'sack_yards', 'player_display_name', 'draft_year', 'turnover', 'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'points_total', 'rushing_tds', 'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'passing_2pt_conversions', 'epa_total', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position', 'season', 'week', 'recent_team', 'home_team', 'completions', 'attempts', 'passing_yards', 'sacks', 'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'yards_total', 'receiving_yards', 'racr', 'wopr', 'fantasy_points', 'home_score', 'away_score', 'draft_pick', 'dom', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'rookie_flag', 'home', 'position_encoded', 'recent_team_points_scored', 'opponent_team_points_allowed'])
    df_merged = pd.merge(df_merged, df_unique_opponent_team_points_allowed, on=['game_id','opponent_team'], how='inner')

    df_unique_recent_team_points_scored['ewm_recent_team_points_scored_l5w'] = (
        df_unique_recent_team_points_scored.groupby('recent_team')['recent_team_points_scored']
        .apply(lambda x: x.shift(1).ewm(span=5, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )

    # for metric in ['median', 'mean', 'std']:
    #         df_unique_recent_team_points_scored[f"{metric}_recent_team_points_scored_l5w"] = (
    #             df_unique_recent_team_points_scored.groupby('recent_team')['recent_team_points_scored']
    #             .apply(lambda x: x.shift(1).rolling(window=5, min_periods=5).agg(metric))  # shift(1) schließt aktuelle Woche aus
    #             .reset_index(level=0, drop=True)  # Index zurücksetzen
    #     )

    for metric in ['min', 'max']:
            df_unique_recent_team_points_scored[f"{metric}_recent_team_points_scored_l3w"] = (
                df_unique_recent_team_points_scored.groupby('recent_team')['recent_team_points_scored']
                .apply(lambda x: x.shift(1).rolling(window=3, min_periods=3).agg(metric))  # shift(1) schließt aktuelle Woche aus
                .reset_index(level=0, drop=True)  # Index zurücksetzen
        )

    df_unique_recent_team_points_scored = df_unique_recent_team_points_scored.drop(columns=['player_id','volume_total', 'player_rating_total', 'acr_total', 'depth_team', 'status', 'sack_yards', 'player_display_name', 'draft_year', 'turnover', 'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'points_total', 'rushing_tds', 'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'passing_2pt_conversions', 'epa_total', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position', 'season', 'week', 'opponent_team', 'home_team', 'completions', 'attempts', 'yards_total', 'passing_yards', 'sacks', 'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'receiving_yards', 'racr', 'wopr', 'fantasy_points', 'home_score', 'away_score', 'draft_pick', 'dom', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'rookie_flag', 'home', 'position_encoded', 'recent_team_points_scored', 'opponent_team_points_allowed'])
    df_merged = pd.merge(df_merged, df_unique_recent_team_points_scored, on=['game_id','recent_team'], how='inner')

    # Liste der Spalten mit Spielerspezifischen numerischen Daten, für die Rolling-Features erstellt werden sollen
    columns_to_roll = ['volume_total', 'player_rating_total', 'aggressiveness', 'efficiency', 'acr_total', 'offense_snaps', 
                       'yards_total', 'fantasy_points', 'passing_bad_throws', 'times_pressured', 'position_encoded', 
                       'rushing_broken_tackles', 'turnover', 'points_total', 'epa_total']


    # Sortiere nach player_id, season und week
    df_merged = df_merged.sort_values(by=['player_id', 'season', 'week']).reset_index(drop=True)


    df_merged['cnt_games_over_20ffpts_l5w'] = (
        df_merged.groupby('player_id')['fantasy_points']
        .apply(lambda x: x.shift(1).rolling(window=5, min_periods=5).apply(lambda y: (y > 20).sum()))
        .reset_index(level=0, drop=True)
    )

    # Rolling-Features erstellen
    for col in columns_to_roll:

        feature_name_1 = f"ewm_{col}_l5w"
        df_merged[feature_name_1] = (
            df_merged.groupby('player_id')[col]
            .apply(lambda x: x.shift(1).ewm(span=5, min_periods=5).mean())
            .reset_index(level=0, drop=True)
        )

        # for metric in ['median', 'mean', 'std']:
        #     feature_name_2 = f"{metric}_{col}_l5w"
        #     rolling_result_5w = (
        #         df_merged.groupby('player_id')[col]
        #             .apply(lambda x: x.shift(1).rolling(window=5, min_periods=5).agg(metric))  # shift(1) schließt aktuelle Woche aus
        #             .reset_index(level=0, drop=True)  # Index zurücksetzen
        #     )
        #     # Einfügen der Rolling-Metrik
        #     df_merged[feature_name_2] = rolling_result_5w

        for metric in ['max', 'min']:
            feature_name_3 = f"{metric}_{col}_l3w"
            # Berechnung der Rolling-Metrik (ohne aktuelle Woche)
            rolling_result_3w = (
                df_merged.groupby('player_id')[col]
                    .apply(lambda x: x.shift(1).rolling(window=3, min_periods=3).agg(metric))  # shift(1) schließt aktuelle Woche aus
                    .reset_index(level=0, drop=True)  # Index zurücksetzen
            )
            # Einfügen der Rolling-Metrik
            df_merged[feature_name_3] = rolling_result_3w
    
    # df_merged = df_merged[(df_merged['depth_team'] == 1) & (df_merged['status'] == 'ACT')] # Nur Starter und aktive Spieler berücksichtigen
    
    df_merged = df_merged.dropna().reset_index(level=0, drop=True)

    return df_merged



def prepare_features(df):
    df = df.drop(columns=['completions', 'player_display_name', 'attempts', 'passing_yards', 'sacks', 'depth_team', 'sack_yards',
                          'passing_air_yards', 'pacr', 'carries', 'rushing_yards', 'receptions', 'targets', 'receiving_yards', 
                          'racr', 'wopr', 'passing_bad_throws', 'times_pressured', 'receiving_rat', 'rushing_broken_tackles', 
                          'draft_year', 'home_team', 'passer_rating', 'aggressiveness', 'efficiency', 'offense_snaps', 'game_id', 
                          'interceptions', 'sack_fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost', 'rushing_tds', 
                          'rushing_2pt_conversions', 'receiving_tds', 'receiving_2pt_conversions', 'passing_tds', 'status',
                          'passing_2pt_conversions', 'passing_epa', 'rushing_epa', 'receiving_epa', 'position_encoded', 'acr_total',
                          'recent_team', 'opponent_team', 'position', 'home_score', 'away_score', 'recent_team_points_scored', 
                          'opponent_team_points_allowed', 'turnover', 'points_total', 'yards_total', 'epa_total', 'volume_total',
                          'player_rating_total'])
    return df




def prepare_output(df):
    df = df[['player_id', 'season', 'week', 'player_display_name', 'position', 'depth_team', 'status', 'recent_team', 'opponent_team', 'fantasy_points']].copy()
    return df