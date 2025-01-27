import pandas as pd
import glob
import nfl_data_py as nfl
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def load_data():

    df_rank = pd.read_csv('/Users/alexanderarens/Desktop/data_science_project/data/FantasyPros_Overall_ADP_Rankings.csv', encoding='ISO-8859-1',delimiter=';') #adp (average draft pick) ranking
    df_weekly = nfl.import_weekly_data(years=range(2015,2024)) #player data
    df_schedule = nfl.import_schedules(years=range(2015,2024)) #game data
    df_weekly = df_weekly.rename(columns={
        'player_display_name': 'name',
        'recent_team': 'team',
        'opponent_team': 'opponent'
    })

    #clean data
    df_weekly = df_weekly[df_weekly['season_type'] == 'REG'] #only regualer season games
    df_schedule = df_schedule[df_schedule['game_type'] == 'REG']
    relevant_columns = ['season','week','home_team','away_team','home_score','away_score','location',
                            'spread_line','roof','surface','home_coach','away_coach','stadium','game_id'] #first feature selection for better overview
    df_schedule = df_schedule[relevant_columns]

    #prepare df_weekly
    df_weekly['season'] = df_weekly['season'].astype('int64') 
    df_weekly['week'] = df_weekly['week'].astype('int64')
    df_weekly['game_id_home_away'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['team']+'_'+df_weekly['opponent'] #create game_id 
    df_weekly['game_id_away_home'] = df_weekly['season'].astype(str) + '_' + df_weekly['week'].apply(lambda x: f"{x:02d}")+'_'+df_weekly['opponent']+'_'+df_weekly['team'] 

    #relevant columms
    rel_columns = ['player_display_name', 'player_id', 'position', 'season', 'week', 'team', 'opponent', 'fantasy_points']

    #merge df_rank with df_weekly
    df_merged = df_weekly.merge(
        df_rank,
        on=['name','season'],
        how='left')
    #prepare df
    df_merged = pd.melt(
        df_merged,
        id_vars=['player_id','name', 'position', 'season', 'week','team', 'opponent', 'fantasy_points','pos','rank'],
        value_vars=['game_id_home_away', 'game_id_away_home'],
        var_name='game_id_type',
        value_name='game_id')
    #merge df_merged with df_schedules
    df_merged = pd.merge(df_merged, 
                        df_schedule[relevant_columns], 
                        on= 'game_id',
                        how='inner')

    #Create df for home teams
    home_df = df_schedule[['game_id','season','week','home_team','home_score','away_team','away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'points_scored', 'away_team': 'opponent', 'away_score': 'points_allowed'}
    ).assign(location='home')

    #Create df for away teams
    away_df = df_schedule[['game_id', 'season', 'week', 'away_team', 'away_score', 'home_team', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'points_scored', 'home_team': 'opponent', 'home_score': 'points_allowed'}
    ).assign(location='away')

    #add the two df above together to have one row per team and game (instead of only one row per game)
    df_combined = pd.concat([home_df, away_df], ignore_index=True)

    # Rolling average for points scored (offense)
    df_combined = df_combined.sort_values(by=['team', 'season', 'week'])
    df_combined['avg_points_scored'] = (
        df_combined.groupby('team')['points_scored']
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(drop=True))
    # Rolling average for points allowed (defense)
    df_combined['avg_points_allowed'] = (
        df_combined.groupby('team')['points_allowed']
        .shift(1)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(drop=True))

    df_merged = df_merged.rename(columns={
        'season_x': 'season',
        'week_x': 'week'})
    df_merged = df_merged.drop(['season_y','week_y'], axis=1)

    ##merge with the rest
    df_merged = pd.merge(
        df_merged,
        df_combined[['season','week','team','points_allowed', 'points_scored', 'avg_points_scored', 'avg_points_allowed']],
        on=['season', 'week', 'team'],
        how='left')

    # Create the dummy variable for home games
    df_merged['is_home_game'] = (df_merged['home_team'] == df_merged['team']).astype(int)
    
    #drop missing values = players that were not in the adp df
    df_merged = df_merged.dropna(subset=['rank'])

    ##Rank players for each team and position based on their adp

    # Sort the players by their draft rank (lower rank is better)
    df_merged['rank'] = pd.to_numeric(df_merged['rank'], errors='coerce')  # Ensure rank is numeric
    # Create new column that ranks players within each team and position for every game
    df_merged['ranked_position'] = (
        df_merged.groupby(['team', 'position','game_id'])['rank']  # Group by team and position
        .rank(method='min', ascending=True)  # Rank with smallest value having rank 1
        .astype(int)  # Convert the rank to an integer
    )
    #Formatted column with the position and rank (e.g., 'WR1', 'RB2', etc.)
    df_merged['role'] = df_merged['position'] + df_merged['ranked_position'].astype(str)

    #rolling average of past fantasy points for each role
    df_merged = df_merged.sort_values(by=['team','role','season','week'])
    df_merged['avg_fantasy_points'] = (
        df_merged.groupby(['team','role'])['fantasy_points']
        .apply(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())  # Shift and calculate rolling mean
    )
    df_merged = df_merged.dropna() #drop the first row for every position in df, because of missing value for avg_fantasy_points
    #final seleciton of features

    #create interaction terms for moderatly correlated features
    df_pred['interaction_term'] = df_pred['ranked_position'] * df_pred['avg_fantasy_points']

    #Final feature seleciton
    df_pred = df_pred[['season','week','team','position','ranked_position','opponent','spread_line','roof', 'is_home_game','points_allowed','points_scored','interaction_term', 'avg_fantasy_points','fantasy_points']]

    #one-hot-encoding for categorical variables
    df_pred = pd.get_dummies(df_pred, columns=['team', 'opponent', 'roof','position'], drop_first=True)

    #create interaction terms for positions
    df_pred['qb_points_interaction'] = df_pred['position_QB'] * df_pred['avg_fantasy_points'] #interaction terms to include their combined effects
    df_pred['rank_points_interaction'] = df_pred['ranked_position'] * df_pred['avg_fantasy_points']

    return df_pred