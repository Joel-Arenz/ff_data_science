import pandas as pd

def predict_2024_season_for_lstm(model, prepared_df, scaler, le, sequence_length=6):
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
            sequence_features = sequence.drop(['player_display_name', 'position', 'recent_team', 'opponent_team', 'depth_team', 'status', 'fantasy_points', 'original_player_id', 'week', 'season'], axis=1)
            
            # Reshape for LSTM
            X = sequence_features.values.reshape(1, sequence_length, sequence_features.shape[1])
            
            # Predict
            pred = model.predict(X, verbose=0)
            
            # Store prediction
            predictions.append({
                'player_id': player,
                'player_display_name': current_week['player_display_name'].values[0],
                'recent_team': current_week['recent_team'].values[0],
                'opponent_team': current_week['opponent_team'].values[0],
                'position': current_week['position'].values[0],
                'season': current_week['season'].values[0],
                'depth_team': current_week['depth_team'].values[0],
                'status': current_week['status'].values[0],
                'week': week,
                'predicted_fantasy_points': float(pred[0][0]),
                'fantasy_points': current_week['fantasy_points'].values[0]
            })
    
    return pd.DataFrame(predictions)