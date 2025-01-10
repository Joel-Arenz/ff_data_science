import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Daten einlesen
def load_data(file_path):
    return pd.read_csv(file_path)

# Lade die CSV-Datei
csv_file = 'fantasy_points_predictions_2024.csv'
data = load_data(csv_file)

@app.route('/')
def index():
    # Filteroptionen
    weeks = sorted(data['week'].unique())
    positions = sorted(data['position'].unique())
    names = sorted(data['name'].unique())

    # Standardwerte für Filter (wenn keine Auswahl getroffen wurde)
    selected_week = request.args.get('week', default=None, type=int)
    selected_position = request.args.get('position', default=None, type=str)
    selected_name = request.args.get('name', default=None, type=str)

    # Filter anwenden (Daten filtern, wenn der Benutzer etwas auswählt)
    filtered_data = data

    if selected_week:
        filtered_data = filtered_data[filtered_data['week'] == selected_week]
    if selected_position:
        filtered_data = filtered_data[filtered_data['position'] == selected_position]
    if selected_name:
        filtered_data = filtered_data[filtered_data['name'] == selected_name]

    # Sortiere nach predicted_fantasy_points
    filtered_data = filtered_data.sort_values(by='predicted_fantasy_points', ascending=False)

    # Zeige die Tabelle im Template an
    return render_template('index.html', data=filtered_data.to_dict(orient='records'),
                           weeks=weeks, positions=positions, names=names,
                           selected_week=selected_week, selected_position=selected_position, selected_name=selected_name)

if __name__ == '__main__':
    app.run(debug=True)


