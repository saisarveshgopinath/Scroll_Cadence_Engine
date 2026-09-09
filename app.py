import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
MODEL_FILE = 'model.pkl'
DATASET_FILE = 'scroll_data.csv'
model = None
if os.path.exists(MODEL_FILE):
  model = joblib.load(MODEL_FILE)
  print(f'✅ Loaded model from {MODEL_FILE}')
def extract_scroll_features(telemetry_data):
  """Data Science Pipeline: Extracts timing, speed, variance, and back-scroll dynamics."""
  df = pd.DataFrame(telemetry_data)
  if len(df) < 3:
    return None
  timestamps = df['timestamp_ms'].values
  positions = df['position_y'].values

  delta_time = np.diff(timestamps)
  delta_pos_raw = np.diff(positions) 
  delta_pos = np.abs(delta_pos_raw)  

  delta_time = np.where(delta_time == 0, 1.0, delta_time)
  velocities = delta_pos / delta_time

  
  mean_pause = np.mean(delta_time)
  pause_variance = np.std(delta_time)
  mean_velocity = np.mean(velocities)
  max_velocity = np.max(velocities)

  #  Back-Scroll Detection
  
  backscroll_events = np.sum(delta_pos_raw < 0)
  total_movements = len(delta_pos_raw)
  backscroll_ratio = (
      backscroll_events / total_movements if total_movements > 0 else 0.0
  )

  return {
      'mean_pause_ms': round(float(mean_pause), 2),
      'pause_variance_std': round(float(pause_variance), 2),
      'mean_velocity_px_ms': round(float(mean_velocity), 4),
      'max_velocity_px_ms': round(float(max_velocity), 4),
      'backscroll_count': int(backscroll_events),
      'backscroll_ratio': round(float(backscroll_ratio), 4),
  }


def append_to_csv(features, label):
  """Appends engineered feature rows including back-scroll metrics into scroll_data.csv."""
  row_data = {
      'mean_pause_ms': features['mean_pause_ms'],
      'pause_variance_std': features['pause_variance_std'],
      'mean_velocity_px_ms': features['mean_velocity_px_ms'],
      'max_velocity_px_ms': features['max_velocity_px_ms'],
      'backscroll_count': features['backscroll_count'],
      'backscroll_ratio': features['backscroll_ratio'],
      'label': label,
  }

  df_row = pd.DataFrame([row_data])

  if not os.path.exists(DATASET_FILE):
    df_row.to_csv(DATASET_FILE, index=False)
  else:
    df_row.to_csv(DATASET_FILE, mode='a', header=False, index=False)


@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
  data = request.get_json()

  if not data:
    return jsonify({'status': 'error', 'message': 'No data received'}), 400

  label = int(request.args.get('label', 1))
  features = extract_scroll_features(data)

  if features:
    
    if 'label' in request.args:
      append_to_csv(features, label)

    
    feature_cols = [
        'mean_pause_ms',
        'pause_variance_std',
        'mean_velocity_px_ms',
        'max_velocity_px_ms',
        'backscroll_count',
        'backscroll_ratio',
    ]

    if model is not None and os.path.exists(MODEL_FILE):
      try:
        input_df = pd.DataFrame(
            [[features[col] for col in feature_cols]], columns=feature_cols
        )
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)
      except Exception:
        
        fallback_cols = [
            'mean_pause_ms',
            'pause_variance_std',
            'mean_velocity_px_ms',
            'max_velocity_px_ms',
        ]
        input_df = pd.DataFrame(
            [[features[col] for col in fallback_cols]], columns=fallback_cols
        )
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

      status_label = (
          'ACTIVE READER' if prediction == 1 else 'MINDLESS SKIMMER / BOT'
      )

      print(
          f'\n--- 🔄 Feature 1 Telemetry: Back-Scrolls = {features["backscroll_count"]} ({features["backscroll_ratio"]*100}%) ---'
      )
      print(f'Prediction: {status_label} ({confidence}%)')

      return jsonify({
          'status': 'success',
          'features': features,
          'prediction': prediction,
          'status_label': status_label,
          'confidence': confidence,
      }), 200

  return jsonify({
      'status': 'notice',
      'message': 'Insufficient telemetry events.',
  }), 200


if __name__ == '__main__':
  app.run(debug=True, port=5000)