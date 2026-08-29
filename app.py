import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_FILE = 'model.pkl'

# Global variable to hold the loaded machine learning model
model = None

if os.path.exists(MODEL_FILE):
  model = joblib.load(MODEL_FILE)
  print(f'✅ Successfully loaded machine learning model from {MODEL_FILE}')
else:
  print(f'⚠️ Warning: {MODEL_FILE} not found! Run train_model.py first.')


def extract_scroll_features(telemetry_data):
  """Data Science Pipeline: Converts raw [{position_y, timestamp_ms}] into mathematical features."""
  df = pd.DataFrame(telemetry_data)

  if len(df) < 3:
    return None

  timestamps = df['timestamp_ms'].values
  positions = df['position_y'].values

  delta_time = np.diff(timestamps)
  delta_pos = np.abs(np.diff(positions))

  delta_time = np.where(delta_time == 0, 1.0, delta_time)
  velocities = delta_pos / delta_time

  mean_pause = np.mean(delta_time)
  pause_variance = np.std(delta_time)
  mean_velocity = np.mean(velocities)
  max_velocity = np.max(velocities)

  return {
      'mean_pause_ms': round(float(mean_pause), 2),
      'pause_variance_std': round(float(pause_variance), 2),
      'mean_velocity_px_ms': round(float(mean_velocity), 4),
      'max_velocity_px_ms': round(float(max_velocity), 4),
  }


@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
  data = request.get_json()

  if not data:
    return jsonify({'status': 'error', 'message': 'No data received'}), 400

  features = extract_scroll_features(data)

  if features:
    if model is not None:
      # Prepare feature row for prediction matching the training format
      input_df = pd.DataFrame([[
          features['mean_pause_ms'],
          features['pause_variance_std'],
          features['mean_velocity_px_ms'],
          features['max_velocity_px_ms'],
      ]],
                              columns=[
                                  'mean_pause_ms',
                                  'pause_variance_std',
                                  'mean_velocity_px_ms',
                                  'max_velocity_px_ms',
                              ])

      # Make real-time prediction (0 or 1)
      prediction = int(model.predict(input_df)[0])

      # Get prediction probability (confidence percentage)
      probabilities = model.predict_proba(input_df)[0]
      confidence = round(float(np.max(probabilities)) * 100, 2)

      status_label = (
          'ACTIVE READER' if prediction == 1 else 'MINDLESS SKIMMER / BOT'
      )

      print('\n--- 🤖 Real-Time ML Prediction ---')
      print(f'Predicted State : {status_label} (Label {prediction})')
      print(f'Confidence      : {confidence}%')
      print(f'Pause Variance  : {features["pause_variance_std"]}')
      print(f'Average Speed   : {features["mean_velocity_px_ms"]} px/ms')
      print('-----------------------------------')

      return jsonify({
          'status': 'success',
          'features': features,
          'prediction': prediction,
          'status_label': status_label,
          'confidence': confidence,
      }), 200
    else:
      return jsonify({
          'status': 'error',
          'message': 'Model not trained yet.',
      }), 500
  else:
    return jsonify({
        'status': 'notice',
        'message': 'Packet too small for prediction.',
    }), 200


if __name__ == '__main__':
  print('🚀 version 6 Live Inference Engine running on http://127.0.0.1:5000')
  app.run(debug=True, port=5000)