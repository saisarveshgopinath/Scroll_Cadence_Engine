import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_FILE = 'scroll_data.csv'
MODEL_FILE = 'model.pkl'


def train_cadence_model():
  if not os.path.exists(DATASET_FILE):
    print(f'❌ Error: {DATASET_FILE} not found!')
    return
  df = pd.read_csv(DATASET_FILE)

  if 'backscroll_count' not in df.columns:
    df['backscroll_count'] = 0
    df['backscroll_ratio'] = 0.0

  feature_columns = [
      'mean_pause_ms',
      'pause_variance_std',
      'mean_velocity_px_ms',
      'max_velocity_px_ms',
      'backscroll_count',
      'backscroll_ratio',
  ]

  X = df[feature_columns]
  y = df['label']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print('--- 🌲 Day 8 Upgrade: Retrained with Back-Scroll Features ---')
  print(f'Model Accuracy: {round(accuracy * 100, 2)}%\n')

  joblib.dump(model, MODEL_FILE)
  print(f'✅ Updated model saved to {MODEL_FILE}')


if __name__ == '__main__':
  train_cadence_model()