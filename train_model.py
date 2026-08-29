import os
import joblib 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATASET_FILE = 'scroll_data.csv'
MODEL_FILE = 'model.pkl'


def train_cadence_model():
  # 1. Check if dataset exists
  if not os.path.exists(DATASET_FILE):
    print(
        f"❌ Error: {DATASET_FILE} not found! Run Day 4 to collect scroll data"
        " first."
    )
    return

  # 2. Load dataset
  df = pd.read_csv(DATASET_FILE)
  print(f"📊 Loaded {len(df)} feature rows from {DATASET_FILE}\n")

  # 3. Separate Features (X) and Target Label (y)
  feature_columns = [
      'mean_pause_ms',
      'pause_variance_std',
      'mean_velocity_px_ms',
      'max_velocity_px_ms',
  ]
  X = df[feature_columns]
  y = df['label']

  # 4. Split data into Training Set (80%) and Test Set (20%)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 5. Initialize and Train the Random Forest Classifier
  # n_estimators=100 builds an ensemble of 100 decision trees
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # 6. Evaluate Model Performance on Test Set
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print("---: Model Training Summary ---")
  print(f"Model Accuracy: {round(accuracy * 100, 2)}%\n")
  print("Detailed Classification Report:")
  print(classification_report(y_test, y_pred, zero_division=0))

  # 7. Export the trained model to disk
  joblib.dump(model, MODEL_FILE)
  print(f"✅ Trained model successfully exported as '{MODEL_FILE}'!")


if __name__ == '__main__':
  train_cadence_model()