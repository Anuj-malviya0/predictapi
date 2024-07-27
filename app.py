from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    
    le = LabelEncoder()
    df['Category_encoded'] = le.fit_transform(df['Category'])
    df['Name_encoded'] = le.fit_transform(df['Name'])
    df['Location_encoded'] = le.fit_transform(df['Location'])
    df['Status_encoded'] = le.fit_transform(df['Status'])
    
    return df, le

# Train model
def train_model(df):
    X = df[['Category_encoded', 'Name_encoded', 'Location_encoded', 'Status_encoded']]
    y = df['Quantity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    return model

# Predict quantities
def predict_quantities(model, df):
    X = df[['Category_encoded', 'Name_encoded', 'Location_encoded', 'Status_encoded']]
    predicted_quantities = model.predict(X)
    df['Predicted_Quantity'] = predicted_quantities
    return df

# Get top distinct items
def get_top_distinct_items(df, n=5):
    grouped_df = df.groupby('Name').agg({
        'Quantity': 'sum',
        'Predicted_Quantity': 'sum',
        'Category': 'first',
    }).reset_index()
    
    return grouped_df.nlargest(n, 'Quantity')

# Load data and train model
df, label_encoder = load_and_prepare_data('healthcare.csv')
model = train_model(df)

# Save model and label encoder
joblib.dump(model, 'model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Encode input data
    encoded_data = {
        'Category_encoded': label_encoder.transform([data['Category']])[0],
        'Name_encoded': label_encoder.transform([data['Name']])[0],
        'Location_encoded': label_encoder.transform([data['Location']])[0],
        'Status_encoded': label_encoder.transform([data['Status']])[0]
    }
    
    # Make prediction
    input_df = pd.DataFrame([encoded_data])
    prediction = model.predict(input_df)[0]
    
    return jsonify({'predicted_quantity': prediction})

@app.route('/top_items', methods=['GET'])
def top_items():
    n = request.args.get('n', default=5, type=int)
    df_with_predictions = predict_quantities(model, df)
    top_items = get_top_distinct_items(df_with_predictions, n)
    
    result = []
    for _, row in top_items.iterrows():
        result.append({
            'name': row['Name'],
            'current_quantity': row['Quantity'],
            'predicted_quantity': row['Predicted_Quantity'],
            'category': row['Category']
        })
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)