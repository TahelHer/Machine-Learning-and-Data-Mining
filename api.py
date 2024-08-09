



from flask import Flask, request, jsonify, render_template, session
import joblib
import pandas as pd
from car_data_prep import prepare_data
from sklearn.preprocessing import StandardScaler
import logging
import random
from datetime import datetime

app = Flask(__name__)
# app.secret_key = 'your_secret_key_here'  # נדרש עבור sessions

logging.basicConfig(level=logging.DEBUG)

def load_model_and_scaler():
    app.logger.debug("טוען מודל וסקיילר")
    loaded_objects = joblib.load('trained_model.pkl')
    if isinstance(loaded_objects, (list, tuple)) and len(loaded_objects) >= 2:
        return loaded_objects[0], loaded_objects[1]
    else:
        raise ValueError("הקובץ 'trained_model.pkl' אינו מכיל את הנתונים הצפויים")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.debug("התקבלה בקשת חיזוי חדשה")
        app.logger.debug(f"Received request data: {request.form}")
        
        # טען את המודל והסקיילר בכל בקשה
        model, scaler = load_model_and_scaler()
        
        # איסוף נתוני הטופס
        form_data = request.form.to_dict()
        app.logger.debug(f"נתונים שהתקבלו: {form_data}")
        
        # המרת נתוני הטופס לרשימת מאפיינים
        features = [
            form_data.get('manufactor', ''),
            form_data.get('Year', ''),
            form_data.get('model', ''),
            form_data.get('Hand', ''),
            form_data.get('Gear', ''),
            form_data.get('capacity_Engine', ''),
            form_data.get('Engine_type', ''),
            form_data.get('Prev_ownership', ''),
            form_data.get('Curr_ownership', ''),
            form_data.get('Pic_num', ''),
            form_data.get('Description', ''),
            form_data.get('Color', ''),
            form_data.get('is_reposted', '')
        ]
        
        # יצירת DataFrame מנתוני המשתמש
        user_data_df = pd.DataFrame([features], columns=[
            'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type',
            'Prev_ownership', 'Curr_ownership', 'Pic_num', 'Description', 'Color', 'is_reposted'
        ])
        
        app.logger.debug(f"User data DataFrame: {user_data_df}")
        
        # טעינת הדאטאסט הקיים
        existing_dataset = pd.read_csv('dataset.csv')
        
        # חיבור נתוני המשתמש עם הדאטאסט הקיים, כאשר נתוני המשתמש בשורה הראשונה
        combined_dataset = pd.concat([user_data_df, existing_dataset], ignore_index=True)
        
        # הכנת הנתונים באמצעות פונקציית prepare_data
        prepared_data = prepare_data(combined_dataset, 1)
        app.logger.debug(f"נתונים מוכנים: {prepared_data}")
        
        # התאמת המאפיינים לאלו שהמודל אומן עליהם
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        else:
            model_features = prepared_data.columns
        
        prepared_data = prepared_data.reindex(columns=model_features, fill_value=0)
        
        # וידוא שכל הערכים הם מספריים
        prepared_data = prepared_data.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        app.logger.debug(f"Final prepared data: {prepared_data}")
        
        # נרמול הנתונים
        prepared_data = pd.DataFrame(scaler.transform(prepared_data), columns=prepared_data.columns)
        
        # ביצוע תחזית עבור השורה הראשונה (נתוני המשתמש)
        prediction = model.predict(prepared_data.iloc[0:1])
        app.logger.debug(f"תוצאת החיזוי: {prediction[0]}")
        
        # החזרת התוצאה
        result = {'prediction_text': f'{prediction[0]:.2f} ₪'}
        app.logger.debug(f"Sending response: {result}")
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"שגיאה בעת ביצוע החיזוי: {str(e)}", exc_info=True)
        return jsonify(error=str(e))

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    response.headers['Last-Modified'] = datetime.now()
    return response

if __name__ == "__main__":
    app.run(debug=True)


