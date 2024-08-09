import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
import os
import sys

# הגדרת נתיב התסריט
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)
from car_data_prep import prepare_data

# פונקציה לחישוב RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# פונקציה להכנת המודל
def train_and_save_model(df):
    X_dropped = df[['Price', 'manufactor', 'Gear', 'model', 'Engine_type', 'Area', 'City', 'Price', 'Pic_num', 'Cre_date', 
        'Repub_date', 'Description', 'Region']]
    X = df.drop(columns=['Price', 'manufactor', 'Gear', 'model', 'Engine_type', 'Area', 'City', 'Price', 'Pic_num', 'Cre_date', 
        'Repub_date', 'Description', 'Region'])
    y = df['Price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_final = pd.concat([X_scaled_df, X_dropped.reset_index(drop=True)], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    elastic_net = ElasticNet(random_state=42)
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring=rmse_scorer, cv=10)
    grid_search.fit(X_scaled_df, y)
    
    best_model = ElasticNet(**grid_search.best_params_, random_state=42)
    best_model.fit(X_scaled_df, y) 

    # שמירת המודל והסקיילר יחד עם שמות התכונות
    models_dir = os.path.join(script_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    joblib.dump((best_model, scaler, X_train.columns.tolist()), os.path.join(models_dir, 'model_trained.pkl'))

# טעינת ה-dataset והכנתו
df = pd.read_csv("dataset.csv")
len_df1 = len(df)
df = prepare_data(df, len_df1)

# יצירת המודל ושמירתו
train_and_save_model(df)
