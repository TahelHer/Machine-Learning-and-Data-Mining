import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:\\Users\\itama\\OneDrive\\מסמכים\\לימודים\\תואר\\שנה ג\\סמסטר ב\\כרייה וניתוח נתונים מתקדם\\פרויקט\\flask מטלה 3 - אפליקציית\\flask app\\dataset.csv')
dataset = pd.read_csv('C:\\Users\\itama\\OneDrive\\מסמכים\\לימודים\\תואר\\שנה ג\\סמסטר ב\\כרייה וניתוח נתונים מתקדם\\פרויקט\\flask מטלה 3 - אפליקציית\\flask app\\dataset.csv')

def km(df):
    if 'Km' in df.columns:
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
        df['Km'] = df['Km'].astype('float64')
        df['Km'] = df['Km'].fillna(df['Km'].mean())
    else:
        df['Km'] = pd.to_numeric(dataset['Km'], errors='coerce').mean()
        df['Km'] = df['Km'].astype('float64')
    return df

def filling(df):
    df = km(df)
    if 'manufactor' not in df.columns:
        df['manufactor'] = dataset['manufactor'].mode()[0]
    df['manufactor'] = df['manufactor'].fillna(df['manufactor'].mode()[0])

    if 'capacity_Engine' not in df.columns:
        df['capacity_Engine'] = dataset['capacity_Engine'].mode()[0]
    df['capacity_Engine'] = df['capacity_Engine'].str.strip().str.replace(',', '')
    df['capacity_Engine'] = df['capacity_Engine'].fillna(df['capacity_Engine'].mode()[0])

    if 'Engine_type' not in df.columns:
        df['Engine_type'] = dataset['Engine_type'].mode()[0]
    df['Engine_type'] = df['Engine_type'].fillna(df['Engine_type'].mode()[0])

    if 'Year' not in df.columns:
        df['Year'] = dataset['Year'].mode()[0]
    df['Year'] = df['Year'].fillna(df['Year'].mode()[0])

    if 'Hand' not in df.columns:
        df['Hand'] = dataset['Hand'].mode()[0]
    df['Hand'] = df['Hand'].fillna(df['Hand'].mode()[0])

    if 'Gear' not in df.columns:
        df['Gear'] = 'אוטומטית'
    df['Gear'] = df['Gear'].fillna(df['Gear'].mode()[0])

    if 'Pic_num' not in df.columns:
        df['Pic_num'] = '0'
    df['Pic_num'] = df['Pic_num'].fillna('0')

    if 'Area' not in df.columns:
        df['Area'] = dataset['Area'].mode()[0]
    df['Area'] = df['Area'].fillna(df['Area'].mode()[0])

    return df

warnings.filterwarnings("ignore")
    
def low_data(df):
    threshold = 0.5
    threshold_count = int(threshold * len(dataset))
    df.dropna(axis=1, thresh=threshold_count, inplace=True)
    return df

def Clearing_Outliers_By_IQR(df, colName, lp, up):
    if colName not in df.columns or df[colName].empty:
        return df
    df[colName] = pd.to_numeric(df[colName], errors='coerce')
    df = df.dropna(subset=[colName])
    if len(df) == 0:
        return df
    try:
        Q1 = np.percentile(df[colName].dropna(), lp)
        Q3 = np.percentile(df[colName].dropna(), up)
    except IndexError:
        return df
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[colName] >= lower_bound) & (df[colName] <= upper_bound)]
    
    return df

import logging

logging.basicConfig(level=logging.DEBUG)

import pandas as pd

def feature_bins(df):
    if 'Km' in df.columns:
        # Remove NaN values from the 'Km' column
        km_values = df['Km'].dropna()
        
        if km_values.empty:
            # Handle empty 'Km' column case
            df['Km_binned'] = pd.NA
        else:
            # Determine the number of bins based on the data
            num_bins = min(len(km_values.unique()), 10)  # Use unique values to avoid too few bins
            if num_bins <= 1:
                # If not enough unique values for binning
                df['Km_binned'] = pd.NA
            else:
                df['Km_binned'] = pd.cut(km_values, bins=num_bins, labels=False)
    else:
        # Handle missing 'Km' column case
        df['Km_binned'] = pd.NA

    return df


def map_area_to_region(area):
    area_bins = {
        'אזור המרכז': ['תל אביב', 'ראשון לציון', 'רמת גן', 'גבעתיים', 'חולון', 'בת ים', 'פתח תקוה והסביבה', 'בקעת אונו', 'מודיעין והסביבה', 'ראש העין והסביבה'],
        'שרון': ['רעננה - כפר סבא', 'מושבים בשרון', 'רמת השרון - הרצליה', 'הוד השרון והסביבה', 'נתניה והסביבה', 'קיסריה והסביבה', 'חדרה וישובי עמק חפר'],
        'דרום': ['אשדוד - אשקלון', 'באר שבע והסביבה', 'אילת והערבה', 'מושבים בדרום', 'מושבים בשפלה'],
        'צפון': ['חיפה וחוף הכרמל', 'קריות', 'עכו - נהריה', 'טבריה והסביבה', 'גליל ועמקים', 'עמק יזרעאל', 'כרמיאל והסביבה', 'זכרון - בנימינה', 'יישובי השומרון'],
        'בקעה ושפלה': ['נס ציונה - רחובות', 'ראשל"צ והסביבה', 'גדרה יבנה והסביבה', 'רמלה - לוד', 'בית שמש והסביבה', 'מושבים במרכז']
    }

    for region, areas in area_bins.items():
        if any(area.startswith(a) for a in areas):
            return region
    return 'אזור לא מוגדר'

def feature_area(df):
    # Convert the 'Area' column to strings and handle missing values
    df['Area'] = df['Area'].astype(str).fillna('')

    # Apply the map_area_to_region function
    df['Region'] = df['Area'].apply(map_area_to_region).astype(str)
    return df


def feature_gear(df,len_df):
    df_gear = df.copy()
    df_gear = df_gear.loc[:, ['Gear']]
    min_frequency_value = max(int(0.45 * len_df), 1)  # Ensure min_frequency is at least 1
    gear_encoder = OneHotEncoder(min_frequency=min_frequency_value, sparse_output=False)
    gear_encoded = gear_encoder.fit_transform(df_gear)
    gear_feature_names = gear_encoder.get_feature_names_out(df_gear.columns)
    gear_encoded = pd.DataFrame(gear_encoded, columns=gear_feature_names, index=df_gear.index)
    
    return gear_encoded


def feature_engine(df, len_df):
    df_engine_type = df.copy()
    df_engine_type = df_engine_type.loc[:, ['Engine_type']]
    min_freq = max(1, int(0.45 * len_df))  # Ensure min_frequency is at least 1
    df_engine_type_encoder = OneHotEncoder(min_frequency=min_freq, sparse_output=False)
    engine_type_encoded = df_engine_type_encoder.fit_transform(df_engine_type)
    engine_type_feature_names = df_engine_type_encoder.get_feature_names_out(df_engine_type.columns)
    engine_type_encoded = pd.DataFrame(engine_type_encoded, columns=engine_type_feature_names, index=df_engine_type.index)
    return engine_type_encoded


def feature_manufactor(df, len_df):
    if 'manufactor' not in df.columns:
        df['manufactor'] = dataset['manufactor'].mode()[0]
    df_manufactor = df.copy()
    df_manufactor = df_manufactor.loc[:, ['manufactor']]
    min_freq = max(1, int(0.45 * len_df))  # Ensure min_frequency is at least 1
    df_manufactor_encoder = OneHotEncoder(min_frequency=min_freq, sparse_output=False)
    manufactor_encoded = df_manufactor_encoder.fit_transform(df_manufactor)
    manufactor_feature_names = df_manufactor_encoder.get_feature_names_out(df_manufactor.columns)
    manufactor_encoded = pd.DataFrame(manufactor_encoded, columns=manufactor_feature_names, index=df_manufactor.index)
    return manufactor_encoded

def feature_region(df, len_df):
    if 'Area' not in df.columns:
        df['Area'] = dataset['Area'].mode()[0]
    df_region = df.copy()
    df_region = df_region.loc[:, ['Area']]
    min_freq = max(1, int(0.45 * len_df))  # Ensure min_frequency is at least 1
    df_region_encoder = OneHotEncoder(min_frequency=min_freq, sparse_output=False)
    region_encoded = df_region_encoder.fit_transform(df_region)
    region_feature_names = df_region_encoder.get_feature_names_out(df_region.columns)
    region_encoded = pd.DataFrame(region_encoded, columns=region_feature_names, index=df_region.index)
    return region_encoded

def feature_km(df, len_df):
    df_Km_binned = df.copy()
    df_Km_binned['Km_binned'] = pd.cut(df_Km_binned['Km'], bins=5)
    df_Km_binned = df_Km_binned.loc[:, ['Km_binned']]
    min_freq = max(1, int(len_df / 10))  # Ensure min_frequency is at least 1
    df_Km_binned_encoder = OneHotEncoder(min_frequency=min_freq, sparse_output=False)
    Km_binned_encoded = df_Km_binned_encoder.fit_transform(df_Km_binned)
    Km_binned_feature_names = df_Km_binned_encoder.get_feature_names_out(df_Km_binned.columns)
    Km_binned_encoded = pd.DataFrame(Km_binned_encoded, columns=Km_binned_feature_names, index=df_Km_binned.index)
    return Km_binned_encoded

def feature_rank(df):
    df_rank = df.copy()
    numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km']
    
    df_rank['capacity_Engine'] = df_rank['capacity_Engine'].astype(str).str.replace(',', '')   
    df_rank[numeric_features] = df_rank[numeric_features].apply(pd.to_numeric, errors='coerce')
    
    scaler = StandardScaler()
    df_rank[numeric_features] = scaler.fit_transform(df_rank[numeric_features])
    df_rank['Rank'] = df_rank[numeric_features].sum(axis=1)
    df_rank['Rank'] = 1 + 99 * (df_rank['Rank'] - df_rank['Rank'].min()) / (df_rank['Rank'].max() - df_rank['Rank'].min())
    df_rank['Rank'] = df_rank['Rank'].round(2)
    
    return df_rank


def prepare_data(df, len_df):
    df = df.drop_duplicates()
    df = low_data(df)
    print("Data after dropping low data columns:\n", df.head())
    df = Clearing_Outliers_By_IQR(df, 'Year', 30, 99)
    print("Data after clearing outliers in 'Year':\n", df.head())
    df = Clearing_Outliers_By_IQR(df, 'Km', 0, 75)
    print("Data after clearing outliers in 'Km':\n", df.head())
    df = Clearing_Outliers_By_IQR(df, 'Hand', 0, 64)
    print("Data after clearing outliers in 'Hand':\n", df.head())
    df = filling(df)
    print("Data after filling missing values:\n", df.head())
    df = feature_area(df)
    print("Data after adding area feature:\n", df.head())
    df = feature_bins(df)
    print("Data after binning 'Km':\n", df.head())
    gear_encoded = feature_gear(df, len_df)
    print("Gear encoded features:\n", gear_encoded.head())
    engine_type_encoded = feature_engine(df, len_df)
    print("Engine type encoded features:\n", engine_type_encoded.head())
    manufactor_encoded = feature_manufactor(df, len_df)
    print("Manufactor encoded features:\n", manufactor_encoded.head())
    region_encoded = feature_region(df, len_df)
    print("Region encoded features:\n", region_encoded.head())
    km_encoded = feature_km(df, len_df)
    print("Km binned encoded features:\n", km_encoded.head())
    df_rank = feature_rank(df)
    print("Data after adding rank feature:\n", df_rank.head())

    df = pd.concat([df, gear_encoded, engine_type_encoded, manufactor_encoded,
                    region_encoded, km_encoded, df_rank['Rank']], axis=1)

    df['Km_per_Year'] = df['Km'] / df['Year']
    print("Data after adding Km_per_Year:\n", df.head())
    return df

len_df1 = len(df)
df = prepare_data(df, len_df1)
df.info()







