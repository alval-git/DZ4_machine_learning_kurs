import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler



def load_data(file_path, sep):
    try:
        return pd.read_csv(file_path, sep=sep, low_memory=False)
    except FileNotFoundError:
        print("file was not found")



def preprocess_data(df, drop_columns, target_feature,  numeric_features, categorical_features):
    new_df = df.copy()
    
    if drop_columns:
        try:
            new_df = new_df.drop(columns=drop_columns, axis=1)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Exception type: {type(e)}")

    try:
        new_df = new_df.dropna()
        print(f"пропущенные значения: {df.isnull().sum()}")
        #кодирую target
        # new_df[target_feature] = new_df[target_feature].map(target_encode)

        # масштабирование числовых признаков
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(new_df[numeric_features])
        new_df[numeric_features] = scaled_values

        # Кодирование категориальных признаков 
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col].astype(str))
            label_encoders[col] = le

        # признаки и таргет
        X = new_df.drop(target_feature, axis=1)
        y = new_df[target_feature]

        print(f"   Данные подготовлены для ML:")
        print(f"     Признаки (X): {X.shape}")
        print(f"     Целевая переменная (y): {y.shape}")
        print(f"     Баланс классов: 0={sum(y==0)}, 1={sum(y==1)}")

        return X, y
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Exception type: {type(e)}")

