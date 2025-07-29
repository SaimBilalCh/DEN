import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Values:")
    print(df.duplicated().sum())
    return df

def preprocess_data(df):
    df.drop_duplicates(inplace=True)

    categorical_features = ["Gender"]
    numerical_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    transformed_data = pipeline.fit_transform(df)

    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)

    transformed_df = pd.DataFrame(transformed_data, columns=all_feature_names)

    print("\nProcessed Data Head:")
    print(transformed_df.head())
    print("\nProcessed Data Info:")
    print(transformed_df.info())
    return transformed_df

if __name__ == "__main__":
    file_path = "Mall_Customers.csv"
    df = load_and_explore_data(file_path)
    processed_df = preprocess_data(df.drop(columns=['CustomerID']))
