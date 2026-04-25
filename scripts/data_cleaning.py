import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def handle_missing(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Deck'] = df['Cabin'].str[0]
    df.drop(columns=['Cabin'], inplace=True)
    return df

def handle_outliers(df):
    fare_cap = df['Fare'].quantile(0.99)
    df['Fare'] = df['Fare'].clip(upper=fare_cap)
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df = handle_missing(df)
    df = handle_outliers(df)
    return df

if __name__ == "__main__":
    df = load_data('data/train.csv')
    df = clean_data(df)
    df.to_csv('data/train_cleaned.csv', index=False)
    print("Cleaning done. Saved to data/train_cleaned.csv")