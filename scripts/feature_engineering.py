import pandas as pd
import numpy as np

def add_family_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

def extract_title(df):
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    rare = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    return df

def add_age_groups(df):
    df['AgeGroup'] = pd.cut(df['Age'],
        bins=[0, 12, 17, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Senior'])
    return df

def add_fare_per_person(df):
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    return df

def encode_features(df):
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup'])
    return df

def log_transform(df):
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Age_log'] = np.log1p(df['Age'])
    return df

def engineer_features(df):
    df = add_family_features(df)
    df = extract_title(df)
    df = add_age_groups(df)
    df = add_fare_per_person(df)
    df = log_transform(df)
    df = encode_features(df)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/train_cleaned.csv')
    df = engineer_features(df)
    df.to_csv('data/train_engineered.csv', index=False)
    print("Feature engineering done. Saved to data/train_engineered.csv")