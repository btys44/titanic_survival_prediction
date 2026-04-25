import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def get_feature_importance(df):
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Survived']
    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['Survived']

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False)

def drop_correlated(df, threshold=0.9):
    X = df.select_dtypes(include=[np.number])
    corr = X.corr().abs()
    upper = corr.where(pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Dropping correlated features: {to_drop}")
    return df.drop(columns=to_drop)

if __name__ == "__main__":
    df = pd.read_csv('data/train_engineered.csv')
    df = drop_correlated(df)

    importances = get_feature_importance(df)
    print("\nTop features:\n", importances.head(15))

    # Keep features above importance threshold
    selected = importances[importances > 0.01].index.tolist()
    print("\nSelected features:", selected)

    importances.head(15).plot(kind='barh', figsize=(8, 6))
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')