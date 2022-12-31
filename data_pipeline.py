import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(filepath):
    """Loads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(filepath)


def load_tsv(filepath):
    """Loads a TSV file and returns a pandas DataFrame."""
    return pd.read_csv(filepath, sep='\t')


def load_excel(filepath):
    """Loads an Excel file and returns a pandas DataFrame."""
    return pd.read_excel(filepath)


def preprocess_text(df, text_column, preprocessor, **preprocessor_kwargs):
    """Applies a text preprocessor function to a specified text column in a DataFrame.
    Returns a new DataFrame with the preprocessed text column."""
    df = df.copy()
    df[text_column] = df[text_column].apply(
        preprocessor, **preprocessor_kwargs)
    return df


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Splits a DataFrame into training and testing sets.
    Returns a tuple of (X_train, X_test, y_train, y_test) arrays."""
    X = df.drop(columns=target_column)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
