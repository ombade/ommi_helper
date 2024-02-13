
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def info(df):
    """
    This is the info function , which provides basic information about a dataframe such as number of rows and columns, non-null valuess
    """
    # 1. Data Types
    print(df.sample(2))
    print("1. Data Types:")
    # print(df.dtypes)

    # # 2. Countplot for Categorical Columns
    print("\n2. Countplot for Categorical Columns:")
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=df)
        plt.title(f'{col} - Countplot')
        plt.show()

    # # 3. Histogram for Numerical Columns
    print("\n3. Histogram for Numerical Columns:")
    numerical_cols = df.select_dtypes(exclude='object').columns
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=20, kde=True)
        plt.title(f'{col} - Histogram')
        plt.show()

def data_info(df):
    # 1. How big is the data?
    print("1. Data Shape:")
    print(df.shape)

    # 2. How does the data look like?
    print("\n2. Sample Data:")
    print(df.sample())

    # 3. What is the data type of cols?
    print("\n3. Data Types:")
    print(df.info())

    # 4. Are there any missing values in the dataset? If yes, how many and what
    print("\n4. Missing Values:")
    print(df.isnull().sum())

    # 5. How does the data look mathematically?
    print("\n5. Data Description:")
    print(df.describe())

    # 6. Are there duplicate values?
    print("\n6. Duplicate Values:")
    print(df.duplicated().sum())



