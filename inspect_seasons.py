
import pandas as pd

files = [
    "epl_2425.csv",
    "epl_2526.csv",
    "epl_2627.csv"
]

for file in files:
    print("\n" + "=" * 50)
    print("FILE:", file)

    try:
        df = pd.read_csv(file)

        print("Rows and columns:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print("Error:", e)
