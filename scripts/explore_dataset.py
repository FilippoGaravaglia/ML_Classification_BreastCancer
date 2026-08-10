from pathlib import Path

from src.data.loader import load_raw_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "wdbc.data"
)


def main() -> None:
    df = load_raw_dataset(DATASET_PATH)

    print("=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== NUMBER OF COLUMNS ===")
    print(len(df.columns))

    print("\n=== TARGET DISTRIBUTION ===")
    print(df["diagnosis"].value_counts())

    print("\n=== TARGET DISTRIBUTION (%) ===")
    print(
        df["diagnosis"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().sum())

    print("\n=== DUPLICATED ROWS ===")
    print(df.duplicated().sum())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)


if __name__ == "__main__":
    main()