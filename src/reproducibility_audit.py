from pathlib import Path
import hashlib
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMNS,
    LEAKAGE_PRONE_COLUMNS,
)

from preprocessing import build_feature_frame


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def dataframe_hash(df):
    """
    Create a stable hash for checking row overlap.
    """

    normalized = (
        df.astype(str)
        .sort_index(axis=1)
        .sort_values(
            by=list(df.columns)
        )
        .reset_index(drop=True)
    )

    data_bytes = pd.util.hash_pandas_object(
        normalized,
        index=True
    ).values.tobytes()

    return hashlib.sha256(
        data_bytes
    ).hexdigest()


def row_hashes(df):
    """
    Generate row-level hashes.
    """

    normalized = (
        df.astype(str)
        .sort_index(axis=1)
    )

    return pd.util.hash_pandas_object(
        normalized,
        index=False
    )


def main():

    print("=" * 75)
    print("SMARTPREDICT-PdM")
    print("LEAKAGE + REPRODUCIBILITY AUDIT")
    print("=" * 75)

    # ========================================================
    # 1. Load original data
    # ========================================================

    section("1. DATASET AUDIT")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Rows         : {len(df)}")
    print(f"Columns      : {len(df.columns)}")

    print("\nColumns:")

    for col in df.columns:
        print(f"  - {col}")

    print(
        f"\nTarget column: {TARGET_COLUMN}"
    )

    print(
        f"Target distribution:\n"
        f"{df[TARGET_COLUMN].value_counts().to_string()}"
    )

    # ========================================================
    # 2. Missing values
    # ========================================================

    section("2. MISSING VALUE AUDIT")

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:
        print("PASS: No missing values detected.")
    else:
        print("WARNING: Missing values detected:")
        print(missing)

    # ========================================================
    # 3. Duplicate audit
    # ========================================================

    section("3. DUPLICATE AUDIT")

    duplicates = df.duplicated().sum()

    print(
        f"Duplicate rows: {duplicates}"
    )

    if duplicates == 0:
        print("PASS: No duplicate rows.")
    else:
        print(
            "WARNING: Duplicate rows exist."
        )

    # ========================================================
    # 4. Target leakage audit
    # ========================================================

    section("4. TARGET / FAILURE-MODE LEAKAGE AUDIT")

    print(
        "\nLeakage-prone columns defined by project:"
    )

    for col in LEAKAGE_PRONE_COLUMNS:
        exists = col in df.columns

        print(
            f"  {col}: "
            f"{'PRESENT' if exists else 'NOT PRESENT'}"
        )

    X = build_feature_frame(df)

    if TARGET_COLUMN in X.columns:
        X = X.drop(
            columns=[TARGET_COLUMN]
        )

    print(
        "\nFeatures passed to modeling:"
    )

    for col in X.columns:
        print(f"  - {col}")

    leakage_in_features = [
        col
        for col in LEAKAGE_PRONE_COLUMNS
        if col in X.columns
    ]

    target_in_features = (
        TARGET_COLUMN in X.columns
    )

    if (
        len(leakage_in_features) == 0
        and not target_in_features
    ):
        print(
            "\nPASS: Target and failure-mode "
            "columns are excluded from X."
        )
    else:
        print(
            "\nFAIL: Potential leakage detected:"
        )

        print(
            "Leakage columns:",
            leakage_in_features
        )

        print(
            "Target included:",
            target_in_features
        )

    # ========================================================
    # 5. Identifier leakage
    # ========================================================

    section("5. IDENTIFIER LEAKAGE AUDIT")

    identifier_in_features = [
        col
        for col in ID_COLUMNS
        if col in X.columns
    ]

    print(
        "Configured identifier columns:"
    )

    for col in ID_COLUMNS:
        print(f"  - {col}")

    print(
        "\nIdentifiers present in modeling features:"
    )

    if identifier_in_features:
        for col in identifier_in_features:
            print(f"  WARNING: {col}")
    else:
        print("  None")

    if not identifier_in_features:
        print(
            "\nPASS: Identifier columns excluded."
        )

    # ========================================================
    # 6. Recreate train/validation/test split
    # ========================================================

    section("6. TRAIN / VALIDATION / TEST SPLIT")

    y = df[TARGET_COLUMN]

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Test       : {len(X_test)}"
    )

    print(
        f"\nTraining failures   : {y_train.sum()}"
    )

    print(
        f"Validation failures : {y_val.sum()}"
    )

    print(
        f"Test failures       : {y_test.sum()}"
    )

    # ========================================================
    # 7. Split overlap check
    # ========================================================

    section("7. DATA SPLIT OVERLAP AUDIT")

    train_hash = set(
        row_hashes(X_train)
    )

    val_hash = set(
        row_hashes(X_val)
    )

    test_hash = set(
        row_hashes(X_test)
    )

    train_val_overlap = (
        train_hash & val_hash
    )

    train_test_overlap = (
        train_hash & test_hash
    )

    val_test_overlap = (
        val_hash & test_hash
    )

    print(
        f"Train ∩ Validation : "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test_overlap)}"
    )

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):
        print(
            "\nPASS: No row overlap detected."
        )
    else:
        print(
            "\nFAIL: Row overlap detected."
        )

    # ========================================================
    # 8. Feature-count audit
    # ========================================================

    section("8. FEATURE COUNT AUDIT")

    print(
        f"Raw dataset columns : {len(df.columns)}"
    )

    print(
        f"Model input columns : {len(X.columns)}"
    )

    print("\nModel input columns:")

    for i, col in enumerate(
        X.columns,
        start=1
    ):
        print(
            f"  {i}. {col}"
        )

    # ========================================================
    # 9. Saved model inspection
    # ========================================================

    section("9. SAVED MODEL AUDIT")

    model_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "xgboost_optimized.joblib"
    )

    print(
        f"Model path:\n{model_path}"
    )

    if not model_path.exists():

        print(
            "\nWARNING: Saved model not found."
        )

    else:

        model = joblib.load(
            model_path
        )

        print(
            f"\nLoaded object type:\n"
            f"{type(model)}"
        )

        print(
            "\nModel representation:"
        )

        print(model)

        # ----------------------------------------------------
        # Pipeline inspection
        # ----------------------------------------------------

        if hasattr(
            model,
            "named_steps"
        ):

            print(
                "\nPipeline components:"
            )

            for name, component in (
                model.named_steps.items()
            ):

                print(
                    f"  {name}: "
                    f"{type(component)}"
                )

            print(
                "\nPASS: Saved model contains "
                "pipeline components."
            )

        else:

            print(
                "\nNOTE: Saved model does not "
                "appear to be a sklearn Pipeline."
            )

        # ----------------------------------------------------
        # Feature metadata
        # ----------------------------------------------------

        if hasattr(
            model,
            "feature_names_in_"
        ):

            print(
                "\nModel feature_names_in_:"
            )

            for feature in (
                model.feature_names_in_
            ):
                print(
                    f"  - {feature}"
                )

        if hasattr(
            model,
            "n_features_in_"
        ):

            print(
                f"\nModel n_features_in_: "
                f"{model.n_features_in_}"
            )

    # ========================================================
    # 10. Dataset hash
    # ========================================================

    section("10. DATASET REPRODUCIBILITY HASH")

    dataset_hash = dataframe_hash(
        df
    )

    print(
        f"Dataset SHA-256:\n"
        f"{dataset_hash}"
    )

    # ========================================================
    # 11. Final summary
    # ========================================================

    section("11. AUDIT SUMMARY")

    print(
        "Dataset integrity      : CHECKED"
    )

    print(
        "Missing values         : CHECKED"
    )

    print(
        "Duplicate rows         : CHECKED"
    )

    print(
        "Target leakage         : CHECKED"
    )

    print(
        "Identifier leakage     : CHECKED"
    )

    print(
        "Train/validation/test  : CHECKED"
    )

    print(
        "Split overlap          : CHECKED"
    )

    print(
        "Feature representation : CHECKED"
    )

    print(
        "Saved model structure  : CHECKED"
    )

    print(
        "\nAudit completed."
    )


if __name__ == "__main__":
    main()