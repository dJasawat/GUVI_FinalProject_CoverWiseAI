"""Train and use the synthetic CoverWise coverage-risk classification model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "Dataset" / "coverwise_synthetic_coverage_risk_dataset_500.csv"
DEFAULT_MODEL = PROJECT_ROOT / "model" / "coverage_risk_model.joblib"

FEATURE_COLUMNS = (
    "policy_type",
    "sum_insured",
    "annual_premium",
    "premium_to_coverage_ratio",
    "waiting_period_months",
    "pre_existing_disease_waiting_period_months",
    "copayment_percentage",
    "room_rent_limit_present",
    "icu_limit_present",
    "disease_sub_limit_count",
    "exclusion_count",
    "restoration_benefit_present",
    "no_claim_bonus_present",
    "child_coverage_present",
    "maternity_coverage_present",
    "claim_process_clarity_score",
    "ambiguous_clause_count",
    "risk_flags_count",
    "coverage_clarity_score",
)
TARGET_COLUMN = "risk_label"
RISK_LABELS = (
    "Strong Coverage",
    "Moderate Coverage",
    "Caution Required",
    "High Risk",
)
NUMERIC_FEATURES = tuple(column for column in FEATURE_COLUMNS if column != "policy_type")
CATEGORICAL_FEATURES = ("policy_type",)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _risk_index(row: dict[str, Any]) -> float:
    """Apply transparent synthetic rules to produce the dataset's teaching labels."""
    return min(
        100.0,
        row["waiting_period_months"] * 0.28
        + row["pre_existing_disease_waiting_period_months"] * 0.22
        + row["copayment_percentage"] * 0.55
        + row["room_rent_limit_present"] * 8
        + row["icu_limit_present"] * 5
        + row["disease_sub_limit_count"] * 2
        + row["exclusion_count"] * 1.3
        + (not row["restoration_benefit_present"]) * 4
        + (not row["no_claim_bonus_present"]) * 1
        + row["ambiguous_clause_count"] * 3
        + (100 - row["claim_process_clarity_score"]) * 0.13,
    )


def _clarity_score(row: dict[str, Any]) -> float:
    clarity = (
        100
        - row["ambiguous_clause_count"] * 8
        - row["disease_sub_limit_count"] * 2
        - row["exclusion_count"] * 1.5
        - row["waiting_period_months"] * 0.15
        - row["pre_existing_disease_waiting_period_months"] * 0.12
        - row["copayment_percentage"] * 0.25
        - row["room_rent_limit_present"] * 5
        - row["icu_limit_present"] * 3
        + row["claim_process_clarity_score"] * 0.15
    )
    return round(_clip(clarity, 0, 100), 1)


def derive_document_model_features(
    extracted_features: dict[str, Any], exclusion_count: int | None
) -> dict[str, Any]:
    """Derive ratio and synthetic teaching scores without filling unknown policy terms."""
    features = dict(extracted_features)
    features["exclusion_count"] = exclusion_count
    sum_insured = features.get("sum_insured")
    annual_premium = features.get("annual_premium")
    features["premium_to_coverage_ratio"] = (
        round(annual_premium / sum_insured, 5)
        if annual_premium is not None and sum_insured is not None and sum_insured > 0
        else None
    )

    risk_inputs = (
        "waiting_period_months",
        "pre_existing_disease_waiting_period_months",
        "copayment_percentage",
        "room_rent_limit_present",
        "icu_limit_present",
        "disease_sub_limit_count",
        "exclusion_count",
        "ambiguous_clause_count",
    )
    features["risk_flags_count"] = (
        sum(
            (
                features["waiting_period_months"] >= 24,
                features["pre_existing_disease_waiting_period_months"] >= 36,
                features["copayment_percentage"] >= 20,
                features["room_rent_limit_present"],
                features["icu_limit_present"],
                features["disease_sub_limit_count"] >= 2,
                features["exclusion_count"] >= 5,
                features["ambiguous_clause_count"] >= 3,
            )
        )
        if all(features.get(name) is not None for name in risk_inputs)
        else None
    )

    clarity_inputs = (
        "ambiguous_clause_count",
        "disease_sub_limit_count",
        "exclusion_count",
        "waiting_period_months",
        "pre_existing_disease_waiting_period_months",
        "copayment_percentage",
        "room_rent_limit_present",
        "icu_limit_present",
        "claim_process_clarity_score",
    )
    features["coverage_clarity_score"] = (
        _clarity_score(features)
        if all(features.get(name) is not None for name in clarity_inputs)
        else None
    )
    return {name: features.get(name) for name in FEATURE_COLUMNS}


def _label_for_risk(risk_index: float) -> str:
    if risk_index < 25:
        return "Strong Coverage"
    if risk_index < 50:
        return "Moderate Coverage"
    if risk_index < 75:
        return "Caution Required"
    return "High Risk"


def _sample_policy(rng: np.random.Generator) -> dict[str, Any]:
    policy_type = str(rng.choice(("Individual", "Family Floater", "Senior Citizen")))
    sum_insured = int(rng.choice((300_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000)))
    ratio = round(float(rng.uniform(0.006, 0.045)), 5)
    row: dict[str, Any] = {
        "policy_type": policy_type,
        "sum_insured": sum_insured,
        "annual_premium": int(round(sum_insured * ratio)),
        "premium_to_coverage_ratio": ratio,
        "waiting_period_months": int(rng.choice((0, 1, 2, 3, 6, 12, 18, 24, 36, 48))),
        "pre_existing_disease_waiting_period_months": int(
            rng.choice((0, 12, 24, 36, 48, 60, 72))
        ),
        "copayment_percentage": int(rng.choice((0, 5, 10, 15, 20, 25, 30, 40))),
        "room_rent_limit_present": bool(rng.random() < 0.32),
        "icu_limit_present": bool(rng.random() < 0.16),
        "disease_sub_limit_count": int(min(rng.poisson(1.2), 7)),
        "exclusion_count": int(rng.integers(0, 11)),
        "restoration_benefit_present": bool(rng.random() < 0.68),
        "no_claim_bonus_present": bool(rng.random() < 0.72),
        "child_coverage_present": bool(rng.random() < 0.48),
        "maternity_coverage_present": bool(rng.random() < 0.34),
        "claim_process_clarity_score": round(float(rng.uniform(35, 100)), 1),
        "ambiguous_clause_count": int(min(rng.poisson(1.6), 9)),
    }
    row["risk_flags_count"] = sum(
        (
            row["waiting_period_months"] >= 24,
            row["pre_existing_disease_waiting_period_months"] >= 36,
            row["copayment_percentage"] >= 20,
            row["room_rent_limit_present"],
            row["icu_limit_present"],
            row["disease_sub_limit_count"] >= 2,
            row["exclusion_count"] >= 5,
            row["ambiguous_clause_count"] >= 3,
        )
    )
    row["coverage_clarity_score"] = _clarity_score(row)
    return row


def generate_synthetic_dataset(
    output_path: str | Path = DEFAULT_DATASET, rows_per_class: int = 125, seed: int = 42
) -> pd.DataFrame:
    """Generate balanced, reproducible synthetic policy features and rule-based labels."""
    if rows_per_class < 1:
        raise ValueError("rows_per_class must be a positive integer.")
    rng = np.random.default_rng(seed)
    counts = {label: 0 for label in RISK_LABELS}
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = rows_per_class * 20_000

    while min(counts.values()) < rows_per_class and attempts < max_attempts:
        attempts += 1
        row = _sample_policy(rng)
        label = _label_for_risk(_risk_index(row))
        if counts[label] >= rows_per_class:
            continue
        row[TARGET_COLUMN] = label
        rows.append(row)
        counts[label] += 1

    if min(counts.values()) < rows_per_class:
        raise RuntimeError(
            f"Could not generate a balanced dataset after {attempts} attempts: {counts}"
        )

    frame = pd.DataFrame(rows, columns=(*FEATURE_COLUMNS, TARGET_COLUMN))
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline((("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()))),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    (
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    )
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def _make_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    dataset_path: str | Path = DEFAULT_DATASET,
    model_path: str | Path = DEFAULT_MODEL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare classifiers, validate on a holdout, and save the best full-data pipeline."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    data = pd.read_csv(dataset_path)
    required_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing values in model features or target.")
    unexpected_labels = set(data[TARGET_COLUMN].unique()).difference(RISK_LABELS)
    if unexpected_labels:
        raise ValueError(f"Dataset contains unsupported risk labels: {sorted(unexpected_labels)}")

    x = data.loc[:, FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )

    comparisons: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, estimator in _make_estimators().items():
        pipeline = Pipeline(
            (("preprocessor", _make_preprocessor()), ("classifier", estimator))
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        predictions[name] = predicted
        comparisons[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
        }

    best_name = max(comparisons, key=lambda name: comparisons[name]["macro_f1"])
    final_model = Pipeline(
        (
            ("preprocessor", _make_preprocessor()),
            ("classifier", _make_estimators()[best_name]),
        )
    )
    final_model.fit(x, y)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, destination)

    report = {
        "dataset_rows": int(len(data)),
        "test_rows": int(len(y_test)),
        "class_counts": {str(label): int(count) for label, count in y.value_counts().items()},
        "model_comparison": comparisons,
        "best_validation_model": best_name,
        "best_model_classification_report": classification_report(
            y_test,
            predictions[best_name],
            labels=list(RISK_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "saved_model": str(destination),
        "disclaimer": (
            "Synthetic educational model only. Predictions are not insurance advice, "
            "actuarial estimates, or a substitute for reviewing policy wording."
        ),
    }
    return report


def predict_coverage_risk(model: Any, features: dict[str, Any]) -> dict[str, Any]:
    """Predict one policy's synthetic risk class with calibrated output shape and caveat."""
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError(f"Missing required model features: {', '.join(missing)}")
    missing_values = [
        column for column in FEATURE_COLUMNS
        if features[column] is None or pd.isna(features[column])
    ]
    if len(missing_values) == len(FEATURE_COLUMNS):
        raise ValueError("Cannot score this policy: no model features were extracted.")
    sample = pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])
    prediction = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    classes = model.named_steps["classifier"].classes_
    return {
        "risk_label": prediction,
        "class_probabilities": {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities)
        },
        "coverage_clarity_score": (
            float(features["coverage_clarity_score"])
            if features["coverage_clarity_score"] is not None
            else None
        ),
        "missing_features": missing_values,
        "provisional": bool(missing_values),
        "disclaimer": (
            "Synthetic educational model only. Missing values, when present, are imputed "
            "from training medians or an Unknown category. Verify policy terms; this is "
            "not a recommendation or insurance advice."
        ),
    }


def load_model(model_path: str | Path = DEFAULT_MODEL) -> Any:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.coverage_risk_model train` first."
        )
    return joblib.load(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="generate balanced synthetic CSV data")
    generate_parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    generate_parser.add_argument("--rows-per-class", type=int, default=125)
    generate_parser.add_argument("--seed", type=int, default=42)
    train_parser = commands.add_parser(
        "train", help="compare classifiers and train the best validation model"
    )
    train_parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    if args.command == "generate":
        frame = generate_synthetic_dataset(args.output, args.rows_per_class, args.seed)
        print(f"Wrote {len(frame)} rows to {args.output}")
        print(frame[TARGET_COLUMN].value_counts().reindex(RISK_LABELS).to_string())
        return 0

    print(json.dumps(train_and_evaluate(args.dataset, args.model_output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
