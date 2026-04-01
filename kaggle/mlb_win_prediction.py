from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TRAIN_PATH = Path('data.csv')
PREDICT_PATH = Path('predict.csv')
OUTPUT_PATH = Path('mlb_win_predictions.csv')
TARGET = 'W'
EXCLUDED_FEATURES = {'ID'}  # identifier only; not informative for prediction


def load_data(train_path: Path, predict_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical training data and the scoring dataset."""
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    return train_df, predict_df


def get_feature_columns(train_df: pd.DataFrame, predict_df: pd.DataFrame) -> list[str]:
    """Keep only columns that exist in both files and are safe for prediction."""
    shared_columns = sorted(set(train_df.columns).intersection(predict_df.columns))
    feature_columns = [col for col in shared_columns if col not in EXCLUDED_FEATURES]
    return feature_columns


def build_model() -> Pipeline:
    """Create a simple, stable regression pipeline.

    Why Ridge?
    - Baseball statistics are often correlated (for example: runs, hits, and at-bats).
    - Ridge regression handles that multicollinearity well by shrinking unstable coefficients.
    - It is interpretable, fast, and performed strongly in grouped cross-validation.
    """
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler(with_mean=False)),
            ('model', Ridge(alpha=3.0)),
        ]
    )


def evaluate_with_grouped_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """Evaluate the model using GroupKFold so each season stays in one fold.

    Grouping by year avoids mixing teams from the same MLB season across train and
    validation folds, which gives a cleaner estimate of real-world performance.
    """
    cv = GroupKFold(n_splits=n_splits)

    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    r2_scores: list[float] = []

    for train_idx, valid_idx in cv.split(X, y, groups=groups):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)

        mae_scores.append(mean_absolute_error(y_valid, preds))
        rmse_scores.append(root_mean_squared_error(y_valid, preds))
        r2_scores.append(r2_score(y_valid, preds))

    return {
        'cv_mae': sum(mae_scores) / len(mae_scores),
        'cv_rmse': sum(rmse_scores) / len(rmse_scores),
        'cv_r2': sum(r2_scores) / len(r2_scores),
    }


def fit_and_predict(
    model: Pipeline,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Train on all historical data and predict wins for the new teams/seasons."""
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    X_score = predict_df[feature_columns]

    model.fit(X_train, y_train)
    predicted_wins = model.predict(X_score)

    results = pd.DataFrame()
    if 'ID' in predict_df.columns:
        results['ID'] = predict_df['ID']
    results['predicted_wins'] = predicted_wins.round(1)
    return results


def main() -> None:
    train_df, predict_df = load_data(TRAIN_PATH, PREDICT_PATH)
    feature_columns = get_feature_columns(train_df, predict_df)

    if TARGET not in train_df.columns:
        raise ValueError(f"Training file must contain the target column: {TARGET}")
    if 'yearID' not in train_df.columns:
        raise ValueError("Training file must contain 'yearID' for grouped validation.")

    model = build_model()

    metrics = evaluate_with_grouped_cv(
        model=model,
        X=train_df[feature_columns],
        y=train_df[TARGET],
        groups=train_df['yearID'],
        n_splits=5,
    )

    predictions = fit_and_predict(
        model=model,
        train_df=train_df,
        predict_df=predict_df,
        feature_columns=feature_columns,
    )
    predictions.to_csv(OUTPUT_PATH, index=False)

    print('Model evaluation (5-fold GroupKFold by season)')
    print(f"MAE :  {metrics['cv_mae']:.3f}")
    print(f"RMSE:  {metrics['cv_rmse']:.3f}")
    print(f"R^2 :  {metrics['cv_r2']:.3f}")
    print(f'\nSaved predictions to: {OUTPUT_PATH.resolve()}')
    print('\nPreview:')
    print(predictions.head())


if __name__ == '__main__':
    main()
