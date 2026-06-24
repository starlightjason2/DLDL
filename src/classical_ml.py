"""Random forest disruption detection from smoothed-residual features."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model.dataset import IpDataset
from util.disruption_predict import (
    DEFAULT_SMOOTHING,
    DISRUPTION_FEATURE_NAMES,
    SmoothingConfig,
    extract_disruption_features,
)

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_REPO / ".env", encoding="utf-8")
_FEATURES_CACHE_VERSION = 4
_DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
}


def _repo_path(path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else _REPO / path


def _features_cache_path() -> Path:
    return _repo_path(os.environ["PROG_DIR"]) / "disruption_features.pkl"


def _features_cache_key(
    train_shot_indices: list[int],
    eval_shot_indices: list[int],
    smoothing: SmoothingConfig,
) -> dict:
    data_path = _repo_path(os.environ["DATA_PATH"])
    feature_module = Path(extract_disruption_features.__code__.co_filename)
    return {
        "version": _FEATURES_CACHE_VERSION,
        "normalization_type": os.environ["NORMALIZATION_TYPE"],
        "data_path": str(data_path),
        "data_mtime": data_path.stat().st_mtime,
        "feature_names": DISRUPTION_FEATURE_NAMES,
        "feature_extraction_mtime": feature_module.stat().st_mtime,
        "smoothing": smoothing.window_divisor,
        "train_shot_indices": train_shot_indices,
        "eval_shot_indices": eval_shot_indices,
    }


def _load_or_extract_features(
    dataset,
    train_shot_indices: list[int],
    eval_shot_indices: list[int],
    smoothing: SmoothingConfig,
):
    cache_path = _features_cache_path()
    cache_key = _features_cache_key(train_shot_indices, eval_shot_indices, smoothing)

    if cache_path.exists():
        with cache_path.open("rb") as cache_file:
            cached = pickle.load(cache_file)
        if cached["cache_key"] == cache_key:
            logger.info(f"Loaded cached features from {cache_path}")
            return (
                cached["train_features"],
                cached["train_is_disruptive"],
                cached["train_disruption_times"],
                cached["eval_features"],
                cached["eval_is_disruptive"],
                cached["eval_disruption_times"],
            )
        logger.info("Feature cache is stale; re-extracting")

    logger.info(f"Extracting train features ({len(train_shot_indices)} shots)...")
    train_features, train_is_disruptive, train_disruption_times = (
        _extract_features_for_indices(dataset, train_shot_indices, smoothing)
    )
    logger.info(f"Extracting eval features ({len(eval_shot_indices)} shots)...")
    eval_features, eval_is_disruptive, eval_disruption_times = (
        _extract_features_for_indices(dataset, eval_shot_indices, smoothing)
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as cache_file:
        pickle.dump(
            {
                "cache_key": cache_key,
                "train_features": train_features,
                "train_is_disruptive": train_is_disruptive,
                "train_disruption_times": train_disruption_times,
                "eval_features": eval_features,
                "eval_is_disruptive": eval_is_disruptive,
                "eval_disruption_times": eval_disruption_times,
            },
            cache_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(f"Saved extracted features to {cache_path}")
    return (
        train_features,
        train_is_disruptive,
        train_disruption_times,
        eval_features,
        eval_is_disruptive,
        eval_disruption_times,
    )


def _extract_features_for_indices(
    dataset, shot_indices: list[int], smoothing: SmoothingConfig
):
    feature_rows, is_disruptive, disruption_times = [], [], []
    for i, shot_index in enumerate(shot_indices):
        shot = dataset.load_shot_view(shot_index)
        feature_rows.append(extract_disruption_features(shot.current, smoothing))
        is_disruptive.append(int(shot.disruptive))
        disruption_times.append(float(shot.t_disrupt) if shot.disruptive else -1.0)
        if (i + 1) % 5000 == 0:
            logger.info(f"  Features extracted: {i + 1}/{len(shot_indices)}")
    return np.stack(feature_rows), np.array(is_disruptive), np.array(disruption_times)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    dataset = IpDataset(
        normalization_type=os.environ["NORMALIZATION_TYPE"],
        data_file=str(_repo_path(os.environ["DATA_PATH"])),
        labels_file=str(_repo_path(os.environ["TRAIN_LABELS_PATH"])),
        labels_path=str(_repo_path(os.environ["LABELS_PATH"])),
        data_dir=str(_repo_path(os.environ["DATA_DIR"])),
        labels_type="scaled",
        cpu_use=float(os.environ["CPU_USE"]),
        preprocessor_max_workers=int(os.environ["PREPROCESSOR_MAX_WORKERS"]),
    )

    train, dev, test = dataset.split()
    train_shot_indices = list(train.indices)
    eval_shot_indices = list(dev.indices) + list(test.indices)

    logger.info(f"Features: {', '.join(DISRUPTION_FEATURE_NAMES)}")
    logger.info(
        f"Using default smoothing (window_divisor={DEFAULT_SMOOTHING.window_divisor})"
    )

    (
        train_features,
        train_is_disruptive,
        train_disruption_times,
        eval_features,
        eval_is_disruptive,
        eval_disruption_times,
    ) = _load_or_extract_features(
        dataset, train_shot_indices, eval_shot_indices, DEFAULT_SMOOTHING
    )

    classifier = RandomForestClassifier(
        **_DEFAULT_RF_PARAMS,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(train_features, train_is_disruptive)
    predicted_is_disruptive = classifier.predict(eval_features)

    logger.info("=" * 60)
    logger.info("Classification (20% holdout)")
    logger.info(
        f"  Accuracy:  {accuracy_score(eval_is_disruptive, predicted_is_disruptive):.4f}"
    )
    logger.info(
        f"  Precision: {precision_score(eval_is_disruptive, predicted_is_disruptive, zero_division=0):.4f}"
    )
    logger.info(
        f"  Recall:    {recall_score(eval_is_disruptive, predicted_is_disruptive, zero_division=0):.4f}"
    )
    logger.info(
        f"  F1:        {f1_score(eval_is_disruptive, predicted_is_disruptive, zero_division=0):.4f}"
    )
    logger.info(
        f"  Confusion [tn fp; fn tp]:\n{confusion_matrix(eval_is_disruptive, predicted_is_disruptive)}"
    )

    eval_shot_indices_array = np.array(eval_shot_indices)
    false_positive_indices = eval_shot_indices_array[
        (eval_is_disruptive == 0) & (predicted_is_disruptive == 1)
    ].tolist()
    false_negative_indices = eval_shot_indices_array[
        (eval_is_disruptive == 1) & (predicted_is_disruptive == 0)
    ].tolist()
    progress_dir = _repo_path(os.environ["PROG_DIR"])
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "random_forest_false_positives.txt").write_text(
        "\n".join(map(str, false_positive_indices))
        + ("\n" if false_positive_indices else "")
    )
    (progress_dir / "random_forest_false_negatives.txt").write_text(
        "\n".join(map(str, false_negative_indices))
        + ("\n" if false_negative_indices else "")
    )
    logger.info(f"  FP ({len(false_positive_indices)}): {false_positive_indices}")
    logger.info(f"  FN ({len(false_negative_indices)}): {false_negative_indices}")

    ranked_feature_importances = sorted(
        zip(DISRUPTION_FEATURE_NAMES, classifier.feature_importances_),
        key=lambda row: row[1],
        reverse=True,
    )
    logger.info(
        "Feature importances: "
        + ", ".join(f"{name}={value:.3f}" for name, value in ranked_feature_importances)
    )

    disruptive_train_mask = train_is_disruptive == 1
    regressor = RandomForestRegressor(
        **_DEFAULT_RF_PARAMS,
        random_state=42,
        n_jobs=-1,
    )
    regressor.fit(
        train_features[disruptive_train_mask],
        train_disruption_times[disruptive_train_mask],
    )

    disruptive_eval_mask = eval_is_disruptive == 1
    predicted_disruption_times = regressor.predict(eval_features[disruptive_eval_mask])
    absolute_time_errors = np.abs(
        predicted_disruption_times - eval_disruption_times[disruptive_eval_mask]
    )
    logger.info("=" * 60)
    logger.info("Time regression (disruptive holdout)")
    logger.info(f"  MAE:     {absolute_time_errors.mean():.6f}")
    logger.info(f"  Median:  {np.median(absolute_time_errors):.6f}")
    logger.info(f"  Within 0.05: {100 * np.mean(absolute_time_errors <= 0.05):.1f}%")


if __name__ == "__main__":
    main()
