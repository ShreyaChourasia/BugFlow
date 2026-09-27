"""US-34/US-36: SHAP top-3 factors turned into plain sentences. Uses SHAP's
exact TreeExplainer (no sampling), so the same input always produces the
same explanation — required determinism, not just a nice property."""

from typing import Any

import numpy as np

from bugflow_ml.models.commit_risk import FEATURE_COLUMNS


def build_explainer(raw_model: Any) -> Any:
    import shap

    return shap.TreeExplainer(raw_model)


def _sentence_for(feature: str, value: float, shap_value: float) -> str:
    rising = shap_value > 0

    if feature == "churn":
        return (
            f"This change touches ~{int(value)} lines in total; larger changes are riskier."
            if rising
            else f"This change only touches ~{int(value)} lines, which is usually safer."
        )
    if feature == "lines_added":
        return (
            f"This change adds ~{int(value)} lines; large additions are riskier."
            if rising
            else f"This change adds only ~{int(value)} lines, which is usually safer."
        )
    if feature == "lines_deleted":
        return (
            f"This change removes ~{int(value)} lines; large deletions are riskier."
            if rising
            else f"This change removes only ~{int(value)} lines, which is usually safer."
        )
    if feature == "files_changed":
        return (
            f"This change touches {int(value)} files; spreading a change across "
            "many files is riskier."
            if rising
            else f"This change stays within {int(value)} file(s), which is usually safer."
        )
    if feature == "directories_touched":
        unit = "directory" if value == 1 else "directories"
        return (
            f"This change spans {int(value)} different directories; wide-reaching "
            "changes are riskier."
            if rising
            else f"This change stays within {int(value)} {unit}, which is usually safer."
        )
    if feature == "subsystems_touched":
        return (
            f"This change crosses {int(value)} subsystems; cross-cutting changes are riskier."
            if rising
            else f"This change stays within {int(value)} subsystem(s), which is usually safer."
        )
    if feature == "is_fix":
        return (
            "This is itself a bug-fix commit; fixes sometimes introduce new regressions."
            if rising
            else "This is not a bug-fix commit."
        )

    # entropy and author_prior_commits: the size/diffusion features above are
    # reliably monotonic (more files/lines touched = riskier) in JIT defect
    # literature, but these two can interact non-monotonically in a tree
    # model, so state the SHAP direction plainly rather than assume a story.
    direction = "increases" if rising else "decreases"
    if feature == "entropy":
        return (
            f"This change's edits are spread across files with an entropy of "
            f"{value:.2f}, which {direction} the predicted risk."
        )
    if feature == "author_prior_commits":
        return (
            f"The author has made {int(value)} prior commit(s) to this project, "
            f"which {direction} the predicted risk."
        )
    return f"{feature} = {value}, which {direction} the predicted risk."


def explain_prediction(
    explainer: Any, feature_vector: dict[str, float], top_n: int = 3
) -> dict[str, Any]:
    X = np.array([[float(feature_vector.get(col, 0.0)) for col in FEATURE_COLUMNS]])
    shap_values = np.asarray(explainer.shap_values(X))
    if shap_values.ndim == 3:
        # Some shap/LightGBM version combinations return one array per class
        # (n_classes, n_samples, n_features); take the positive class.
        shap_values = shap_values[-1]
    row = shap_values[0]

    order = np.argsort(-np.abs(row))[:top_n]
    factors = []
    sentences = []
    for idx in order:
        feature = FEATURE_COLUMNS[idx]
        value = feature_vector.get(feature, 0.0)
        shap_value = float(row[idx])
        factors.append(
            {
                "feature": feature,
                "value": value,
                "shap_value": shap_value,
                "impact": "increases" if shap_value > 0 else "decreases",
            }
        )
        sentences.append(_sentence_for(feature, value, shap_value))

    text = " ".join(sentences) if sentences else "No dominant risk factors identified."
    return {"text": text, "factors": factors}
