import pandas as pd
from sdmetrics.single_table import DCRBaselineProtection, DCROverfittingProtection
from .epsilon_estimate import get_eps_audit
from tqdm import tqdm

PRIVACY_METRICS = {}


def privacy_metric(func):
    PRIVACY_METRICS[func.__name__] = func
    return func


@privacy_metric
def dcr_baseline(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    metadata: dict,
    **kwargs,
):
    out = DCRBaselineProtection.compute_breakdown(real_df, syn_df, metadata)
    return {
        "score": out["score"],
        "syn_median_dcr": out["median_DCR_to_real_data"]["synthetic_data"],
        "rand_median_dcr": out["median_DCR_to_real_data"]["random_data_baseline"],
    }


@privacy_metric
def dcr_overfit(
    real_df: pd.DataFrame,
    val_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    metadata: dict,
    **kwargs,
):
    out = {"score": 0, "syn_closer_to_val": 0, "syn_closer_to_train": 0}
    slice_span = -(-real_df.shape[0]//val_df.shape[0])
    for fold in tqdm(range(slice_span)):
        res = DCROverfittingProtection.compute_breakdown(
            real_training_data=real_df[fold::slice_span],
            synthetic_data=syn_df[fold::slice_span],
            real_validation_data=val_df,
            metadata=metadata,
        )
        out["score"] += res["score"]
        out["syn_closer_to_val"] += res["synthetic_data_percentages"]["closer_to_holdout"]
        out["syn_closer_to_train"] += res["synthetic_data_percentages"]["closer_to_training"]
    return {k: v/slice_span for k, v in out.items()}

@privacy_metric
def dcr_val_to_real_vs_synth(
    real_df: pd.DataFrame,
    val_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    metadata: dict,
    **kwargs,
):
    # These are poorly labeled, but this applies the DCR algorithm to the pairs
    # real * validation and synthetic * validation
    out = DCROverfittingProtection.compute_breakdown(
        real_training_data=real_df,
        real_validation_data=syn_df,
        synthetic_data=val_df,
        metadata=metadata,
    )
    train_count = real_df.shape[0]
    val_count = val_df.shape[0]

    # would be better to have this reported directly...
    closer_to_synth_count = out["synthetic_data_percentages"]["closer_to_holdout"] * val_count
    closer_to_train_count = out["synthetic_data_percentages"]["closer_to_training"] * val_count

    # Because we're only getting the aggregate data out of compute_breakdown, we can't adjust r
    eps_est = get_eps_audit(
            val_count, val_count,
            max(int(closer_to_synth_count+0.5), int(closer_to_train_count+0.5)),
            0.5/train_count,
            1/3.0)
    return {
        "score": out["score"],
        "val_count": val_count,
        "val_closer_to_synth": out["synthetic_data_percentages"]["closer_to_holdout"],
        "val_closer_to_synth_count": closer_to_synth_count,
        "val_closer_to_train": out["synthetic_data_percentages"]["closer_to_training"],
        "val_closer_to_train_count": closer_to_train_count,
        "epsilon_estimate": eps_est,
    }

from xgboost import XGBClassifier
import numpy as np
import sklearn.metrics

@privacy_metric
def o1_estimate(
    real_df: pd.DataFrame,
    val_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    canary_df: pd.DataFrame = None,
    **kwargs,
):
    """
    Post-hoc "One (1) Training Run"-style audit
    """

    conditions = []
    if canary_df is not None:
        conditions.append(("syn-canary-test", syn_df, canary_df, test_df))
        conditions.append(("real-canary-test", real_df, canary_df, test_df))
    conditions.append(("real-real-test", real_df, real_df, test_df))
    conditions.append(("syn-real-test", syn_df, real_df, test_df))
    conditions.append(("syn-syn-test", syn_df, syn_df, test_df))
    conditions.append(("real-syn-test", real_df, syn_df, test_df))

    out = {}
    for (cond, train_df, ref_df, ext_df) in conditions:
        canary_count = min(20000, ref_df.shape[0], ext_df.shape[0])
        clf = XGBClassifier()
        # FIXME: last column is target due to caller impl
        clf.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

        ref_samples = np.random.binomial(canary_count, 0.5)
        ext_samples = canary_count - ref_samples

        indicator = np.zeros(canary_count, dtype=int)
        indicator[:ref_samples] = 1

        samples = pd.concat([ref_df.sample(ref_samples), ext_df.sample(ext_samples)])
        y_pred = clf.predict_proba(samples.iloc[:, :-1])
        losses = -np.log(y_pred[np.arange(canary_count), samples.iloc[:, -1]])

        sorted_ixs = losses.argsort()
        for guessrate in 10, 20, 100:
            guesses = max(1, canary_count * guessrate//200)
            lo_ref = indicator[sorted_ixs[:guesses]].sum()
            hi_ext = guesses - indicator[sorted_ixs[-guesses:]].sum()
            out[f"{cond}-{guessrate}/canaries"] = canary_count
            out[f"{cond}-{guessrate}/guesses"] = guesses*2
            out[f"{cond}-{guessrate}/correct"] = float(lo_ref + hi_ext) / (guesses*2)
            out[f"{cond}-{guessrate}/lo_ref"] = int(lo_ref)
            out[f"{cond}-{guessrate}/hi_ext"] = int(hi_ext)
            out[f"{cond}-{guessrate}/eps_est"] = get_eps_audit(
                    canary_count, guesses*2, lo_ref + hi_ext,
                    0.5/real_df.shape[0], 0.5)
    return out

def compute_metric(
    metric: str,
    **kwargs,
) -> dict:
    if metric not in PRIVACY_METRICS:
        raise ValueError(
            f"Unknown privacy metric {metric}. Available: {list(PRIVACY_METRICS.keys())}"
        )
    func = PRIVACY_METRICS[metric]
    out = func(**kwargs)
    return out
