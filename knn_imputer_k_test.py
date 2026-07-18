"""
KNN GPA imputation workflow (trainer checklist).
- k=3 vs k=5 masking RMSE on train (single seed + multi-seed robustness)
- Final KNNImputer(k=5) fit on full train only; transform train + test
- GPA_missing = 1 if GPA was missing before imputation
- Optional: downstream model on synthetic test_score (replace with your target)

Replace `build_synthetic_df()` with pd.read_csv(...) for real data.
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


features = ["hours_studied", "attendance_rate", "num_extracurriculars"]
target = "previous_GPA"
downstream_target = "test_score"  # set None to skip downstream demo
CHOSEN_K = 5
ROBUSTNESS_SEEDS = (42, 123, 7, 2024, 99)


def build_synthetic_df(n=500, rng=None):
    """Correlated fake data + a downstream target for demo."""
    if rng is None:
        rng = np.random.default_rng(0)
    hours = rng.uniform(5, 40, n)
    attendance = np.clip(0.5 + 0.01 * hours + rng.normal(0, 0.15, n), 0, 1)
    extracurriculars = rng.poisson(2, n)
    gpa = (
        0.05 * hours
        + 2.0 * attendance
        + 0.08 * extracurriculars
        + rng.normal(0, 0.25, n)
    )
    gpa = np.clip(gpa, 0, 4)
    # ~15% missing GPA (MCAR-ish) for imputation demo
    miss = rng.random(n) < 0.15
    gpa_obs = gpa.copy()
    gpa_obs[miss] = np.nan
    test_score = (
        55
        + 8.0 * np.nan_to_num(gpa_obs, nan=np.nanmean(gpa_obs))
        + 0.4 * hours
        + 12.0 * attendance
        + rng.normal(0, 4, n)
    )
    return pd.DataFrame(
        {
            "hours_studied": hours,
            "attendance_rate": attendance,
            "num_extracurriculars": extracurriculars,
            "previous_GPA": gpa_obs,
            "test_score": test_score,
        }
    )


def masking_rmse_for_seed(df, seed, k_list=(3, 5), mask_frac=0.10):
    """Mask known train GPAs; return RMSE per k (train only, no test)."""
    np.random.seed(seed)
    train, _ = train_test_split(df, test_size=0.2, random_state=seed)
    train_known = train[train[target].notnull()].copy()
    if len(train_known) < 20:
        return {k: np.nan for k in k_list}

    mask_idx = train_known.sample(frac=mask_frac, random_state=seed).index
    true_vals = train_known.loc[mask_idx, target].copy()
    vk = train_known.copy()
    vk.loc[mask_idx, target] = np.nan

    X_train = vk[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    out = {}
    for k in k_list:
        imputer = KNNImputer(n_neighbors=k)
        stacked = np.hstack([X_scaled, vk[[target]].values])
        imputed = imputer.fit_transform(stacked)
        imputed_gpa = imputed[:, -1]
        pos = vk.index.get_indexer(mask_idx)
        pred = imputed_gpa[pos]
        out[k] = root_mean_squared_error(true_vals.values, pred)
    return out


def robustness_report(df, seeds=ROBUSTNESS_SEEDS, k_list=(3, 5)):
    wins = {k: 0 for k in k_list}
    rows = []
    for s in seeds:
        r = masking_rmse_for_seed(df, s, k_list=k_list)
        rows.append((s, r[3], r[5]))
        best = min(k_list, key=lambda k: r[k])
        # tie-break: prefer smaller k
        tied = [k for k in k_list if r[k] == r[best]]
        wins[min(tied)] += 1
    return rows, wins


def impute_gpa_knn_full_train(
    train_df,
    test_df,
    features_cols,
    target_col,
    k=5,
):
    """
    Fit StandardScaler + KNNImputer on train only.
    Impute missing target_col in train and test. Add GPA_missing from *original* NaNs.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_gpa_missing = train_df[target_col].isna().astype(int)
    test_gpa_missing = test_df[target_col].isna().astype(int)

    X_tr = train_df[features_cols].fillna(0)
    X_te = test_df[features_cols].fillna(0)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    stack_tr = np.hstack([X_tr_s, train_df[[target_col]].values.astype(float)])
    stack_te = np.hstack([X_te_s, test_df[[target_col]].values.astype(float)])

    imputer = KNNImputer(n_neighbors=k)
    imputer.fit(stack_tr)
    imp_tr = imputer.transform(stack_tr)
    imp_te = imputer.transform(stack_te)

    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out[target_col] = imp_tr[:, -1]
    test_out[target_col] = imp_te[:, -1]
    train_out["GPA_missing"] = train_gpa_missing.values
    test_out["GPA_missing"] = test_gpa_missing.values
    return train_out, test_out, scaler, imputer


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    df = build_synthetic_df(500)

    # --- Initial k=3 vs k=5 (seed 42) ---
    print("=== Masking validation (seed=42) ===")
    r0 = masking_rmse_for_seed(df, 42)
    for k in (3, 5):
        print(f"  k={k} RMSE={r0[k]:.4f}")
    print(f"  Best k (RMSE): {min(r0, key=r0.get)}")

    # --- Robustness: multiple seeds ---
    print("\n=== Robustness (masking RMSE, several seeds) ===")
    table, wins = robustness_report(df)
    print(f"{'seed':>6} {'RMSE_k3':>10} {'RMSE_k5':>10} {'pick':>6}")
    for s, a, b in table:
        pick = 3 if a <= b else 5
        if abs(a - b) < 1e-6:
            pick = 3
        print(f"{s:>6} {a:>10.4f} {b:>10.4f} {pick:>6}")
    print(f"  Wins (tie -> smaller k): k=3 -> {wins[3]}, k=5 -> {wins[5]}")

    # --- Final imputation: k=5, train fit only ---
    print(f"\n=== Final imputation (KNN k={CHOSEN_K}, fit on train only) ===")
    train_raw, test_raw = train_test_split(df, test_size=0.2, random_state=seed)
    drop_demo = [downstream_target] if downstream_target and downstream_target in train_raw.columns else []
    train_x = train_raw.drop(columns=drop_demo, errors="ignore")
    test_x = test_raw.drop(columns=drop_demo, errors="ignore")

    train_imp, test_imp, _, _ = impute_gpa_knn_full_train(
        train_x, test_x, features, target, k=CHOSEN_K
    )
    print(f"  Train rows: {len(train_imp)}, Test rows: {len(test_imp)}")
    print(f"  Train GPA_missing rate: {train_imp['GPA_missing'].mean():.2%}")

    # --- Downstream: predict test_score (demo) ---
    if downstream_target and downstream_target in train_raw.columns:
        y_tr = train_raw[downstream_target].values
        y_te = test_raw[downstream_target].values
        model_features = features + [target, "GPA_missing"]
        X_m_tr = train_imp[model_features].fillna(0)
        X_m_te = test_imp[model_features].fillna(0)
        model = Ridge(alpha=1.0)
        model.fit(X_m_tr, y_tr)
        pred_te = model.predict(X_m_te)
        rmse_all = root_mean_squared_error(y_te, pred_te)
        orig_miss = test_raw[target].isna()
        if orig_miss.any() and (~orig_miss).any():
            rmse_miss = root_mean_squared_error(y_te[orig_miss], pred_te[orig_miss])
            rmse_obs = root_mean_squared_error(y_te[~orig_miss], pred_te[~orig_miss])
        elif orig_miss.any():
            rmse_miss = root_mean_squared_error(y_te[orig_miss], pred_te[orig_miss])
            rmse_obs = float("nan")
        else:
            rmse_miss = float("nan")
            rmse_obs = root_mean_squared_error(y_te, pred_te)
        print("\n=== Downstream (Ridge on imputed train) - hold-out test ===")
        print(f"  RMSE all test rows:     {rmse_all:.4f}")
        print(f"  RMSE orig GPA missing:  {rmse_miss:.4f}" if not np.isnan(rmse_miss) else "  RMSE orig GPA missing:  n/a")
        print(f"  RMSE orig GPA present:  {rmse_obs:.4f}" if not np.isnan(rmse_obs) else "  RMSE orig GPA present:  n/a")

    print("\nDone. Document: chosen k, masking RMSEs, robustness table, and test metrics.")
