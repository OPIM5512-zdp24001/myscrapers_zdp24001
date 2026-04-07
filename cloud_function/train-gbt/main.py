# Gradient Boosted Trees: train on all data < today (local TZ); hold out today
# Uses LLM-extracted features: transmission, fuel_type, body_style, color
# Hyperparameter tuning via Optuna (Bayesian optimization)
# Outputs: predictions, permutation importance, PDP plots
# HTTP entrypoint: train_gbt_http

import os, io, json, logging, traceback, base64
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score, KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ---- ENV ----
PROJECT_ID     = os.getenv("PROJECT_ID", "")
GCS_BUCKET     = os.getenv("GCS_BUCKET", "")
DATA_KEY       = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
OUTPUT_PREFIX  = os.getenv("OUTPUT_PREFIX", "preds_gbt")
TIMEZONE       = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def _write_csv_to_gcs(client: storage.Client, bucket: str, key: str, df: pd.DataFrame):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")


def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str,
                         data: bytes, content_type: str = "image/png"):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(data, content_type=content_type)


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def run_once(dry_run: bool = False, n_trials: int = 50):
    client = storage.Client(project=PROJECT_ID)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "make", "model", "year", "mileage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # --- Parse timestamps and choose local-day split ---
    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    # --- Clean numerics ---
    orig_rows = len(df)
    df["price_num"]   = _clean_numeric(df["price"])
    df["year_num"]    = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df["mileage"])

    # --- Feature engineering ---
    CURRENT_YEAR = pd.Timestamp.utcnow().year
    df["vehicle_age"] = CURRENT_YEAR - df["year_num"]

    # Normalize make
    if "make" in df.columns:
        df["make"] = df["make"].astype(str).str.lower().str.strip()
        df["make"] = df["make"].replace({"chevy": "chevrolet", "vw": "volkswagen"})

    # Clean LLM categorical fields
    for col in ["transmission", "fuel_type", "body_style", "color"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({"nan": "unknown", "": "unknown", "none": "unknown"})
        else:
            df[col] = "unknown"

    # Normalize color shades
    color_map = {"grey": "gray", "ecotronic gray": "gray",
                 "baltic gray": "gray", "gray magnetic metallic": "gray"}
    df["color"] = df["color"].map(lambda x: color_map.get(x, x))

    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | with valid numeric price=%d", orig_rows, valid_price_rows)

    # Remove mileage outliers
    df = df[df["mileage_num"] < 500_000].copy()

    counts = df["date_local"].value_counts().sort_index()
    logging.info("Recent date counts (local): %s",
                 json.dumps({str(k): int(v) for k, v in counts.tail(8).items()}))

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {"status": "noop", "reason": "need at least two distinct dates",
                "dates": [str(d) for d in unique_dates]}

    today_local = unique_dates[-1]
    train_df   = df[df["date_local"] <  today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()
    train_df = train_df[train_df["price_num"].notna()]

    logging.info("Train rows: %d | Holdout rows today (%s): %d",
                 len(train_df), today_local, len(holdout_df))

    if len(train_df) < 20:
        return {"status": "noop", "reason": "too few training rows",
                "train_rows": int(len(train_df))}

    # --- Feature columns ---
    target = "price_num"
    cat_cols = ["make", "model", "transmission", "fuel_type", "body_style", "color"]
    num_cols = ["vehicle_age", "mileage_num"]
    feats = cat_cols + num_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    # --- Hyperparameter tuning with Optuna ---
    X_train = train_df[feats]
    y_train = train_df[target]

    n_cv = min(5, len(X_train))
    cv = KFold(n_splits=n_cv, shuffle=True, random_state=42)

    best_params = {"n_estimators": 200, "max_depth": 3,
                   "learning_rate": 0.05, "subsample": 0.8,
                   "min_samples_split": 5, "min_samples_leaf": 3}

    if HAS_OPTUNA and len(X_train) >= 30:
        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
                "max_depth":         trial.suggest_int("max_depth", 2, 8),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            }
            pipe_t = Pipeline([
                ("pre", pre),
                ("model", GradientBoostingRegressor(random_state=42, **params))
            ])
            scores = cross_val_score(pipe_t, X_train, y_train, cv=cv,
                                     scoring="neg_mean_absolute_error")
            return -scores.mean()

        study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        logging.info("Optuna best MAE (CV): $%.0f | params: %s",
                     study.best_value, best_params)
    else:
        logging.info("Skipping Optuna (not installed or too few rows). Using defaults.")

    # --- Train final model ---
    pipe = Pipeline([
        ("pre", pre),
        ("model", GradientBoostingRegressor(random_state=42, **best_params))
    ])
    pipe.fit(X_train, y_train)

    # --- Output path ---
    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
    run_folder = f"{OUTPUT_PREFIX}/{now_utc.strftime('%Y%m%d%H')}"

    # ---- Predictions on holdout ----
    mae_today = None
    preds_df = pd.DataFrame()
    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = pipe.predict(X_h)
        cols = ["post_id", "scraped_at", "make", "model", "year", "mileage", "price",
                "transmission", "fuel_type", "body_style", "color"]
        cols = [c for c in cols if c in holdout_df.columns]
        preds_df = holdout_df[cols].copy()
        preds_df["actual_price"] = holdout_df["price_num"]
        preds_df["pred_price"]   = np.round(y_hat, 2)
        preds_df["residual"]     = preds_df["pred_price"] - preds_df["actual_price"]

        if holdout_df["price_num"].notna().any():
            mask = holdout_df["price_num"].notna()
            if mask.any():
                mae_today = float(mean_absolute_error(
                    holdout_df.loc[mask, "price_num"], y_hat[mask.values]))

    # ---- Permutation importance ----
    X_train_transformed = pipe.named_steps["pre"].transform(X_train)
    perm_result = permutation_importance(
        pipe.named_steps["model"], X_train_transformed, y_train,
        n_repeats=20, random_state=42, scoring="neg_mean_absolute_error"
    )

    # Map feature names from the transformer
    num_feat_names = num_cols
    cat_feat_names = list(pipe.named_steps["pre"]
                          .named_transformers_["cat"]
                          .named_steps["oh"]
                          .get_feature_names_out(cat_cols))
    all_feat_names = num_feat_names + cat_feat_names

    imp_df = pd.DataFrame({
        "feature": all_feat_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std":  perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Also create a grouped-by-original-column importance
    grouped_imp = {}
    for col in num_cols:
        idx = all_feat_names.index(col)
        grouped_imp[col] = perm_result.importances_mean[idx]
    for col in cat_cols:
        mask = [f.startswith(col + "_") for f in all_feat_names]
        grouped_imp[col] = np.mean(perm_result.importances_mean[np.array(mask)])
    grouped_imp_df = pd.DataFrame([
        {"feature": k, "importance_mean": v} for k, v in grouped_imp.items()
    ]).sort_values("importance_mean", ascending=False)

    # ---- Permutation importance chart ----
    fig_imp, ax = plt.subplots(figsize=(8, 5))
    gi = grouped_imp_df.sort_values("importance_mean", ascending=True)
    ax.barh(gi["feature"], gi["importance_mean"], color="steelblue", edgecolor="k")
    ax.set_xlabel("Mean Permutation Importance")
    ax.set_title("Permutation Importance — All Features (GBT)")
    plt.tight_layout()
    imp_png = _fig_to_bytes(fig_imp)

    # ---- PDP for top 3 numeric-position features ----
    top3 = grouped_imp_df.head(3)["feature"].tolist()
    top3_positions = []
    for feat in top3:
        if feat in num_cols:
            top3_positions.append(all_feat_names.index(feat))
        else:
            matches = [i for i, f in enumerate(all_feat_names) if f.startswith(feat + "_")]
            if matches:
                top3_positions.append(matches[0])

    pdp_png = None
    if top3_positions:
        fig_pdp, axes = plt.subplots(1, len(top3_positions), figsize=(6 * len(top3_positions), 5))
        if len(top3_positions) == 1:
            axes = [axes]
        for ax_i, (pos, label) in enumerate(zip(top3_positions, top3)):
            PartialDependenceDisplay.from_estimator(
                pipe.named_steps["model"], X_train_transformed,
                features=[pos], ax=axes[ax_i]
            )
            axes[ax_i].set_title(f"PDP: {label}")
        plt.suptitle("Partial Dependence Plots — Top 3 Features", fontsize=14, y=1.02)
        plt.tight_layout()
        pdp_png = _fig_to_bytes(fig_pdp)

    # ---- Write outputs to GCS ----
    if not dry_run:
        if len(preds_df) > 0:
            _write_csv_to_gcs(client, GCS_BUCKET,
                              f"{run_folder}/preds.csv", preds_df)
            logging.info("Wrote predictions: %d rows", len(preds_df))

        _write_csv_to_gcs(client, GCS_BUCKET,
                          f"{run_folder}/permutation_importance.csv", grouped_imp_df)
        _write_bytes_to_gcs(client, GCS_BUCKET,
                            f"{run_folder}/permutation_importance.png", imp_png)
        logging.info("Wrote permutation importance")

        if pdp_png:
            _write_bytes_to_gcs(client, GCS_BUCKET,
                                f"{run_folder}/pdp_top3.png", pdp_png)
            logging.info("Wrote PDP plot")

        # Save model params
        params_json = json.dumps({
            "best_params": best_params,
            "train_rows": int(len(train_df)),
            "mae_today": mae_today,
            "top3_features": top3,
            "optuna_used": HAS_OPTUNA,
        })
        b = client.bucket(GCS_BUCKET)
        b.blob(f"{run_folder}/model_params.json").upload_from_string(
            params_json, content_type="application/json")
        logging.info("Wrote model_params.json")
    else:
        logging.info("Dry run — skipping GCS writes")

    return {
        "status": "ok",
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "valid_price_rows": valid_price_rows,
        "mae_today": mae_today,
        "best_params": best_params,
        "top3_features": top3,
        "output_folder": run_folder,
        "dry_run": dry_run,
        "timezone": TIMEZONE,
        "optuna_used": HAS_OPTUNA,
    }


def train_gbt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(
            dry_run=bool(body.get("dry_run", False)),
            n_trials=int(body.get("n_trials", 50)),
        )
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500,
                {"Content-Type": "application/json"})
