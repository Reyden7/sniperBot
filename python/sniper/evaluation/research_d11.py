"""Phase D.11 diagnostic ranking and regime audit on frozen D.10 predictions."""

import hashlib
import json
import math
import tracemalloc
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import polars as pl
from scipy.stats import kendalltau, spearmanr, wasserstein_distance  # type: ignore[import-untyped]

from sniper.data.time import as_utc
from sniper.domain.research_d11 import D11Conclusion, D11Performance, ResearchD11Report
from sniper.evaluation.research_d10 import (
    SOURCE_TICKS,
    _fit_side,
    _nested_folds,
    _primary_configuration,
    _validate_research_scope,
    d10_temporal_folds,
)
from sniper.evaluation.research_v3 import V3_FEATURES, _prefix

RANDOM_SEED_D11 = 20260911
BOOTSTRAP_REPLICATIONS_D11 = 2000
TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001)
EV_THRESHOLDS = (0, 1, 2, 3, 5, 10)
EXPECTED_D10_PROTOCOL_SHA256 = "20d8967079e4503b581e8729aa140efe8bbdf7713e29e86c230ef4256b1cfcf5"
EXPECTED_D10_FREEZE_SHA256 = "1a20efb21268db540c6c6f3547f1b76d0dff4c1950ef2f4570ecd468caca4699"
EXPECTED_D10_ENGINE_SHA256 = "93630689811933c21773ca50b4e0236b7407dcd207f6e358a0daafe3a16969dd"
EXPECTED_D10_V3_DEPENDENCY_SHA256 = (
    "9bf8dbb528167cb723df6943477743c941e378e7ab418b539cd12b9fcc6d1271"
)
EXPECTED_D10_REPORT_SHA256 = "31836898ece3cf439682579055f0b3b9871ef73ee2b98e0b3445daccee05957d"
REGIME_NUMERIC = (
    "spread_points",
    "m1_atr_points",
    "m5_atr_points",
    "m15_atr_points",
    "tick_rate_1s",
    "tick_rate_5s",
    "tick_rate_15s",
    "m5_momentum_points",
    "m1_momentum_1_points",
    "predicted_EV",
    "P_TARGET",
    "P_STOP",
    "P_NEITHER",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_d10_hashes(data_root: Path, repository_root: Path) -> dict[str, Any]:
    protocol = repository_root / "docs" / "research-protocol-d10.yaml"
    engine = repository_root / "python" / "sniper" / "evaluation" / "research_d10.py"
    v3_dependency = repository_root / "python" / "sniper" / "evaluation" / "research_v3.py"
    report = data_root / "reports" / "research-d10-1781049600000-1788825600000.json"
    manifest_path = (
        data_root / "reports" / "research-d10-1781049600000-1788825600000-freeze-manifest.json"
    )
    required = (protocol, engine, v3_dependency, report, manifest_path)
    if any(not path.exists() for path in required):
        raise ValueError("D.11 requires the complete frozen D.10 artifact set")
    actual = {
        "d10_protocol_sha256": _sha256(protocol),
        "d10_engine_sha256": _sha256(engine),
        "d10_v3_dependency_sha256": _sha256(v3_dependency),
        "d10_report_sha256": _sha256(report),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual["d10_freeze_manifest_sha256"] = manifest.get("freeze_manifest_sha256")
    expected = {
        "d10_protocol_sha256": EXPECTED_D10_PROTOCOL_SHA256,
        "d10_engine_sha256": EXPECTED_D10_ENGINE_SHA256,
        "d10_v3_dependency_sha256": EXPECTED_D10_V3_DEPENDENCY_SHA256,
        "d10_report_sha256": EXPECTED_D10_REPORT_SHA256,
        "d10_freeze_manifest_sha256": EXPECTED_D10_FREEZE_SHA256,
    }
    checks = {name: actual[name] == value for name, value in expected.items()}
    if not all(checks.values()):
        raise RuntimeError("D.10 hash verification failed; D.11 reproduction refused")
    d10 = json.loads(report.read_text(encoding="utf-8"))
    if d10.get("conclusion") != "D10_INSUFFICIENT_EVIDENCE":
        raise RuntimeError("D.10 official verdict differs from the frozen verdict")
    return {"actual": actual, "expected": expected, "checks": checks, "all_passed": True}


def _prediction_frame(
    frame: pl.DataFrame,
    validation: np.ndarray,
    side: str,
    result: dict[str, Any],
    fold: int,
    role: str,
    candidate: np.ndarray,
) -> pl.DataFrame:
    configuration = _primary_configuration()
    prefix = _prefix(configuration, side)
    scoped = frame[validation]
    probability = cast(np.ndarray, result["primary_probability"])
    timestamps = scoped["timestamp_utc"]
    return pl.DataFrame(
        {
            "timestamp_utc": timestamps,
            "day": timestamps.dt.strftime("%Y-%m-%d"),
            "fold": np.full(len(validation), fold),
            "role": np.full(len(validation), role),
            "week": scoped["week"],
            "session": scoped["session"],
            "side": np.full(len(validation), side),
            "P_TARGET": probability[:, 0],
            "P_STOP": probability[:, 1],
            "P_NEITHER": probability[:, 2],
            "predicted_EV": result["predicted_ev"],
            "safety_buffer": np.full(len(validation), result["safety_buffer"]),
            "conservative_EV": result["conservative_ev"],
            "actual_gross_pnl": scoped[f"{prefix}_gross_return_points"],
            "actual_executable_pnl": scoped[f"{prefix}_executable_return_points"],
            "actual_BASE_net_pnl": scoped[f"{prefix}_base_net_points"],
            "actual_STRESS_net_pnl": scoped[f"{prefix}_stress_net_points"],
            "actual_outcome": scoped[f"{prefix}_outcome"],
            "MFE": scoped[f"{prefix}_mfe_points"],
            "MAE": scoped[f"{prefix}_mae_points"],
            "is_D10_candidate": candidate,
            "spread_points": scoped["spread_points"],
            "m1_atr_points": scoped["m1_atr_points"],
            "m5_atr_points": scoped["m5_atr_points"],
            "m15_atr_points": scoped["m15_atr_points"],
            "tick_rate_1s": scoped["tick_rate_1s"],
            "tick_rate_5s": scoped["tick_rate_5s"],
            "tick_rate_15s": scoped["tick_rate_15s"],
            "m5_momentum_points": scoped["m5_momentum_points"],
            "m1_momentum_1_points": scoped["m1_momentum_1_points"],
        }
    )


def reproduce_d10_oof(
    frame: pl.DataFrame, start: datetime, end: datetime
) -> tuple[pl.DataFrame, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Re-fit the exact frozen D.10 models and retain only external OOF predictions."""
    x_all = frame.select(V3_FEATURES).to_pandas()
    folds, outer_audit = d10_temporal_folds(frame, start, end)
    rows: list[pl.DataFrame] = []
    safety: list[dict[str, Any]] = []
    nested_audit: list[dict[str, Any]] = []
    for fold in folds:
        fold_number = cast(int, fold["fold"])
        role = cast(str, fold["role"])
        train = cast(np.ndarray, fold["train_indices"])
        validation = cast(np.ndarray, fold["validation_indices"])
        nested, nested_rows = _nested_folds(
            frame,
            train,
            start,
            cast(datetime, fold["validation_start_utc"]),
            fold_number,
        )
        nested_audit.extend(nested_rows)
        long = _fit_side(frame, x_all, train, validation, nested, "LONG")
        short = _fit_side(frame, x_all, train, validation, nested, "SHORT")
        long_ev = cast(np.ndarray, long["conservative_ev"])
        short_ev = cast(np.ndarray, short["conservative_ev"])
        long_eligible, short_eligible = long_ev > 0, short_ev > 0
        choose_long = long_eligible & (~short_eligible | (long_ev >= short_ev))
        choose_short = short_eligible & ~choose_long
        rows.append(
            _prediction_frame(frame, validation, "LONG", long, fold_number, role, choose_long)
        )
        rows.append(
            _prediction_frame(frame, validation, "SHORT", short, fold_number, role, choose_short)
        )
        for side, result in (("LONG", long), ("SHORT", short)):
            item = dict(cast(dict[str, Any], result["buffer_report"]))
            item.update(
                {
                    "fold": fold_number,
                    "role": role,
                    "side": side,
                    "raw_unclamped_penalty_points": (
                        -item["mean_daily_residual_points"]
                        + 1.645 * item["daily_residual_standard_error_points"]
                    ),
                    "daily_residual_standard_deviation_points": (
                        item["daily_residual_standard_error_points"] * math.sqrt(item["utc_days"])
                    ),
                    "zero_buffer_explanation": (
                        "Raw penalty <= 0, therefore frozen max(0, raw penalty) returns 0."
                        if item["safety_buffer_points"] == 0
                        else None
                    ),
                }
            )
            safety.append(item)
    return (
        pl.concat(rows, how="vertical").sort(["timestamp_utc", "side"]),
        tuple(safety),
        (outer_audit + tuple(nested_audit)),
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)
    names = ("min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max")
    return {
        name: float(value)
        for name, value in zip(names, np.quantile(values, probabilities), strict=True)
    }


def _ev_distribution(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    rows = []
    for fold in sorted(frame["fold"].unique().to_list()):
        for side in ("LONG", "SHORT"):
            scoped = frame.filter((pl.col("fold") == fold) & (pl.col("side") == side))
            if scoped.is_empty():
                continue
            conservative = scoped["conservative_EV"].to_numpy().astype(float)
            rows.append(
                {
                    "fold": fold,
                    "role": str(scoped["role"][0]),
                    "side": side,
                    "observations": scoped.height,
                    "predicted_EV": _quantiles(scoped["predicted_EV"].to_numpy().astype(float)),
                    "safety_buffer": _quantiles(scoped["safety_buffer"].to_numpy().astype(float)),
                    "conservative_EV": _quantiles(conservative),
                    "density_counts": {
                        f"conservative_EV_gt_{threshold}": int(np.sum(conservative > threshold))
                        for threshold in EV_THRESHOLDS
                    },
                }
            )
    return tuple(rows)


def _correlation(kind: str, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = spearmanr(x, y).statistic if kind == "spearman" else kendalltau(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _daily_correlation_ci(
    frame: pl.DataFrame, x_column: str, kind: str, seed_offset: int
) -> dict[str, Any]:
    daily = (
        frame.group_by("day", maintain_order=True)
        .agg(pl.col(x_column).mean(), pl.col("actual_BASE_net_pnl").mean())
        .sort("day")
    )
    x = daily[x_column].to_numpy().astype(float)
    y = daily["actual_BASE_net_pnl"].to_numpy().astype(float)
    rng = np.random.default_rng(RANDOM_SEED_D11 + seed_offset)
    values = []
    for _ in range(BOOTSTRAP_REPLICATIONS_D11):
        indices = rng.integers(0, len(daily), size=len(daily))
        value = _correlation(kind, x[indices], y[indices])
        if value is not None:
            values.append(value)
    return {
        "estimand": f"{kind.upper()}_OF_UTC_DAY_MEANS",
        "point_estimate": _correlation(kind, x, y),
        "lower": float(np.quantile(values, 0.025)) if values else None,
        "upper": float(np.quantile(values, 0.975)) if values else None,
        "replications": len(values),
        "block": "UTC_DAY_MEANS",
        "days": len(daily),
        "seed": RANDOM_SEED_D11 + seed_offset,
    }


def _scopes(frame: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    scopes: list[tuple[str, str, pl.DataFrame]] = []
    for fold in sorted(frame["fold"].unique().to_list()):
        folded = frame.filter(pl.col("fold") == fold)
        for side in ("LONG", "SHORT"):
            scoped = folded.filter(pl.col("side") == side)
            if not scoped.is_empty():
                scopes.append((f"FOLD_{fold}", side, scoped))
        scopes.append((f"FOLD_{fold}", "COMBINED", folded))
    for side in ("LONG", "SHORT"):
        scoped = frame.filter(pl.col("side") == side)
        if not scoped.is_empty():
            scopes.append(("ALL_VALIDATION", side, scoped))
    scopes.append(("ALL_VALIDATION", "COMBINED", frame))
    return scopes


def _ranking_correlations(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    rows = []
    for offset, (scope, side, scoped) in enumerate(_scopes(frame), start=1):
        predicted = scoped["predicted_EV"].to_numpy().astype(float)
        conservative = scoped["conservative_EV"].to_numpy().astype(float)
        actual = scoped["actual_BASE_net_pnl"].to_numpy().astype(float)
        rows.append(
            {
                "scope": scope,
                "side": side,
                "observations": scoped.height,
                "spearman_predicted_EV": _correlation("spearman", predicted, actual),
                "spearman_predicted_EV_daily_block_ci95": _daily_correlation_ci(
                    scoped, "predicted_EV", "spearman", offset * 10
                ),
                "kendall_predicted_EV": _correlation("kendall", predicted, actual),
                "kendall_predicted_EV_daily_block_ci95": _daily_correlation_ci(
                    scoped, "predicted_EV", "kendall", offset * 10 + 1
                ),
                "spearman_conservative_EV": _correlation("spearman", conservative, actual),
                "spearman_conservative_EV_daily_block_ci95": _daily_correlation_ci(
                    scoped, "conservative_EV", "spearman", offset * 10 + 2
                ),
            }
        )
    return tuple(rows)


def _subset_metrics(frame: pl.DataFrame, unconditional: float) -> dict[str, Any]:
    base = frame["actual_BASE_net_pnl"].to_numpy().astype(float)
    positive = float(np.sum(base[base > 0]))
    negative = float(-np.sum(base[base < 0]))
    sessions = {
        str(value): int(frame.filter(pl.col("session") == value).height)
        for value in sorted(frame["session"].unique().to_list())
    }
    return {
        "N": frame.height,
        "executable_expectancy": float(str(frame["actual_executable_pnl"].mean())),
        "BASE_expectancy": float(np.mean(base)),
        "STRESS_expectancy": float(str(frame["actual_STRESS_net_pnl"].mean())),
        "uplift_vs_unconditional_BASE": float(np.mean(base) - unconditional),
        "win_rate": float(np.mean(base > 0)),
        "profit_factor_BASE": positive / negative if negative > 0 else None,
        "outcome_proportions": {
            outcome: float(np.mean(frame["actual_outcome"].to_numpy() == outcome))
            for outcome in ("TARGET_FIRST", "STOP_FIRST", "NEITHER")
        },
        "average_MFE": float(str(frame["MFE"].mean())),
        "average_MAE": float(str(frame["MAE"].mean())),
        "active_days": frame["day"].n_unique(),
        "active_weeks": frame["week"].n_unique(),
        "sessions": sessions,
        "largest_session_share": max(sessions.values()) / frame.height,
    }


def _ranking_curves(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    rows = []
    for scope, side, scoped in _scopes(frame):
        ordered = scoped.sort(
            ["predicted_EV", "timestamp_utc", "side"],
            descending=[True, False, False],
        )
        unconditional = float(str(scoped["actual_BASE_net_pnl"].mean()))
        for fraction in TOP_FRACTIONS:
            count = max(1, math.ceil(fraction * ordered.height))
            rows.append(
                {
                    "scope": scope,
                    "side": side,
                    "top_fraction": fraction,
                    "unconditional_BASE_expectancy": unconditional,
                    **_subset_metrics(ordered.head(count), unconditional),
                }
            )
    return tuple(rows)


def _economic_calibration(frame: pl.DataFrame) -> tuple[dict[str, Any], ...]:
    rows = []
    for scope, side, scoped in _scopes(frame):
        ordered = scoped.sort("predicted_EV")
        predicted = ordered["predicted_EV"].to_numpy().astype(float)
        actual = ordered["actual_BASE_net_pnl"].to_numpy().astype(float)
        bins = []
        for group_number, indices in enumerate(np.array_split(np.arange(len(ordered)), 10), 1):
            mean_predicted = float(np.mean(predicted[indices]))
            mean_actual = float(np.mean(actual[indices]))
            bins.append(
                {
                    "group": group_number,
                    "N": len(indices),
                    "mean_predicted_EV": mean_predicted,
                    "mean_realized_BASE_pnl": mean_actual,
                    "difference_actual_minus_predicted": mean_actual - mean_predicted,
                }
            )
        rows.append(
            {
                "scope": scope,
                "side": side,
                "observations": scoped.height,
                "mean_actual_minus_predicted": float(np.mean(actual - predicted)),
                "MAE_EV": float(np.mean(np.abs(actual - predicted))),
                "RMSE_EV": float(np.mean((actual - predicted) ** 2) ** 0.5),
                "groups": bins,
            }
        )
    return tuple(rows)


def _numeric_shift(reference: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    reference = reference[np.isfinite(reference)]
    target = target[np.isfinite(target)]
    pooled = float(((np.var(reference) + np.var(target)) / 2) ** 0.5)
    smd = (float(np.mean(target)) - float(np.mean(reference))) / pooled if pooled else 0.0
    quantile_edges = np.unique(np.quantile(reference, np.linspace(0.1, 0.9, 9)))
    edges = np.concatenate(([-np.inf], quantile_edges, [np.inf]))
    reference_hist = np.histogram(reference, bins=edges)[0].astype(float)
    target_hist = np.histogram(target, bins=edges)[0].astype(float)
    reference_share = np.maximum(reference_hist / len(reference), 1e-6)
    target_share = np.maximum(target_hist / len(target), 1e-6)
    psi = float(np.sum((target_share - reference_share) * np.log(target_share / reference_share)))
    wasserstein = float(wasserstein_distance(reference, target))
    return {
        "reference_mean": float(np.mean(reference)),
        "target_mean": float(np.mean(target)),
        "standardized_mean_difference": smd,
        "PSI": psi,
        "wasserstein": wasserstein,
        "normalized_wasserstein": wasserstein / pooled if pooled else 0.0,
    }


def _categorical_shift(reference: pl.Series, target: pl.Series) -> dict[str, Any]:
    categories = sorted(set(reference.to_list()) | set(target.to_list()))
    reference_share = np.array([float((reference == value).mean()) for value in categories])
    target_share = np.array([float((target == value).mean()) for value in categories])
    ref_safe, target_safe = np.maximum(reference_share, 1e-6), np.maximum(target_share, 1e-6)
    return {
        "reference_frequencies": dict(zip(map(str, categories), reference_share, strict=True)),
        "target_frequencies": dict(zip(map(str, categories), target_share, strict=True)),
        "PSI": float(np.sum((target_safe - ref_safe) * np.log(target_safe / ref_safe))),
    }


def _regime_shift(frame: pl.DataFrame) -> dict[str, Any]:
    unique_market = frame.filter(pl.col("side") == "LONG")
    comparisons: list[tuple[str, pl.DataFrame, pl.DataFrame]] = []
    for left, right in ((1, 2), (2, 3), (3, 4)):
        comparisons.append(
            (
                f"FOLD_{left}_VS_FOLD_{right}",
                frame.filter(pl.col("fold") == left),
                frame.filter(pl.col("fold") == right),
            )
        )
    comparisons.append(
        (
            "MODEL_DEVELOPMENT_VS_INTERNAL_FREEZE_CHECK",
            frame.filter(pl.col("role") == "MODEL_DEVELOPMENT"),
            frame.filter(pl.col("role") == "INTERNAL_FREEZE_CHECK"),
        )
    )
    results = []
    for name, reference, target in comparisons:
        results.append(
            {
                "comparison": name,
                "reference_side_observations": reference.height,
                "target_side_observations": target.height,
                "numeric": {
                    column: _numeric_shift(
                        reference[column].to_numpy().astype(float),
                        target[column].to_numpy().astype(float),
                    )
                    for column in REGIME_NUMERIC
                },
                "categorical": {
                    column: _categorical_shift(reference[column], target[column])
                    for column in ("session", "actual_outcome")
                },
            }
        )
    fold_summaries = []
    for fold in (1, 2, 3, 4):
        side_rows = frame.filter(pl.col("fold") == fold)
        market = unique_market.filter(pl.col("fold") == fold)
        fold_summaries.append(
            {
                "fold": fold,
                "market_observations": market.height,
                "means": {
                    column: float(str(market[column].mean()))
                    for column in REGIME_NUMERIC
                    if column in market.columns and not column.startswith("P_")
                },
                "session_frequencies": {
                    str(value): float(str((market["session"] == value).mean()))
                    for value in sorted(market["session"].unique().to_list())
                },
                "outcome_frequencies": {
                    outcome: float(str((side_rows["actual_outcome"] == outcome).mean()))
                    for outcome in ("TARGET_FIRST", "STOP_FIRST", "NEITHER")
                },
                "mean_probabilities": {
                    column: float(str(side_rows[column].mean()))
                    for column in ("P_TARGET", "P_STOP", "P_NEITHER")
                },
            }
        )
    return {"fold_summaries": fold_summaries, "comparisons": results}


def _candidate_anatomy(frame: pl.DataFrame) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    candidates = frame.filter(pl.col("is_D10_candidate")).sort(["timestamp_utc", "side"])
    columns = (
        "timestamp_utc",
        "side",
        "fold",
        "role",
        "session",
        "predicted_EV",
        "conservative_EV",
        "P_TARGET",
        "P_STOP",
        "P_NEITHER",
        "actual_outcome",
        "actual_executable_pnl",
        "actual_BASE_net_pnl",
        "MFE",
        "MAE",
    )
    records = tuple(candidates.select(columns).to_dicts())
    summary: dict[str, Any] = {"interpretation": "EXPLORATORY_POST_HOC"}
    for role in ("MODEL_DEVELOPMENT", "INTERNAL_FREEZE_CHECK"):
        scoped = candidates.filter(pl.col("role") == role)
        if scoped.is_empty():
            summary[role] = {
                "candidates": 0,
                "mean_predicted_EV": None,
                "mean_conservative_EV": None,
                "mean_BASE_pnl": None,
                "outcomes": {outcome: 0 for outcome in ("TARGET_FIRST", "STOP_FIRST", "NEITHER")},
                "sessions": {},
            }
            continue
        summary[role] = {
            "candidates": scoped.height,
            "mean_predicted_EV": float(str(scoped["predicted_EV"].mean())),
            "mean_conservative_EV": float(str(scoped["conservative_EV"].mean())),
            "mean_BASE_pnl": float(str(scoped["actual_BASE_net_pnl"].mean())),
            "outcomes": {
                outcome: int((scoped["actual_outcome"] == outcome).sum())
                for outcome in ("TARGET_FIRST", "STOP_FIRST", "NEITHER")
            },
            "sessions": {
                str(value): int((scoped["session"] == value).sum())
                for value in sorted(scoped["session"].unique().to_list())
            },
        }
    return records, summary


def _flatten_calibration(rows: tuple[dict[str, Any], ...]) -> pl.DataFrame:
    flattened = []
    for row in rows:
        for group in row["groups"]:
            flattened.append({"scope": row["scope"], "side": row["side"], **group})
    return pl.DataFrame(flattened)


def _render_candidate_anatomy(rows: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# D.11 — Candidate Anatomy",
        "",
        "Toutes les observations ci-dessous sont **EXPLORATORY_POST_HOC**.",
        "",
        "| Timestamp | Side | Fold | Session | Pred EV | Cons EV | P(T/S/N) | "
        "Outcome | Exec | BASE | MFE | MAE |",
        "|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['timestamp_utc']} | {row['side']} | {row['fold']} | {row['session']} | "
            f"{row['predicted_EV']:.4f} | {row['conservative_EV']:.4f} | "
            f"{row['P_TARGET']:.3f}/{row['P_STOP']:.3f}/{row['P_NEITHER']:.3f} | "
            f"{row['actual_outcome']} | {row['actual_executable_pnl']:.4f} | "
            f"{row['actual_BASE_net_pnl']:.4f} | {row['MFE']:.4f} | {row['MAE']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _render_regime_shift(regime: dict[str, Any]) -> str:
    lines = [
        "# D.11 — Regime Shift Report",
        "",
        "Mesures descriptives uniquement ; aucune règle ou sélection n'en est dérivée.",
        "",
        "## Comparaisons numériques",
        "",
        "| Comparaison | Variable | SMD | PSI | Wasserstein normalisé |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison in regime["comparisons"]:
        for variable, metrics in comparison["numeric"].items():
            lines.append(
                f"| {comparison['comparison']} | {variable} | "
                f"{metrics['standardized_mean_difference']:.4f} | {metrics['PSI']:.4f} | "
                f"{metrics['normalized_wasserstein']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Comparaisons catégorielles",
            "",
            "| Comparaison | Variable | PSI |",
            "|---|---|---:|",
        ]
    )
    for comparison in regime["comparisons"]:
        for variable, metrics in comparison["categorical"].items():
            lines.append(f"| {comparison['comparison']} | {variable} | {metrics['PSI']:.4f} |")
    return "\n".join(lines) + "\n"


class ResearchD11Engine:
    def run(
        self,
        *,
        data_root: Path,
        start_utc: datetime,
        end_utc: datetime,
        protocol_path: Path,
        repository_root: Path,
    ) -> ResearchD11Report:
        start, end = as_utc(start_utc), as_utc(end_utc)
        dataset_path = _validate_research_scope(data_root, start, end)
        hashes = _verify_d10_hashes(data_root, repository_root)
        tracemalloc.start()
        started = perf_counter()
        frame = pl.read_parquet(dataset_path).sort("timestamp_utc")
        observations, safety, purge_audit = reproduce_d10_oof(frame, start, end)
        development_candidates = observations.filter(
            pl.col("is_D10_candidate") & (pl.col("role") == "MODEL_DEVELOPMENT")
        ).height
        freeze_candidates = observations.filter(
            pl.col("is_D10_candidate") & (pl.col("role") == "INTERNAL_FREEZE_CHECK")
        ).height
        if development_candidates != 14 or freeze_candidates != 4:
            raise RuntimeError("D.11 did not reproduce the frozen 14/4 D.10 candidates")
        distributions = _ev_distribution(observations)
        correlations = _ranking_correlations(observations)
        curves = _ranking_curves(observations)
        calibration = _economic_calibration(observations)
        regime = _regime_shift(observations)
        anatomy, anatomy_summary = _candidate_anatomy(observations)
        all_combined_correlation = next(
            row
            for row in correlations
            if row["scope"] == "ALL_VALIDATION" and row["side"] == "COMBINED"
        )
        fold_top5 = [
            row
            for row in curves
            if row["scope"].startswith("FOLD_")
            and row["side"] == "COMBINED"
            and row["top_fraction"] == 0.05
        ]
        positive_uplift_folds = sum(
            cast(float, row["uplift_vs_unconditional_BASE"]) > 0 for row in fold_top5
        )
        all_top5 = next(
            row
            for row in curves
            if row["scope"] == "ALL_VALIDATION"
            and row["side"] == "COMBINED"
            and row["top_fraction"] == 0.05
        )
        all_calibration = next(
            row
            for row in calibration
            if row["scope"] == "ALL_VALIDATION" and row["side"] == "COMBINED"
        )
        decile_actual = np.array(
            [group["mean_realized_BASE_pnl"] for group in all_calibration["groups"]]
        )
        monotonic_correlation = _correlation(
            "spearman", np.arange(1, 11, dtype=float), decile_actual
        )
        pooled_spearman = all_combined_correlation["spearman_predicted_EV"]
        pooled_ci_lower = all_combined_correlation["spearman_predicted_EV_daily_block_ci95"][
            "lower"
        ]
        criteria = {
            "top_5pct_positive_uplift_in_at_least_3_of_4_folds": positive_uplift_folds >= 3,
            "positive_global_decile_monotonicity": (
                monotonic_correlation is not None and monotonic_correlation > 0
            ),
            "positive_pooled_spearman": pooled_spearman is not None and pooled_spearman > 0,
            "positive_pooled_spearman_ci_lower": (
                pooled_ci_lower is not None and pooled_ci_lower > 0
            ),
            "top_5pct_largest_session_share_lte_60pct": (all_top5["largest_session_share"] <= 0.60),
        }
        worth = all(criteria.values())
        aggregate_hint = cast(float, all_top5["uplift_vs_unconditional_BASE"]) > 0 or (
            pooled_spearman is not None and pooled_spearman > 0
        )
        if worth:
            conclusion: D11Conclusion = "D11_RANKING_SIGNAL_WORTH_FURTHER_RESEARCH"
        elif aggregate_hint:
            conclusion = "D11_UNSTABLE_RANKING_SIGNAL"
        else:
            conclusion = "D11_NO_RANKING_SIGNAL"
        evaluation_dir = (
            data_root / "evaluations" / "research-d11-20260610T000000Z-20260908T000000Z"
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        observation_path = evaluation_dir / "ranking-observations.parquet"
        calibration_path = evaluation_dir / "ev-calibration.parquet"
        regime_json_path = evaluation_dir / "regime-shift-report.json"
        regime_md_path = evaluation_dir / "regime-shift-report.md"
        anatomy_json_path = evaluation_dir / "candidate-anatomy-report.json"
        anatomy_md_path = evaluation_dir / "candidate-anatomy-report.md"
        observations.write_parquet(observation_path, compression="zstd", statistics=True)
        _flatten_calibration(calibration).write_parquet(
            calibration_path, compression="zstd", statistics=True
        )
        regime_json_path.write_text(json.dumps(regime, indent=2), encoding="utf-8")
        regime_md_path.write_text(_render_regime_shift(regime), encoding="utf-8")
        anatomy_json_path.write_text(json.dumps(anatomy, indent=2, default=str), encoding="utf-8")
        anatomy_md_path.write_text(_render_candidate_anatomy(anatomy), encoding="utf-8")
        elapsed = perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        source_hashes = {
            "research_d11.py": _sha256(Path(__file__)),
            "research_d10.py": EXPECTED_D10_ENGINE_SHA256,
            "research_protocol_d11.yaml": _sha256(protocol_path),
            "ranking_observations.parquet": _sha256(observation_path),
        }
        return ResearchD11Report(
            requested_start_utc=start,
            requested_end_utc=end,
            hashes_verified_before_reproduction=hashes,
            source_hashes=source_hashes,
            locked_elements_unchanged={
                "features": True,
                "models": True,
                "B02_PRIMARY": True,
                "EV_formula": True,
                "costs": True,
                "safety_buffer": True,
                "new_trading_threshold_selected": False,
            },
            ev_distribution=distributions,
            ranking_correlations=correlations,
            ranking_curves=curves,
            economic_calibration=calibration,
            fold_stability={
                "fold_top_5pct": fold_top5,
                "positive_uplift_folds": positive_uplift_folds,
                "global_decile_mean_spearman": monotonic_correlation,
                "effect_sign_changes": len(
                    {np.sign(cast(float, row["uplift_vs_unconditional_BASE"])) for row in fold_top5}
                )
                > 1,
            },
            regime_shift=regime,
            safety_buffer_audit=safety,
            candidate_anatomy=anatomy,
            candidate_anatomy_summary=anatomy_summary,
            verdict_evidence={
                "criteria": criteria,
                "all_worth_further_research_criteria_passed": worth,
                "aggregate_hint_present": aggregate_hint,
                "all_validation_top_5pct": all_top5,
                "pooled_ranking": all_combined_correlation,
            },
            artifacts={
                "ranking_observations_parquet": str(observation_path.resolve()),
                "ev_calibration_parquet": str(calibration_path.resolve()),
                "regime_shift_json": str(regime_json_path.resolve()),
                "regime_shift_md": str(regime_md_path.resolve()),
                "candidate_anatomy_json": str(anatomy_json_path.resolve()),
                "candidate_anatomy_md": str(anatomy_md_path.resolve()),
            },
            performance=D11Performance(
                elapsed_seconds=elapsed,
                source_ticks=SOURCE_TICKS,
                source_observations=frame.height,
                ranking_side_observations=observations.height,
                d10_candidates_reproduced=development_candidates + freeze_candidates,
                peak_python_memory_mb=peak / 1024 / 1024,
            ),
            conclusion=conclusion,
            limitations=(
                "D.11 is descriptive and cannot create or validate a trading threshold.",
                "All candidate anatomy findings are EXPLORATORY_POST_HOC.",
                "Daily-block intervals estimate rank correlation between UTC-day means, not an "
                "IID interval around the observation-level coefficient.",
                "The HOLDOUT remained sealed and was neither opened nor evaluated.",
                "No D.12, Phase E, sizing, order execution or live trading was created.",
            ),
        )


def render_research_d11_report(report: ResearchD11Report) -> str:
    pooled = report.verdict_evidence["pooled_ranking"]
    top5 = report.verdict_evidence["all_validation_top_5pct"]
    lines = [
        "# SNIPER Phase D.11 — Predictive Ranking & Regime Stability Audit",
        "",
        f"Verdict : **{report.conclusion}**",
        "",
        f"Prédictions side-OOF : {report.performance.ranking_side_observations:,}",
        f"Candidats D.10 reproduits : {report.performance.d10_candidates_reproduced}",
        f"Temps : {report.performance.elapsed_seconds:.2f} s",
        f"Peak mémoire Python : {report.performance.peak_python_memory_mb:.2f} MiB",
        "",
        "## Ranking agrégé",
        "",
        f"- Spearman predicted EV / BASE : {pooled['spearman_predicted_EV']}",
        f"- Kendall tau : {pooled['kendall_predicted_EV']}",
        f"- Spearman conservative EV / BASE : {pooled['spearman_conservative_EV']}",
        f"- Top 5 % N : {top5['N']}",
        f"- Top 5 % BASE expectancy : {top5['BASE_expectancy']}",
        f"- Top 5 % uplift : {top5['uplift_vs_unconditional_BASE']}",
        "",
        "## Verdict préenregistré",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in report.verdict_evidence["criteria"].items()
        ],
        "",
        "## Garde-fous",
        "",
        "- Diagnostic uniquement ; aucun seuil sélectionné",
        "- D.10 inchangée : D10_INSUFFICIENT_EVIDENCE",
        "- HOLDOUT : SEALED, non ouvert, non évalué",
        "- D.12 et Phase E : NON COMMENCÉES",
        "- Sizing et trading live : ABSENTS",
        "",
    ]
    return "\n".join(lines)
