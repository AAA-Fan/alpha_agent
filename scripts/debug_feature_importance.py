#!/usr/bin/env python3
"""
Stage 0 — Plan B: Feature Importance Analysis
Loads the V2 LightGBM model and prints feature importance ranked by gain and split count.
Also categorises features into groups to show which category the model relies on most.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb

# ── paths ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("FORECAST_LGB_MODEL_PATH", "data/forecast_model_v2.lgb")
META_PATH = os.getenv("FORECAST_LGB_META_PATH", "data/forecast_model_v2_meta.json")


def load_model_and_meta():
    booster = lgb.Booster(model_file=MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return booster, meta


def categorise_feature(name: str, meta: dict) -> str:
    """Return a human-readable category for a feature name."""
    rank_cols = set(meta.get("rank_feature_columns", []))
    regime_cols = set(meta.get("regime_features", []))
    macro_fund_cols = set(meta.get("macro_fundamental_features", []))
    cat_cols = set(meta.get("categorical_features", []))
    base_cols = set(meta.get("feature_columns", []))

    if name in rank_cols:
        return "RANK"
    if name in regime_cols:
        return "REGIME"
    if name in macro_fund_cols:
        return "MACRO/FUND"
    if name in cat_cols:
        return "CATEGORICAL"
    if name in base_cols:
        return "BASE (tech)"
    return "UNKNOWN"


def main():
    booster, meta = load_model_and_meta()

    feature_names = booster.feature_name()
    gain_importance = booster.feature_importance(importance_type="gain")
    split_importance = booster.feature_importance(importance_type="split")

    # normalise to percentages
    gain_total = gain_importance.sum() if gain_importance.sum() > 0 else 1
    split_total = split_importance.sum() if split_importance.sum() > 0 else 1

    rows = []
    for i, name in enumerate(feature_names):
        cat = categorise_feature(name, meta)
        rows.append({
            "name": name,
            "category": cat,
            "gain": gain_importance[i],
            "gain_pct": gain_importance[i] / gain_total * 100,
            "split": split_importance[i],
            "split_pct": split_importance[i] / split_total * 100,
        })

    # ── sort by gain ──
    rows.sort(key=lambda r: r["gain"], reverse=True)

    print("=" * 80)
    print("  Feature Importance Report — V2 LightGBM Model")
    print("=" * 80)
    print(f"\n  Model: {MODEL_PATH}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Num trees: {booster.num_trees()}")
    print(f"  Best iteration: {meta.get('training_info', {}).get('best_iteration', '?')}")
    print()

    # ── Top features by gain ──
    print("── Top 30 Features by Gain ─────────────────────────────────────────")
    print(f"  {'Rank':<5} {'Feature':<35} {'Category':<12} {'Gain%':>7} {'Split%':>7} {'Splits':>7}")
    print("  " + "─" * 75)
    for idx, r in enumerate(rows[:30], 1):
        bar = "█" * int(r["gain_pct"] / max(rows[0]["gain_pct"], 1) * 20)
        print(f"  {idx:<5} {r['name']:<35} {r['category']:<12} {r['gain_pct']:>6.2f}% {r['split_pct']:>6.2f}% {r['split']:>7.0f}  {bar}")

    # ── Category breakdown ──
    print("\n── Category Breakdown ──────────────────────────────────────────────")
    cat_gain = {}
    cat_split = {}
    cat_count = {}
    for r in rows:
        c = r["category"]
        cat_gain[c] = cat_gain.get(c, 0) + r["gain_pct"]
        cat_split[c] = cat_split.get(c, 0) + r["split_pct"]
        cat_count[c] = cat_count.get(c, 0) + 1

    print(f"  {'Category':<15} {'#Feats':>7} {'Gain%':>8} {'Split%':>8} {'Avg Gain%':>10}")
    print("  " + "─" * 50)
    for c in sorted(cat_gain, key=lambda x: cat_gain[x], reverse=True):
        avg = cat_gain[c] / cat_count[c] if cat_count[c] > 0 else 0
        bar = "█" * int(cat_gain[c] / 2)
        print(f"  {c:<15} {cat_count[c]:>7} {cat_gain[c]:>7.2f}% {cat_split[c]:>7.2f}% {avg:>9.2f}%  {bar}")

    # ── Zero-importance features ──
    zero_gain = [r for r in rows if r["gain"] == 0]
    if zero_gain:
        print(f"\n── Zero-Gain Features ({len(zero_gain)}) ────────────────────────────────────")
        for r in zero_gain:
            print(f"  {r['name']:<35} {r['category']}")

    # ── Diagnostic summary ──
    print("\n── Diagnostic Summary ─────────────────────────────────────────────")
    rank_gain = cat_gain.get("RANK", 0)
    base_gain = cat_gain.get("BASE (tech)", 0)
    regime_gain = cat_gain.get("REGIME", 0)
    macro_gain = cat_gain.get("MACRO/FUND", 0)

    print(f"  BASE (tech) features contribute:  {base_gain:.1f}% of total gain")
    print(f"  RANK features contribute:         {rank_gain:.1f}% of total gain")
    print(f"  REGIME features contribute:       {regime_gain:.1f}% of total gain")
    print(f"  MACRO/FUND features contribute:   {macro_gain:.1f}% of total gain")
    print(f"  CATEGORICAL features contribute:  {cat_gain.get('CATEGORICAL', 0):.1f}% of total gain")

    print()
    if rank_gain > 40:
        print("  ⚠️  RANK features dominate (>40% gain).")
        print("     → Single-ticker inference will fail because rank features = 0.5 (constant).")
        print("     → The model is fundamentally a cross-sectional model, not a time-series model.")
    if base_gain < 30:
        print("  ⚠️  BASE (tech) features contribute <30% of gain.")
        print("     → The model has weak time-series signal; technical features alone are insufficient.")
    if regime_gain > 20:
        print("  ℹ️  REGIME features are important (>20% gain).")
        print("     → Backtest without RegimeAgent will lose this signal.")

    # ── CV metrics reminder ──
    cv = meta.get("cv_metrics", {})
    if cv:
        print(f"\n── Training CV Metrics (reminder) ──────────────────────────────────")
        print(f"  Mean AUC:      {cv.get('mean_auc', '?')}")
        print(f"  Mean Accuracy: {cv.get('mean_accuracy', '?')}")
        print(f"  Mean Log Loss: {cv.get('mean_log_loss', '?')}")
        aucs = [f["auc"] for f in cv.get("fold_details", [])]
        if aucs:
            print(f"  AUC range:     [{min(aucs):.4f}, {max(aucs):.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
