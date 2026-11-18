"""Risk, scenarios and the BUY / SELL / HOLD decision-support signal.

Rule: a directional signal needs >= 2 independent *validated* evidence sources
and no strong contradiction; otherwise HOLD with the reason spelled out. Nothing
here is a guarantee or personal advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.analytics import volatility as vol


def validate_score(score: pd.Series, close: pd.Series, sigma: pd.Series, horizons=(1, 5, 10, 20)) -> dict:
    """Does the technical score relate to forward returns? Non-overlapping samples,
    plus a first-half / second-half stability check."""
    out = {}
    for h in horizons:
        y = vol.forward_returns(close, h)
        d = pd.DataFrame({"s": score, "y": y}).dropna().iloc[::h]
        if len(d) < 60:
            out[h] = dict(valid=False, n=len(d)); continue
        ic, p = stats.spearmanr(d.s, d.y)
        half = len(d) // 2
        ic1 = stats.spearmanr(d.s.iloc[:half], d.y.iloc[:half])[0]
        ic2 = stats.spearmanr(d.s.iloc[half:], d.y.iloc[half:])[0]
        b = pd.cut(d.s, [0, 40, 60, 100], labels=["bearish", "neutral", "bullish"])
        g = d.groupby(b, observed=True).y.agg(["count", "mean", lambda x: (x > 0).mean()])
        g.columns = ["n", "mean_ret", "up_rate"]
        out[h] = dict(valid=True, n=int(len(d)), ic=float(ic), p=float(p), ic_first_half=float(ic1),
                      ic_second_half=float(ic2), stable=bool(ic1 > 0 and ic2 > 0),
                      buckets={k: {a: float(b_) for a, b_ in v.items()} for k, v in g.to_dict("index").items()})
    ref = out.get(10, {})
    validated = bool(ref.get("valid") and ref["p"] < 0.05 and ref["ic"] > 0 and ref["stable"])
    return dict(by_horizon=out, validated=validated,
                note="امتیاز فنی ماهیت مومنتومی دارد؛ بخش بزرگی از تاریخچه روند صعودی بوده و ممکن است در شرایط دیگر ضعیف شود.")


def regime_context(ind_last: dict, reg: dict) -> dict:
    return dict(trend=reg.get("trend"), vol=reg.get("vol"))


def decide(tech: dict, tech_validation: dict, forecast: dict, patterns: dict | None, regime: dict,
           data_health: dict, h_main: int = 5) -> dict:
    reasons, evidence = [], []
    if data_health.get("blocked"):
        return dict(signal="HOLD", reasons=[data_health["blocked"]], evidence=[], validated_votes=0,
                    confidence_status="no_signal_insufficient_data")
    # 1) technical score (validated or not, always shown)
    s = tech.get("score")
    if s is not None:
        vote = 1 if s >= 60 else -1 if s <= 40 else 0
        evidence.append(dict(source="technical", vote=vote, validated=tech_validation.get("validated", False),
                             detail=f"امتیاز فنی {s:.0f}/100"))
    # 2) model forecast (only if the ensemble really beats the random walk)
    f = forecast.get("horizons", {}).get(h_main, {})
    if f:
        if f.get("ensemble_active"):
            thr = 0.25 * f["sigma_h"]
            vote = 1 if f["pred_return"] > thr else -1 if f["pred_return"] < -thr else 0
            evidence.append(dict(source="forecast", vote=vote, validated=True,
                                 detail=f"Ensemble فعال؛ بازده مورد انتظار {f['pred_return']*100:.2f}٪"))
        else:
            evidence.append(dict(source="forecast", vote=0, validated=True,
                                 detail="هیچ مدلی در بک‌تست از Random Walk بهتر نبود؛ مدل‌ها رأی جهت‌دار نمی‌دهند"))
    # 3) historical pattern analogues (descriptive -> never validated)
    if patterns and patterns.get("valid"):
        sm = patterns["summary"].get(h_main) or patterns["summary"].get(str(h_main))
        if sm and sm["n"] >= 8:
            diff = sm["up_pct"] - sm["base_rate_up_pct"]
            vote = 1 if diff >= 20 else -1 if diff <= -20 else 0
            evidence.append(dict(source="patterns", vote=vote, validated=False,
                                 detail=f"الگوهای مشابه: {sm['up_pct']:.0f}٪ صعودی در برابر نرخ پایه {sm['base_rate_up_pct']:.0f}٪"))
    val_sum = sum(e["vote"] for e in evidence if e["validated"])
    all_sum = sum(e["vote"] for e in evidence)
    n_val_dir = sum(1 for e in evidence if e["validated"] and e["vote"] != 0)
    contradict = any(e["vote"] * np.sign(all_sum or 1) < 0 for e in evidence if e["vote"] != 0) and all_sum != 0
    sig = "HOLD"
    if regime.get("vol") == "high":
        reasons.append("نوسان بالاست؛ سیگنال جهت‌دار صادر نمی‌شود مگر شواهد قوی‌تر باشد")
    needed = 2
    if val_sum >= needed and not contradict and all_sum > 0: sig = "BUY"
    elif val_sum <= -needed and not contradict and all_sum < 0: sig = "SELL"
    else:
        if n_val_dir < needed:
            reasons.append("کمتر از دو منبع شواهد اعتبارسنجی‌شده هم‌جهت وجود دارد")
        if contradict:
            reasons.append("شواهد با هم اختلاف جهت دارند")
    if sig != "HOLD" and regime.get("vol") == "high":
        sig = "HOLD"
    return dict(signal=sig, reasons=reasons, evidence=evidence, validated_votes=int(val_sum),
                confidence_status="uncalibrated_probability_not_provided",
                disclaimer="ابزار تصمیم‌یار است، نه توصیه مالی یا تضمین سود.")


def scenarios(base: float, h: int, forecast_h: dict, levels: dict, ind: dict, regime: dict) -> dict:
    lo, hi = forecast_h["lower"], forecast_h["upper"]
    sigma = forecast_h["sigma_h"]
    sup = (levels.get("supports") or [{}])[0].get("price")
    res = (levels.get("resistances") or [{}])[0].get("price")
    ema50 = ind.get("ema_50")
    return dict(
        horizon=h,
        bullish=dict(price=float(hi), return_pct=float((hi / base - 1) * 100),
                     invalidated_if=(f"کلوز زیر {ema50:,.0f} (EMA50)" if ema50 else "بازگشت زیر آخرین حمایت")),
        neutral=dict(price=float(base * np.exp(forecast_h["pred_return"])),
                     low=float(base * np.exp(-0.25 * sigma)), high=float(base * np.exp(0.25 * sigma)),
                     invalidated_if="خروج قیمت از محدوده ±۰٫۲۵σ"),
        bearish=dict(price=float(lo), return_pct=float((lo / base - 1) * 100),
                     invalidated_if=(f"کلوز بالای {res:,.0f} (مقاومت)" if res else "کلوز بالای EMA50")),
        support=sup, resistance=res)


def risk_block(base: float, sig_daily: float, forecast: dict, h: int = 5) -> dict:
    f = forecast["horizons"][h]
    downside = (f["lower"] / base - 1) * 100
    upside = (f["upper"] / base - 1) * 100
    rr = None if downside == 0 else (abs(upside) / abs(downside))
    return dict(daily_vol_pct=float(sig_daily * 100), horizon_sigma_pct=float(f["sigma_h"] * 100),
                downside_pct=float(downside), upside_pct=float(upside), range_ratio=None if rr is None else float(rr),
                disagreement=forecast.get("disagreement"), uncertainty_pct=float((f["upper"] - f["lower"]) / base * 100),
                note="نسبت دامنه‌ی بالا به پایین از بازه‌ی کالیبره‌شده می‌آید؛ بازده مورد انتظار مدل‌ها در غیاب برتری آماری صفر است.")
