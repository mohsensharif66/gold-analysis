"""Specialised analyst agents + supervisor.

Agents are deterministic: they read ONLY numbers computed by the engines in the
snapshot and phrase them. They cannot invent prices, indicators or backtest
results. An optional LLM (assistant.py) may rephrase, never add numbers.
"""
from __future__ import annotations

F = lambda x, d=0: "—" if x is None else f"{x:,.{d}f}"
STANCE_FA = {"bullish": "صعودی", "bearish": "نزولی", "neutral": "خنثی", "no_view": "بدون نظر معتبر"}


def _agent(name, title, stance, confidence, findings, refs):
    return dict(agent=name, title=title, stance=stance, stance_fa=STANCE_FA[stance],
                confidence=confidence, findings=findings, data_refs=refs)


def technical(s):
    e = s["technical"]["explain"]; v = s["technical"]["validation"]
    if e["score"] is None:
        return _agent("technical", "تحلیلگر تکنیکال", "no_view", "none", ["داده کافی برای امتیاز فنی نیست"], [])
    top = [c for c in e["components"] if c["valid"]][:3]
    f = [f"امتیاز فنی {e['score']:.0f} از ۱۰۰ ({STANCE_FA[e['label']]})"]
    f += [f"{c['label']}: {STANCE_FA[c['stance']]} (سهم {c['contribution']:+.1f})" for c in top]
    ds = s["technical"]["divergences"]
    if ds:
        d = ds[-1]; f.append(f"آخرین واگرایی تأییدشده: {d['indicator']} {'مثبت' if d['type']=='bullish' else 'منفی'} در {d['date']}")
    if e["dropped"]:
        f.append("مؤلفه‌های حذف‌شده (داده نامعتبر): " + "، ".join(e["dropped"]))
    f.append("اعتبار تاریخی امتیاز: " + ("تأییدشده (IC مثبت و پایدار)" if v["validated"] else "تأیید نشده"))
    conf = "medium" if v["validated"] else "low"
    return _agent("technical", "تحلیلگر تکنیکال", e["label"], conf, f, ["technical.explain", "technical.validation"])


def statistical(s):
    v, r = s["volatility"], s["regime"]
    f = [f"نوسان روزانه (EWMA) {v['ewma']*100:.2f}٪؛ انحراف معیار ۲۰ روزه {v['std20']*100:.2f}٪",
         f"رژیم: {r['trend_fa']}، {r['vol_fa']}" + (f" (صدک نوسان {r['vol_percentile']*100:.0f}٪)" if r['vol_percentile'] is not None else "")]
    c5 = v["calibration"].get("5") or v["calibration"].get(5)
    if c5 and c5.get("valid"):
        f.append(f"کالیبراسیون بازه ۸۰٪ پنج‌روزه: پوشش خارج‌نمونه {c5['heldout_coverage']*100:.0f}٪")
    return _agent("statistical", "تحلیلگر آماری", "neutral", "medium", f, ["volatility", "regime"])


def intermarket(s):
    i = s["intermarket"]; p = i["premium"]; cm = i["comovement"]
    live = i["live"]
    prem = live["premium_pct"] if live.get("valid") else p["last_close_based"]
    src = "لحظه‌ای" if live.get("valid") else f"بر اساس کلوز {p['last_date']}"
    f = [f"پریمیوم/دیسکانت نسبت به ارزش نظری ({src}): {prem:+.2f}٪ (میانگین تاریخی {p['mean']:+.2f}٪)"]
    if p["z60"] is not None: f.append(f"z-score ۶۰ روزه پریمیوم: {p['z60']:+.2f}")
    g = i["correlations"]["full"]["gold~usd"]["pearson"]; x = i["correlations"]["full"]["gold~xau"]["pearson"]
    f.append(f"همبستگی بازده روزانه: طلا~دلار {g:.2f}، طلا~انس {x:.2f} (همبستگی ≠ علیت)")
    if cm.get("valid"):
        f.append(f"سهم تغییرات توضیح‌داده‌شده توسط دلار و انس R²={cm['r2']:.2f}؛ انحراف ۲۰ روزه از مقدار ضمنی {cm['divergence_20']*100:+.2f}٪")
    z = p["z60"]
    stance = "no_view"
    return _agent("intermarket", "تحلیلگر بین‌بازاری", stance, "low", f, ["intermarket"])


def pattern(s):
    p = s["patterns"]["20"] if "20" in s["patterns"] else s["patterns"][20]
    if not p.get("valid"):
        return _agent("pattern", "تحلیلگر الگو", "no_view", "none", ["الگوی کافی یافت نشد"], [])
    f = []
    for h in ("5", "10"):
        sm = p["summary"][h]
        f.append(f"{h} روز پس از ۱۰ الگوی مشابه: میانه {sm['median']*100:+.2f}٪، {sm['up_pct']:.0f}٪ صعودی (نرخ پایه {sm['base_rate_up_pct']:.0f}٪)")
    sm = p["summary"]["5"]; d = sm["up_pct"] - sm["base_rate_up_pct"]
    stance = "bullish" if d >= 20 else "bearish" if d <= -20 else "neutral"
    return _agent("pattern", "تحلیلگر الگو", stance, "low", f + ["نتیجه توصیفی است و اعتبارسنجی نشده"], ["patterns"])


def forecast(s):
    fc, bt = s["forecast"], s["backtest"]
    if not fc:
        return _agent("forecast", "تحلیلگر پیش‌بینی", "no_view", "none", ["بک‌تست هنوز اجرا نشده"], [])
    h5 = fc["horizons"]["5"]; r5 = bt["5"]
    best = r5["best_single"]; bm = r5["models"][best]
    f = [f"بهترین مدل تکی در افق ۵ روز: {bm['label']} با نسبت RMSE به Random Walk برابر {bm['rmse_ratio']:.3f}",
         "Ensemble " + ("فعال است" if h5["ensemble_active"] else "غیرفعال است: برتری آماری نسبت به Random Walk دیده نشد"),
         f"قیمت مرکزی ۵ روزه {F(h5['pred_price'])} با بازه ۸۰٪ [{F(h5['lower'])} ، {F(h5['upper'])}]"]
    if h5.get("dispersion") is not None:
        f.append(f"پراکندگی نظر مدل‌ها {h5['dispersion']:.2f} برابر σ افق")
    stance = {"up": "bullish", "down": "bearish", "flat": "neutral"}.get(h5["trend_state"], "no_view")
    return _agent("forecast", "تحلیلگر پیش‌بینی", stance, "medium" if h5["ensemble_active"] else "low", f, ["forecast", "backtest"])


def risk(s):
    r, h = s.get("risk"), s["data_health"]
    f = [f"وضعیت داده: {s['quality']['status']}؛ سن آخرین کندل کامل {h['history_age_days']} روز"]
    if r:
        f += [f"دامنه ۵ روزه (بازه ۸۰٪): {r['downside_pct']:+.1f}٪ تا {r['upside_pct']:+.1f}٪",
              f"نوسان روزانه {r['daily_vol_pct']:.2f}٪؛ عدم‌قطعیت بازه {r['uncertainty_pct']:.1f}٪"]
    stale = [k for k, q in h["quotes"].items() if q.get("stale")]
    if stale:
        f.append("قیمت‌های لحظه‌ای کهنه‌اند و به‌عنوان قیمت زنده نمایش داده نمی‌شوند: " + "، ".join(stale))
    return _agent("risk", "تحلیلگر ریسک", "neutral", "medium", f, ["risk", "data_health"])


def supervisor(s) -> dict:
    ag = [technical(s), statistical(s), intermarket(s), pattern(s), forecast(s), risk(s)]
    dirs = [a for a in ag if a["stance"] in ("bullish", "bearish")]
    disagreements = []
    for i, a in enumerate(dirs):
        for b in dirs[i + 1:]:
            if a["stance"] != b["stance"]:
                disagreements.append(f"{a['title']} ({a['stance_fa']}) ≠ {b['title']} ({b['stance_fa']})")
    d = s["decision"]
    summary = [f"سیگنال نهایی: {d['signal']}"] + d["reasons"]
    return dict(agents=ag, disagreements=disagreements, signal=d["signal"], summary=summary,
                prediction_ids=s.get("prediction_ids", []), data_cutoff=s["meta"]["data_cutoff"],
                disclaimer="این خروجی ابزار تصمیم‌یار است؛ پیش‌بینی قطعی یا تضمین سود نیست.")


def daily_report(s) -> str:
    sup = supervisor(s)
    q = s["last_bar"]
    lines = [f"گزارش GoldTrend Pro — داده تا {s['meta']['data_cutoff']}",
             f"آخرین کلوز طلای ۱۸: {F(q['close'])} تومان ({q['change_pct']:+.2f}٪)", ""]
    for a in sup["agents"]:
        lines.append(f"■ {a['title']} [{a['stance_fa']}]")
        lines += [f"  - {x}" for x in a["findings"]]
    lines += ["", "اختلاف نظرها: " + ("؛ ".join(sup["disagreements"]) if sup["disagreements"] else "ندارد"),
              *sup["summary"], sup["disclaimer"]]
    return "\n".join(lines)
