"""Conversational assistant. Answers are built from the snapshot; an optional
OpenAI-compatible LLM may rephrase them, but any answer containing a number that
is not present in the snapshot is rejected and replaced by the grounded answer."""
from __future__ import annotations

import json
import os
import re

import requests

from app.agents import agents as ag
from app.core.logsafe import logger

F = ag.F
INTENTS = [
    ("signal", ("سیگنال", "بخرم", "بفروشم", "خرید", "فروش", "signal", "buy", "sell", "hold")),
    ("forecast", ("پیش‌بینی", "پیش بینی", "قیمت آینده", "forecast", "هفته", "ده روز")),
    ("risk", ("ریسک", "نوسان", "risk", "افت")),
    ("intermarket", ("دلار", "انس", "پریمیوم", "حباب", "ارزش نظری", "usd", "premium")),
    ("technical", ("تکنیکال", "rsi", "macd", "اندیکاتور", "امتیاز", "روند")),
    ("pattern", ("الگو", "مشابه", "تاریخی", "pattern")),
    ("backtest", ("دقت", "بک‌تست", "بک تست", "backtest", "عملکرد", "قابل اعتماد")),
    ("data", ("داده", "منبع", "تازه", "به‌روز", "data")),
    ("scenario", ("سناریو", "scenario")),
    ("report", ("گزارش", "report", "خلاصه")),
]


def intent_of(q: str) -> str:
    ql = q.lower()
    for name, kws in INTENTS:
        if any(k in ql for k in kws):
            return name
    return "overview"


def grounded_answer(q: str, s: dict) -> tuple[str, dict]:
    it = intent_of(q)
    sup = ag.supervisor(s)
    byname = {a["agent"]: a for a in sup["agents"]}
    refs = []
    if it in ("technical", "risk", "intermarket", "pattern"):
        a = byname[it]; text = "\n".join("• " + x for x in a["findings"]); refs = a["data_refs"]
    elif it == "forecast":
        a = byname["forecast"]; text = "\n".join("• " + x for x in a["findings"]); refs = a["data_refs"]
    elif it == "backtest":
        bt = s.get("backtest")
        if not bt:
            text = "بک‌تست هنوز اجرا نشده است."
        else:
            lines = []
            for h in ("1", "5", "10"):
                r = bt[h]; b = r["models"][r["best_single"]]
                lines.append(f"افق {h} روز: بهترین مدل {b['label']}، نسبت RMSE به Random Walk {b['rmse_ratio']:.3f}، "
                             f"دقت جهت {F(b['dir_acc']*100 if b['dir_acc'] is not None else None)}٪ (نرخ پایه {F(b['base_up_rate']*100)}٪)، "
                             f"پوشش بازه ۸۰٪: {F(b['interval_80_coverage']*100)}٪؛ Ensemble {'فعال' if r['ensemble_enabled'] else 'غیرفعال'}")
            text = "\n".join("• " + x for x in lines); refs = ["backtest"]
    elif it == "data":
        h = s["data_health"]
        text = f"وضعیت کیفیت: {s['quality']['status']}؛ سن آخرین کندل {h['history_age_days']} روز.\n" + \
            "\n".join(f"• {k}: " + ("کهنه" if q_.get("stale") else "تازه") + (f"، {q_['age_minutes']:.0f} دقیقه قبل از {q_['source']}" if q_.get("age_minutes") is not None else "")
                      for k, q_ in h["quotes"].items()); refs = ["data_health"]
    elif it == "scenario":
        sc = s.get("scenarios")
        text = "بدون بک‌تست سناریو ساخته نمی‌شود." if not sc else (
            f"سناریوی صعودی ({sc['horizon']} روز): {F(sc['bullish']['price'])} ({sc['bullish']['return_pct']:+.1f}٪) — ابطال: {sc['bullish']['invalidated_if']}\n"
            f"سناریوی خنثی: {F(sc['neutral']['low'])} تا {F(sc['neutral']['high'])} — ابطال: {sc['neutral']['invalidated_if']}\n"
            f"سناریوی نزولی: {F(sc['bearish']['price'])} ({sc['bearish']['return_pct']:+.1f}٪) — ابطال: {sc['bearish']['invalidated_if']}")
        refs = ["scenarios"]
    elif it == "report":
        text = ag.daily_report(s); refs = ["all"]
    else:  # signal / overview
        d = s["decision"]
        text = f"سیگنال: {d['signal']}\n" + "\n".join("• " + x["detail"] for x in d["evidence"]) + "\n" + \
            "\n".join("• " + r for r in d["reasons"]) + f"\n{d.get('disclaimer','')}"
        refs = ["decision"]
    return text, dict(intent=it, refs=refs, prediction_ids=s.get("prediction_ids", []), data_cutoff=s["meta"]["data_cutoff"])


def _numbers(o, acc):
    if isinstance(o, dict):
        for v in o.values(): _numbers(v, acc)
    elif isinstance(o, list):
        for v in o: _numbers(v, acc)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        acc.add(float(o)); acc.add(float(o) * 100)
    return acc


def numbers_grounded(text: str, context: dict, tol: float = 0.006) -> bool:
    ctx = _numbers(context, set())
    for m in re.findall(r"[-+]?\d[\d,]*\.?\d*", text):
        try:
            x = float(m.replace(",", ""))
        except ValueError:
            continue
        if abs(x) < 100 and float(x).is_integer():
            continue                       # small integers (counts, days, horizons)
        if not any(abs(x - c) <= tol * max(abs(c), 1e-9) or abs(abs(x) - abs(c)) <= tol * max(abs(c), 1e-9) for c in ctx):
            return False
    return True


def llm_rephrase(grounded: str, question: str, s: dict) -> str | None:
    key, base = os.getenv("LLM_API_KEY", ""), os.getenv("LLM_BASE_URL", "")
    if not key or not base:
        return None
    try:
        r = requests.post(base.rstrip("/") + "/chat/completions", timeout=30,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": os.getenv("LLM_MODEL", "default"), "temperature": 0.2,
                                "messages": [{"role": "system", "content":
                                              "فقط با استفاده از متن «پاسخ مبنا» به فارسی ساده توضیح بده. "
                                              "هیچ عدد، قیمت یا نتیجه‌ی جدیدی نساز و توصیه‌ی قطعی نده."},
                                             {"role": "user", "content": f"پرسش: {question}\n\nپاسخ مبنا:\n{grounded}"}]})
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out if numbers_grounded(out, s) or numbers_grounded(out, {"g": grounded}) else None
    except Exception as exc:
        logger.warning("LLM unavailable: {}", exc)
        return None


def answer(question: str, s: dict) -> dict:
    text, meta = grounded_answer(question, s)
    rephrased = llm_rephrase(text, question, s)
    return dict(answer=rephrased or text, grounded=text, llm_used=bool(rephrased), **meta)
