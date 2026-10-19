import hashlib, json, random, re
from datetime import datetime
from pathlib import Path
import pandas as pd, yaml
from scipy import stats
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH, SEEDS_PATH = ROOT / "config" / "assumptions.yaml", ROOT / "config" / "seeds.yaml"
OUT_TABLES, OUT_META = ROOT / "outputs" / "tables", ROOT / "outputs" / "meta"
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
NOM = {10: "десять", 15: "пятнадцать", 20: "двадцать", 25: "двадцать пять", 30: "тридцать"}
ADJ = {10: "десятипроцентная", 15: "пятнадцатипроцентная", 20: "двадцатипроцентная", 25: "двадцатипятипроцентная", 30: "тридцатипроцентная"}
SERV = ["на кофе", "на топливо", "на билеты в кино", "на доставку еды"]
GREET = ["Здравствуйте! ", "Добрый день! ", "", "Рады видеть вас в Городе. "]
CLOSE = ["Условия в приложении.", "Подробности в разделе Город.", "Будем рады видеть вас снова.", "Предложение действует в приложении."]
INV_IN = ["Плюс кэшбэк 5% на следующую покупку.", "Доставка бесплатная.", "К заказу полагается подарок."]
INV_OUT = ["Плюс приятный бонус на следующий заказ.", "Часть суммы вернём на счёт.", "Дополнительные привилегии уже ждут вас."]
PR_IN = ["Только сегодня!", "Последний шанс!", "Успейте воспользоваться!"]
PR_OUT = ["Завтра будет поздно.", "Не откладывайте на потом.", "Второго шанса не будет."]
PM_IN = ["Гарантируем выгоду.", "Выгода без условий."]
PM_OUT = ["Вы точно сэкономите.", "Это выгодно каждому."]
SEVERITY = {"wrong_discount": "S3", "wrong_date": "S3", "wrong_min": "S3", "missing_condition": "S3", "invented_offer": "S3",
            "ty_form": "S2", "pressure": "S2", "promise": "S2", "typo": "S1", "spacing": "S1"}
LEX_INV = re.compile(r"кеш?бэк|кэшбэк|бесплатн|подарок|гарантир|без условий", re.I)
LEX_PR = re.compile(r"только сегодня|последний шанс|успейте|не упустите|срочно", re.I)
LEX_TY = re.compile(r"\b(ты|тебе|тебя|твой|твоя|твоё|получи|успей|привет)\b", re.I)
LEX_SP = re.compile(r"  |\s[,.!?]|!!")


class Gen:
    def __init__(self, seed): self.r = random.Random(seed)
    def cfg(self):
        r = self.r; mn = r.choice([500, 700, 1000, 1500]) if r.random() < 0.65 else None
        return dict(d=r.choice(list(NOM)), day=r.randint(8, 28), month=r.choice([10, 11]), min=mn, serv=r.choice(SERV))
    def disc(self, d, form=None):
        form = form if form is not None else self.r.choices(range(5), [0.55, 0.10, 0.12, 0.12, 0.11])[0]
        return [f"скидка {d}%", f"скидка {d} %", f"скидка {d} процентов", f"скидка {NOM[d]} процентов", f"{ADJ[d]} скидка"][form]
    def date(self, day, month):
        f = self.r.choices(range(3), [0.4, 0.4, 0.2])[0]
        return [f"до {day:02d}.{month:02d}", f"до {day} {MONTHS[month-1]}", f"по {day} {MONTHS[month-1]} включительно"][f]
    def minp(self, m):
        if m is None: return ""
        return " " + self.r.choice([f"при заказе от {m} ₽", (f"при заказе от {m//1000} {m%1000:03d} руб." if m >= 1000 else f"при заказе от {m} руб."), f"при заказе от {m} рублей"])
    def render(self, c, d=None, day=None, month=None, mn="cfg", extra="", ty=None):
        d = c["d"] if d is None else d; day = c["day"] if day is None else day; month = c["month"] if month is None else month
        mn = c["min"] if mn == "cfg" else mn
        dp = self.disc(d, 0 if ty else None); tail = f"{c['serv']} {self.date(day, month)}{self.minp(mn)}. {self.r.choice(CLOSE)}"
        if ty == "ty_hi": t = f"Привет! Для тебя {dp} {tail}"
        elif ty == "ty_get": t = f"Получи {dp} {tail}"
        elif ty == "ty_out": t = f"Забирай {dp} {tail}"
        else: t = f"{self.r.choice(GREET)}Для вас {dp} {tail}{(' ' + extra) if extra else ''}"
        return t.replace("..", ".")
    def typo(self, text):
        w = self.r.choice(re.findall(r"[а-яё]{6,}", text)); i = self.r.randint(1, len(w) - 3)
        w2 = w[:i] + w[i+1] + w[i] + w[i+2:]
        return text.replace(w, w2 if w2 != w else w[:i] + w[i]*2 + w[i+1:], 1)


def generate_corpus(counts: dict, n_clean: int, seed: int) -> pd.DataFrame:
    g = Gen(seed); r = g.r; rows = []
    plan = [("clean", n_clean)] + list(counts.items())
    for cls, n in plan:
        for _ in range(n):
            c = g.cfg()
            if cls in ("wrong_min", "missing_condition") and c["min"] is None: c["min"] = r.choice([500, 700, 1000, 1500])
            if cls == "clean": t = g.render(c)
            elif cls == "wrong_discount": t = g.render(c, d=r.choice([x for x in NOM if x != c["d"]]))
            elif cls == "wrong_date": t = g.render(c, day=min(28, max(1, c["day"] + r.choice([-6, -4, -2, 2, 3, 5]))))
            elif cls == "wrong_min": t = g.render(c, mn=r.choice([m for m in (500, 700, 1000, 1500, 2000) if m != c["min"]]))
            elif cls == "missing_condition": t = g.render(c, mn=None)
            elif cls == "invented_offer": t = g.render(c, extra=r.choice(INV_IN if r.random() < 0.5 else INV_OUT))
            elif cls == "ty_form": t = g.render(c, ty=r.choice(["ty_hi", "ty_get", "ty_out"]))
            elif cls == "pressure": t = g.render(c, extra=r.choice(PR_IN if r.random() < 0.5 else PR_OUT))
            elif cls == "promise": t = g.render(c, extra=r.choice(PM_IN if r.random() < 0.5 else PM_OUT))
            elif cls == "typo": t = g.typo(g.render(c))
            else: t = g.render(c).replace(". ", " . ", 1) if r.random() < 0.5 else g.render(c).replace(" ", "  ", 2)
            rows.append(dict(cls=cls, severity=SEVERITY.get(cls), text=t, **c))
    return pd.DataFrame(rows)


def ext_discount(t):
    v = {int(m) for m in re.findall(r"(\d{1,2})\s?%", t)} | {int(m) for m in re.findall(r"(\d{1,2})\s+процент", t)}
    return v | {k for k, w in NOM.items() if re.search(w + r"\s+процент", t)}
def ext_date(t):
    v = {(int(a), int(b)) for a, b in re.findall(r"(\d{1,2})\.(\d{2})", t)}
    for a, mn in re.findall(r"(\d{1,2})\s+(" + "|".join(MONTHS) + ")", t): v.add((int(a), MONTHS.index(mn) + 1))
    return v
def ext_min(t): return {int(re.sub(r"\s", "", a)) for a, _ in re.findall(r"от\s+(\d[\d\s]*?)\s*(₽|руб)", t)}


def check_anchor(r, strict=False) -> bool:
    t = r["text"]; dv, tv, mv = ext_discount(t), ext_date(t), ext_min(t); fl = False
    if dv and dv != {r["d"]}: fl = True
    if tv and tv != {(r["day"], r["month"])}: fl = True
    if strict and (not dv or not tv): fl = True
    m0 = None if pd.isna(r["min"]) else int(r["min"])
    if m0 is not None and mv != {m0}: fl = True
    if m0 is None and mv: fl = True
    return fl
def check_whitelist(r) -> bool:
    t = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", r["text"])
    allowed = {r["d"], r["day"], r["month"]} | (set() if pd.isna(r["min"]) else {int(r["min"])})
    return any(int(x) not in allowed for x in re.findall(r"\d+", t))
def check_lexicon(r) -> bool:
    t = r["text"]; return bool(LEX_INV.search(t) or LEX_PR.search(t) or LEX_TY.search(t) or LEX_SP.search(t))


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["anchor"] = x.apply(check_anchor, axis=1); x["anchor_strict"] = x.apply(lambda r: check_anchor(r, True), axis=1)
    x["whitelist"] = x.apply(check_whitelist, axis=1); x["lexicon"] = x.apply(check_lexicon, axis=1)
    x["flag"] = x.anchor | x.whitelist | x.lexicon; x["flag_strict"] = x.anchor_strict | x.whitelist | x.lexicon
    return x


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1); hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarize(x: pd.DataFrame):
    clean = x[x.cls == "clean"]; rows = []
    for cls in [c for c in x.cls.unique() if c != "clean"]:
        s = x[x.cls == cls]; k = int(s.flag.sum()); lo, hi = clopper_pearson(k, len(s))
        rows.append(dict(error_class=cls, severity=SEVERITY[cls], n=len(s), caught=k, recall=k / len(s), ci_low=lo, ci_high=hi, recall_strict=float(s.flag_strict.mean())))
    by_class = pd.DataFrame(rows); out = []
    for sev in ("S3", "S2", "S1"):
        s = x[x.severity == sev]; k = int(s.flag.sum()); lo, hi = clopper_pearson(k, len(s)); out.append(dict(group=sev, n=len(s), caught=k, recall=k / len(s), ci_low=lo, ci_high=hi))
    kf = int(clean.flag.sum()); lo, hi = clopper_pearson(kf, len(clean)); out.append(dict(group="false_alarm_clean", n=len(clean), caught=kf, recall=kf / len(clean), ci_low=lo, ci_high=hi))
    ks = int(clean.flag_strict.sum()); lo, hi = clopper_pearson(ks, len(clean)); out.append(dict(group="false_alarm_clean_strict_mode", n=len(clean), caught=ks, recall=ks / len(clean), ci_low=lo, ci_high=hi))
    return by_class, pd.DataFrame(out)


def required_llm_recall(rules_recall: float, target: float = 0.98) -> float:
    return max(0.0, 1 - (1 - target) / (1 - rules_recall))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))["judge_benchmark"]; seed = int(yaml.safe_load(open(SEEDS_PATH))["global_seed"])
    df = generate_corpus(cfg["class_counts"], int(cfg["n_clean"]), seed); x = evaluate(df)
    by_class, summary = summarize(x)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True); OUT_TABLES.mkdir(parents=True, exist_ok=True); OUT_META.mkdir(parents=True, exist_ok=True)
    x[["cls", "severity", "text", "flag", "flag_strict"]].to_csv(ROOT / "data" / "processed" / "judge_synthetic_corpus.csv", index=False, encoding="utf-8-sig")
    by_class.to_csv(OUT_TABLES / "judge_benchmark_by_class.csv", index=False); summary.to_csv(OUT_TABLES / "judge_benchmark_summary.csv", index=False)
    s3 = float(summary.set_index("group").loc["S3", "recall"]); need = required_llm_recall(s3, float(cfg["required_cascade_recall"]))
    json.dump({"script": "judge_benchmark.py", "timestamp": datetime.now().isoformat(), "seed": seed, "synthetic": True, "n_messages": len(df), "s3_rules_recall": s3,
               "llm_residual_recall_required_if_independent": need, "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]},
              open(OUT_META / "judge_benchmark.json", "w", encoding="utf-8"), indent=2)
    print(summary.round(3).to_string(index=False)); print(by_class.round(3).to_string(index=False)); print(f"LLM residual recall needed (independence assumed): {need:.3f}")


if __name__ == "__main__":
    main()
