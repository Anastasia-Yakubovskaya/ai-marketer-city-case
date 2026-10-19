# AI-маркетолог для Города

Репозиторий содержит аналитический прототип для кейса AI-маркетолога. Он воспроизводит расчёты, на которых основаны выводы презентации: сроки запуска, дизайн эксперимента, ограничения по риску и уровню автоматизации, стоимость пилота и экономика масштабирования.

Это не production-реализация Campaign OS. Здесь нет интеграций с внутренними системами банка, реального LLM-сервиса и запуска коммуникаций. Код нужен для проверки гипотез и управленческих решений на доступных данных и явных допущениях.

## Основные материалы

- `report.pdf` — основная презентация на 12 слайдов.
- `report_with_appendix.pdf` — та же презентация с расчётным приложением.
- `Pilot-Charter-Campaign-OS-FINAL.pdf` — одностраничный устав пилота.


## Что можно проверить

| Блок | Что проверяем |
|---|---|
| Процесс | насколько можно сократить время запуска и как влияет юридическая очередь |
| Эксперименты | сколько клиентов нужно для оценки GMV, non-inferiority и почему экономический порог 4,6% нельзя считать статистически доказанным в M6 |
| Безопасность | какое качество требуется от Judge, какой охват допустим и как работает общий лимит контактов |
| Экономика | сколько стоит пилот, как меняется окупаемость при полном и 50% eligible-охвате и что происходит при нулевом uplift |
| Доказательства | откуда взялось каждое ключевое число и к какому типу оно относится |

## Быстрый запуск

Требуется Python 3.11+.

```bash
pip install -r requirements.txt
python run_all.py
pytest -q
```

`python run_all.py` пересчитывает модели и обновляет таблицы и метаданные в `outputs/`.

Графики для внутренней проверки можно пересобрать отдельно:

```bash
python run_all.py --figures
```

В текущей версии — 32 теста.

## Важные границы интерпретации

- Primary business metric — **incremental GMV vs control**.
- `campaign vs holdout` измеряет инкрементальный эффект кампании; `AI vs human` отвечает на вопрос неухудшения и сравнительного эффекта AI.
- В MDE используется 10% контроль внутри конкретного эксперимента; постоянный holdout 1–2% — отдельный долгосрочный контур для cumulative effect и fatigue.
- Порог `δ = 4,6%` — экономический break-even для полного eligible-контура, а не статистический M6-гейт. В M6 оцениваются эффект и 95% ДИ; при недостаточной мощности продолжается контролируемый эксперимент.
- `100% coverage` в экономике означает полный **eligible post-pilot contour**, а не все типы кампаний и не весь бюджет банка. Есть отдельная sensitivity-модель для 50% eligible-ценности.
- Adoption `≥70%` означает долю **eligible-кампаний**, в которых команда использует Campaign OS.
- В пилоте каждую отправку в prod подтверждает человек. Green означает больший допустимый охват и больше автоматической подготовки, но не auto-send.
- Для one-click отдельно измеряются active review (цель 1–2 минуты) и lead time очереди подтверждения (target p50 ≤ 0,5 дня).
- Значения, которых нет в исходном кейсе и для которых не приложен документированный ответ организаторов, маркируются как `ASSUMPTION`.

## Структура

```text
ai-marketer-city-case/
├── README.md
├── requirements.txt
├── run_all.py
├── report.pdf
├── report_with_appendix.pdf
├── report.pptx
├── report_with_appendix.pptx
├── Pilot-Charter-Campaign-OS-FINAL.pdf
├── Pilot-Charter-Campaign-OS-FINAL.html
│
├── config/
│   ├── assumptions.yaml
│   └── seeds.yaml
│
├── data/
│   └── processed/
│       ├── evidence_registry.csv
│       ├── stop_thresholds.csv
│       ├── red_team_cases.csv
│       ├── judge_synthetic_corpus.csv
│       ├── fmea_assumptions.csv
│       └── campaign_portfolio_assumptions.csv
│
├── src/
│   ├── process/
│   ├── experiments/
│   ├── safety/
│   ├── economics/
│   ├── governance/
│   └── reporting/
│
├── tests/
├── outputs/
│   ├── meta/
│   ├── tables/
│   └── reports/
│
└── docs/
    ├── methodology.md
    ├── learning_agenda.md
    ├── campaign_card_spec.md
    ├── working_assumptions.md
    └── one_click_card.html
```

## Ключевые файлы расчётов

### Процесс

- `src/process/process_model.py` — Monte Carlo текущего и целевого процесса.
- `src/process/legal_queue.py` — чувствительность к юридической очереди.

### Эксперименты

- `src/experiments/experiment_design.py` — sample size, CUPED, MDE, SRM и диагностический non-inferiority по конверсии.
- `src/experiments/gmv_sample_size.py` — выборки для GMV, AI-vs-human и feasibility экономического порога.

### Безопасность

- `src/safety/judge_statistics.py` — биномиальные границы и требования к количеству чистых примеров.
- `src/safety/judge_benchmark.py` — benchmark правил на синтетическом корпусе; не доказывает качество на реальных данных банка.
- `src/safety/judge_target.py` — модельный минимум качества Judge при рабочей таксономии ошибок.
- `src/safety/contact_policy.py` — симуляция общего лимита контактов.
- `src/safety/autonomy.py` — лимиты охвата для Yellow / Green tiers.
- `src/safety/error_budget.py` — error budget и реакция на инциденты.
- `src/safety/risk_register.py` — FMEA; S/O/D — экспертные оценки, а не измеренные вероятности.

### Экономика

- `src/economics/pilot_cost.py` — bottom-up стоимость пилота.
- `src/economics/capacity_budget.py` — сценарии высвобождаемой ёмкости и предел расходов на push.
- `src/economics/economics.py` — Monte Carlo окупаемости, break-even, tornado и sensitivity по охвату.
- `src/economics/uplift_zero.py` — downside при нулевом uplift.
- `src/economics/portfolio.py` — иллюстративная приоритизация 10 архетипов, не фактический портфель Т-Банка.

### Governance

- `src/governance/reconciliation.py` — фиксирует внутренние несостыковки входных цифр кейса.
- `src/governance/deck_numbers.py` — собирает числа для сверки с презентацией и обновляет evidence registry.
- `src/governance/evidence_registry.py` — валидирует типы доказательств.

## Типы доказательств

| Тип | Значение |
|---|---|
| `CASE_FACT` | значение прямо дано в условии кейса |
| `DERIVED_FACT` | арифметическое следствие из фактов кейса |
| `ASSUMPTION` | рабочее допущение, которое нужно проверить или заменить фактом |
| `MODEL_RESULT` | результат воспроизводимого расчёта |
| `PILOT_TARGET` | управленческий порог или цель пилота |

## Основные выходные файлы

- `outputs/meta/deck_numbers.json` — агрегированные числа для сверки с презентацией.
- `outputs/tables/process_summary.csv` — сроки процесса.
- `outputs/tables/gmv_sample_size.csv` — выборки для GMV.
- `outputs/tables/gmv_noninferiority.csv` — выборки AI-vs-human по GMV.
- `outputs/tables/economic_delta_power.csv` — объём данных, необходимый для статистического доказательства экономического порога 4,6%.
- `outputs/tables/economics_coverage_sensitivity.csv` — сравнение полного eligible-контура и 50% eligible-ценности.
- `outputs/tables/economics_modal_payback.csv` — окупаемость при модальных параметрах.
- `outputs/tables/tornado.csv` — чувствительность NPV к ключевым допущениям.
- `outputs/tables/judge_target.csv` — чувствительность цели Judge.
- `outputs/tables/contact_policy_violations.csv` — симуляция контактной политики.
- `outputs/tables/pilot_cost_build_up.csv` — структура стоимости пилота.
- `outputs/tables/uplift_zero.csv` — сценарий нулевого эффекта.

