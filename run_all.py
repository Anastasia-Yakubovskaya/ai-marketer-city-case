from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "src/governance/reconciliation.py",
    "src/process/process_model.py",
    "src/process/legal_queue.py",
    "src/experiments/experiment_design.py",
    "src/experiments/gmv_sample_size.py",
    "src/safety/judge_statistics.py",
    "src/safety/judge_benchmark.py",
    "src/safety/judge_target.py",
    "src/safety/contact_policy.py",
    "src/safety/autonomy.py",
    "src/safety/error_budget.py",
    "src/safety/risk_register.py",
    "src/economics/pilot_cost.py",
    "src/economics/capacity_budget.py",
    "src/economics/economics.py",
    "src/economics/uplift_zero.py",
    "src/economics/portfolio.py",
    "src/governance/deck_numbers.py",
    "src/governance/evidence_registry.py",
]

if "--figures" in sys.argv:
    SCRIPTS.append("src/reporting/slide_figures.py")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for script in SCRIPTS:
    print(f"[run] {script}")
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

print("расчёты и выходные файлы обновлены")
