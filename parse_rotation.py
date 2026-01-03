import re
import csv
from pathlib import Path
from collections import Counter

# === пути как в основном скрипте ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LOG_FILE = DATA_DIR / "phase_rotation_log.txt"
OUT_CSV = DATA_DIR / "points_dataset.csv"

# ===== REGEX =====
hdr = re.compile(r"----- DIR\s+(\d+)\s+ANG\s+([+-]?\d+\.\d+)°")
reg = re.compile(r"regime=\s*([0-9a-f]+)", re.I)
ent = re.compile(r"entropy\(first\)=\s*([\d\.]+)")
txt = re.compile(r"text\(first line\)=\s*(.+)", re.I)

rows = []
freq = Counter()

current_dir = None
current_ang = None
current_entropy = None
current_text = None

# ==== проверяем, есть ли файл ====
if not LOG_FILE.exists():
    raise FileNotFoundError(f"Лог не найден: {LOG_FILE}")

print(f"Читаю лог: {LOG_FILE}")

with LOG_FILE.open("r", encoding="utf-8") as f:
    for line in f:

        m = hdr.search(line)
        if m:
            current_dir = int(m.group(1))
            current_ang = float(m.group(2))
            current_entropy = None
            current_text = None
            continue

        m = ent.search(line)
        if m:
            current_entropy = float(m.group(1))
            continue

        m = txt.search(line)
        if m:
            current_text = m.group(1).strip()
            continue

        m = reg.search(line)
        if m and current_dir is not None:
            h = m.group(1).lower()
            freq[h] += 1

            rows.append((
                current_dir,
                current_ang,
                h,
                current_entropy,
                current_text
            ))

# ---- убираем только ПОЛНЫЕ дубликаты ----
rows = sorted(set(rows))

# ---- сохраняем CSV ----
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dir","angle","regime","entropy","text"])
    w.writerows(rows)

print(f"\n💾 Saved dataset → {OUT_CSV}")

print("\n🔝 Top regimes:")
for h, c in freq.most_common(10):
    print(h, c)
