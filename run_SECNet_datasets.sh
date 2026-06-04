#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_ROOT="${LOG_ROOT:-logs/long_term_forecast/SECNet}"
log_dir="${LOG_ROOT}/${timestamp}"
task_dir="${log_dir}/datasets"
mkdir -p "$log_dir" "$task_dir"

GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"

if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "No GPU selected. Example: GPUS=\"0,1,2\" ./run_SECNet_datasets.sh"
  exit 1
fi

echo "Using GPUs: ${GPU_LIST[*]}"
echo "Logs: ${log_dir}"
echo "Parallelism: ${#GPU_LIST[@]} dataset(s) at a time"

create_dataset_tasks() {
  cat > "${task_dir}/001_Exchange.task" <<'EOF'
Exchange	./scripts/SECNet_Exchange.sh
EOF
  cat > "${task_dir}/002_ETTh1.task" <<'EOF'
ETTh1	./scripts/SECNet_ETTh1.sh
EOF
  cat > "${task_dir}/003_ETTh2.task" <<'EOF'
ETTh2	./scripts/SECNet_ETTh2.sh
EOF
  cat > "${task_dir}/004_ETTm1.task" <<'EOF'
ETTm1	./scripts/SECNet_ETTm1.sh
EOF
  cat > "${task_dir}/005_ETTm2.task" <<'EOF'
ETTm2	./scripts/SECNet_ETTm2.sh
EOF
  cat > "${task_dir}/006_Flight.task" <<'EOF'
Flight	./scripts/SECNet_Flight.sh
EOF
  cat > "${task_dir}/007_Weather.task" <<'EOF'
Weather	./scripts/SECNet_Weather.sh
EOF
}

worker() {
  local gpu="$1"

  while true; do
    local task_file claimed="" found=0

    for task_file in "$task_dir"/*.task; do
      [ -e "$task_file" ] || continue
      found=1
      claimed="${task_file%.task}.gpu${gpu}.running"
      if mv "$task_file" "$claimed" 2>/dev/null; then
        break
      fi
      claimed=""
    done

    [ "$found" -eq 0 ] && return 0
    [ -n "$claimed" ] || continue

    IFS=$'\t' read -r dataset_name script_path < "$claimed"
    local log_file="${log_dir}/${dataset_name}.log"
    local start_ts end_ts status=99
    start_ts=$(date +%s)

    {
      echo "[GPU ${gpu}] Start ${dataset_name}: ${script_path} at $(date '+%F %T')"
      CUDA_VISIBLE_DEVICES="${gpu}" bash "$script_path"
      status=$?
      end_ts=$(date +%s)
      echo "[GPU ${gpu}] End ${dataset_name} at $(date '+%F %T'), status=${status}, seconds=$((end_ts - start_ts))"
    } > "$log_file" 2>&1

    mv "$claimed" "${claimed%.running}.done"
  done
}

create_dataset_tasks

for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" &
done

wait

python - "$log_dir" <<'PY'
import csv
import re
import sys
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-9;]*m")
log_dir = Path(sys.argv[1])
rows = []

def clean(line):
    return ansi.sub("", line).strip()

for log_file in sorted(log_dir.glob("*.log")):
    current = None
    dataset_seconds = ""
    dataset_gpu = ""
    dataset_status = ""

    for raw_line in log_file.read_text(errors="ignore").splitlines():
        line = clean(raw_line)

        end_match = re.search(r"\[GPU (\d+)\] End .* status=(\d+), seconds=(\d+)", line)
        if end_match:
            dataset_gpu = end_match.group(1)
            dataset_status = end_match.group(2)
            dataset_seconds = end_match.group(3)

        if "Model ID:" in line and "Model:" in line:
            match = re.search(r"Model ID:\s*(\S+)\s+Model:\s*(\S+)", line)
            if match:
                current = {
                    "log_dataset": log_file.stem,
                    "model_id": match.group(1),
                    "model": match.group(2),
                    "log_file": str(log_file),
                }

        elif current and "Data:" in line and "Root Path:" in line:
            match = re.search(r"Data:\s*(\S+)\s+Root Path:\s*(\S+)", line)
            if match:
                current["data"] = match.group(1)
                current["root_path"] = match.group(2)

        elif current and "Data Path:" in line and "Features:" in line:
            match = re.search(r"Data Path:\s*(\S+)\s+Features:\s*(\S+)", line)
            if match:
                current["data_path"] = match.group(1)
                current["features"] = match.group(2)

        elif current and "Seq Len:" in line and "Label Len:" in line:
            match = re.search(r"Seq Len:\s*(\d+)\s+Label Len:\s*(\d+)", line)
            if match:
                current["seq_len"] = match.group(1)
                current["label_len"] = match.group(2)

        elif current and "Pred Len:" in line:
            match = re.search(r"Pred Len:\s*(\d+)", line)
            if match:
                current["pred_len"] = match.group(1)

        elif current and "Enc In:" in line and "Dec In:" in line:
            match = re.search(r"Enc In:\s*(\d+)\s+Dec In:\s*(\d+)", line)
            if match:
                current["enc_in"] = match.group(1)
                current["dec_in"] = match.group(2)

        elif current and "C Out:" in line and "d model:" in line:
            match = re.search(r"C Out:\s*(\d+)\s+d model:\s*(\d+)", line)
            if match:
                current["c_out"] = match.group(1)
                current["d_model"] = match.group(2)

        elif current and "d layers:" in line and "d FF:" in line:
            match = re.search(r"d layers:\s*(\d+)\s+d FF:\s*(\d+)", line)
            if match:
                current["d_layers"] = match.group(1)
                current["d_ff"] = match.group(2)

        elif current and "Factor:" in line:
            match = re.search(r"Factor:\s*(\d+)", line)
            if match:
                current["factor"] = match.group(1)

        metric = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", line)
        if metric and current:
            current["mse"] = metric.group(1)
            current["mae"] = metric.group(2)
            current["dataset_gpu"] = dataset_gpu
            current["dataset_status"] = dataset_status
            current["dataset_seconds"] = dataset_seconds
            rows.append(current)
            current = None

out = log_dir / "summary.csv"
columns = [
    "log_dataset", "data", "model", "model_id", "seq_len", "label_len",
    "pred_len", "mse", "mae", "enc_in", "c_out", "d_model", "d_ff",
    "factor", "dataset_gpu", "dataset_status", "dataset_seconds",
    "data_path", "log_file",
]

with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})

print(f"Saved summary: {out}")
print(f"Parsed finished runs: {len(rows)}")
PY

echo "All logs and summary saved in: ${log_dir}"
