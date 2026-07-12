#!/bin/bash
# SATS Ground Operations Dashboard — launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  SATS Ground Operations Dashboard"
echo "============================================"

if [ ! -f "data/flights.parquet" ] || [ ! -f "data/model.pkl" ]; then
    echo "[INFO] Processed data not found. Running data preparation pipeline..."
    echo "[INFO] This will take a few minutes on first run."
    python prepare_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Data preparation failed. Please check prepare_data.py output above."
        exit 1
    fi
    echo "[INFO] Data preparation complete!"
else
    echo "[INFO] Processed data found. Skipping preparation."
fi

echo "[INFO] Launching Streamlit dashboard..."
streamlit run Home.py
