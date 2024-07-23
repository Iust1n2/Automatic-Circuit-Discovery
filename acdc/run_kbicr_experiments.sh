#!/bin/bash

# Define the common parameters
TASK="hybrid-retrieval"
ZERO_ABLATION="--zero-ablation"
METRIC="kl_div"
INDICES_MODE="reverse"
FIRST_CACHE_CPU="False"
SECOND_CACHE_CPU="False"
MAX_NUM_EPOCHS=100000
LOCAL_DIR="hybridretrieval/acdc_results"
DATASET="kbicr_indirect"

# Multiple thresholds to test
THRESHOLDS=(0.05 0.1)

# Iterate over each threshold value
for i in "${!THRESHOLDS[@]}"; do
  THRESHOLD=${THRESHOLDS[$i]}

  # Define the log file path
  LOG_FILE="${LOCAL_DIR}/log${THRESHOLD}.txt"
  
  # Run the experiment
  python main.py --task $TASK $ZERO_ABLATION --metric $METRIC --threshold $THRESHOLD --indices-mode $INDICES_MODE --first-cache-cpu $FIRST_CACHE_CPU --second-cache-cpu $SECOND_CACHE_CPU --max-num-epochs $MAX_NUM_EPOCHS --local-dir $LOCAL_DIR --dataset $DATASET > $LOG_FILE 2>&1 &
  
  echo "Started experiment with threshold $THRESHOLD, logging to $LOG_FILE"
done

echo "All experiments started."