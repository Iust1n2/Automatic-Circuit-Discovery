#!/bin/bash

# Define the paths to the metrics.json files
METRICS_FILES=(
    "hybridretrieval/acdc_results/kbicr_indirect_kl_div_0.1/logs/metrics.json"
    "hybridretrieval/acdc_results/kbicr_indirect_kl_div_0.05/logs/metrics.json"
    "hybridretrieval/acdc_results/kbicr_indirect_kl_div_0.15/logs/metrics.json"
)

# Define the output directory for the ROC curve plots
OUTPUT_DIR="roc_plots"

# Ensure the output directory exists
mkdir -p $OUTPUT_DIR

# Iterate over each file and generate the ROC curve
for METRICS_FILE in "${METRICS_FILES[@]}"; do
    if [ -f "$METRICS_FILE" ]; then
        # Run the ROC script and save the plot
        python roc.py --input-file "$METRICS_FILE" --output-dir "$OUTPUT_DIR"
    else
        echo "metrics.json not found in $METRICS_FILE"
    fi
done

echo "All ROC curves generated."