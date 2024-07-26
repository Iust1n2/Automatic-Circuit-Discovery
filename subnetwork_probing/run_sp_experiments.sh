#!/bin/bash

# Define the variables
TASK="hybrid-retrieval"
LR="0.01"
LAMBDA_REG="150"
SAVE_DIR="results"
LOG_FILE="logs/log_kbicr_lr_${LR}_lambda_${LAMBDA_REG}.txt"

# Run the Python script with the specified parameters and redirect output to log file
python train.py --task $TASK --lr $LR --lambda_reg $LAMBDA_REG --save_dir $SAVE_DIR > $LOG_FILE 2>&1