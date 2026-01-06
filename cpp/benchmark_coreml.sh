#!/bin/bash

# benchmark_coreml.sh - Systematic benchmark script for CoreML backend optimization
#
# This script runs benchmarks across various thread counts and batch sizes
# to find optimal hyperparameters for the CoreML hybrid backend.
#
# Usage: ./benchmark_coreml.sh -model <model.bin.gz> [-config <config.cfg>] [-visits <N>] [-output <dir>]
#
# For detailed per-batch logging, rebuild with:
#   cmake -G Ninja -DUSE_BACKEND=COREML -DVERBOSE_COREML=1
#   ninja

set -e

# Default values
MODEL=""
CONFIG="configs/gtp_example.cfg"
VISITS=800
OUTPUT_DIR="benchmark_results"
THREADS="1,2,4,6,8,12,16,24,32"
BATCH_MODES="default half"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -model)
            MODEL="$2"
            shift 2
            ;;
        -config)
            CONFIG="$2"
            shift 2
            ;;
        -visits)
            VISITS="$2"
            shift 2
            ;;
        -output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -threads)
            THREADS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -model <model.bin.gz> [-config <config.cfg>] [-visits <N>] [-output <dir>] [-threads <list>]"
            echo ""
            echo "Options:"
            echo "  -model    Path to neural network model file (required)"
            echo "  -config   Path to config file (default: configs/gtp_example.cfg)"
            echo "  -visits   Number of visits per position (default: 800)"
            echo "  -output   Output directory for results (default: benchmark_results)"
            echo "  -threads  Comma-separated list of thread counts to test (default: 1,2,4,6,8,12,16,24,32)"
            echo ""
            echo "Example:"
            echo "  $0 -model ~/models/kata1-b28c512nbt.bin.gz -visits 800 -threads \"8,16,24,32\""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
    echo "Error: -model is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

if [[ ! -f ./katago ]]; then
    echo "Error: katago executable not found in current directory"
    echo "Please run this script from the cpp build directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.csv"
SUMMARY_FILE="$OUTPUT_DIR/summary_${TIMESTAMP}.txt"

echo "CoreML Backend Benchmark"
echo "========================"
echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "Visits: $VISITS"
echo "Threads: $THREADS"
echo "Output: $OUTPUT_DIR"
echo ""

# Write CSV header
echo "threads,batch_mode,visits_per_sec,nnevals_per_sec,avg_batch_size,nn_batches" > "$RESULTS_FILE"

# Convert thread list to array
IFS=',' read -ra THREAD_ARRAY <<< "$THREADS"

# Run benchmarks
for threads in "${THREAD_ARRAY[@]}"; do
    for batch_mode in $BATCH_MODES; do
        echo "Testing: threads=$threads, batch_mode=$batch_mode"

        BATCH_ARG=""
        if [[ "$batch_mode" == "half" ]]; then
            BATCH_ARG="--half-batch-size"
        fi

        # Run benchmark and capture output
        OUTPUT=$(./katago benchmark \
            -model "$MODEL" \
            -config "$CONFIG" \
            -threads "$threads" \
            -visits "$VISITS" \
            $BATCH_ARG 2>&1 || true)

        # Save full output to log file
        LOG_FILE="$OUTPUT_DIR/t${threads}_${batch_mode}_${TIMESTAMP}.log"
        echo "$OUTPUT" > "$LOG_FILE"

        # Extract metrics from output
        # Format: "threads visits/s nnevals/s nnbatches/s avg_batch_size"
        VISITS_PER_SEC=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+ threads:" | tail -1 | awk '{print $3}')
        NNEVALS_PER_SEC=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+ threads:" | tail -1 | awk '{print $5}')
        AVG_BATCH=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+ threads:" | tail -1 | awk '{print $9}')
        NN_BATCHES=$(echo "$OUTPUT" | grep -E "^\s*[0-9]+ threads:" | tail -1 | awk '{print $7}')

        # Write to CSV
        echo "$threads,$batch_mode,$VISITS_PER_SEC,$NNEVALS_PER_SEC,$AVG_BATCH,$NN_BATCHES" >> "$RESULTS_FILE"

        echo "  visits/s=$VISITS_PER_SEC nnevals/s=$NNEVALS_PER_SEC avg_batch=$AVG_BATCH"
    done
done

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Log files saved to: $OUTPUT_DIR/"

# Generate summary
{
    echo "CoreML Backend Benchmark Summary"
    echo "================================"
    echo "Date: $(date)"
    echo "Model: $MODEL"
    echo "Visits: $VISITS"
    echo ""
    echo "Results (sorted by nnevals/s):"
    echo ""
    # Sort by nnevals/s (column 4) descending, skip header
    tail -n +2 "$RESULTS_FILE" | sort -t',' -k4 -rn | head -20
} > "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
