#!/bin/bash

# Number of circuit samples to use for calculating the RF proxy.
sample_lines=10000

# Arrays to store the execution time for each task and seed.
declare -a Heisenberg_times
declare -a TFIM_times
declare -a TFCluster_times

# --- Task: Heisenberg_8 ---
echo "--- Starting RF calculation for Heisenberg_8 model ---"
for seed in 0 1 2 3 4; do
    echo "Running script with seed $seed..."
    echo "Sample lines set to $sample_lines"
    start_time=$(date +%s)
    python Relative_fluctuation_calculator_Heisenberg_8.py --seed $seed --sample_lines $sample_lines --qubits 8
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    Heisenberg_times[$seed]=$duration
    echo "Seed $seed completed, time taken: ${duration} seconds."
done
echo "--- All seeds for Heisenberg_8 pre-training proxy calculation finished. ---"
echo "" # Add a blank line for better readability

# --- Task: TFIM_8 ---
echo "--- Starting RF calculation for TFIM_8 model ---"
for seed in 0 1 2 3 4; do
    echo "Running script with seed $seed..."
    echo "Sample lines set to $sample_lines"
    start_time=$(date +%s)
    python Relative_fluctuation_calculator_TFIM_8.py --seed $seed --sample_lines $sample_lines --qubits 8
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    TFIM_times[$seed]=$duration
    echo "Seed $seed completed, time taken: ${duration} seconds."
done
echo "--- All seeds for TFIM_8 pre-training proxy calculation finished. ---"
echo "" # Add a blank line for better readability

# --- Task: TFCluster_8 ---
echo "--- Starting RF calculation for TFCluster_8 model ---"
for seed in 0 1 2 3 4; do
    echo "Running script with seed $seed..."
    echo "Sample lines set to $sample_lines"
    start_time=$(date +%s)
    python Relative_fluctuation_calculator_TFCluster_8.py --seed $seed --sample_lines $sample_lines --qubits 8
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    TFCluster_times[$seed]=$duration
    echo "Seed $seed completed, time taken: ${duration} seconds."
done
echo "--- All seeds for TFCluster_8 pre-training proxy calculation finished. ---"
echo "" # Add a blank line for better readability

# --- Final Summary ---
echo "========================================="
echo "       Execution Time Summary"
echo "========================================="

echo "Heisenberg_8:"
for seed in 0 1 2 3 4; do
    echo "  Seed $seed: ${Heisenberg_times[$seed]} seconds"
done
echo ""

echo "TFIM_8:"
for seed in 0 1 2 3 4; do
    echo "  Seed $seed: ${TFIM_times[$seed]} seconds"
done
echo ""

echo "TFCluster_8:"
for seed in 0 1 2 3 4; do
    echo "  Seed $seed: ${TFCluster_times[$seed]} seconds"
done
echo "========================================="