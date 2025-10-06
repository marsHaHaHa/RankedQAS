#!/bin/bash

# Loop through seeds for the Heisenberg experiment
for seed in 0 1 2 3 4;
do
    echo "Running experiment with seed: $seed"
    python VQETrainer_Heisenberg.py --seed $seed --noise True
done

# Loop through seeds for the TFIM experiment
for seed in 0 1 2 3 4;
do
    echo "Running experiment with seed: $seed"
    python VQETrainer_TFIM.py --seed $seed --noise True
done

# Loop through seeds for the (repeated) TFIM experiment
# Note: This block seems to be a duplicate of the previous one.
for seed in 0 1 2 3 4;
do
    echo "Running experiment with seed: $seed"
    python VQETrainer_TFIM.py --seed $seed --noise True
done