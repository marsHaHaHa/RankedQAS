#!/bin/bash

# --- 1. Core Configuration ---
PYTHON_SCRIPT="main.py"
TASKS=("Heisenberg_8" "TFIM_8" "TFCluster_8")
SEEDS=(0 1 2 3 4)

# --- 2. Pre-training Hyperparameters ---
# Note: These parameters are used for pre-training and for automatically locating
# the model path during the fine-tuning phase.
PRETRAIN_SAMPLES=10000
PRETRAIN_LR="1e-4"
PRETRAIN_LOSS="huber"
EPOCHS_PRETRAIN=20
#PRETRAIN_BS_LIST=(128 64 32)
PRETRAIN_BS_LIST=(32)

# Define the list of pre-training label normalization methods to be tested.
PRETRAIN_LABEL_SCALING_METHODS=("quantile") # Quantile normalization was ultimately chosen.

# --- 3. Downstream Fine-tuning Common Parameters ---
FINETUNE_SAMPLES_LIST=(200)
FINETUNE_LOSS_LIST=("SoftNDCG" "mse")
FINETUNE_BS_LIST=(32)
NUM_VAL_SAMPLES=300

# --- 4. Strategy Hyperparameters ---
# (A) Two-stage Fine-tuning
TWO_STAGE_EPOCHS_HEAD=200
TWO_STAGE_EPOCHS_FULL=200
TWO_STAGE_LR_HEAD="1e-3"
TWO_STAGE_LR_ENCODER="1e-4" # The learning rate for the encoder should be very small.

# (B) Training from Scratch (Baseline)
SCRATCH_EPOCHS=400
SCRATCH_LR="1e-3"

# --- 5. Model Architecture and Regularization Parameters ---
MODEL_NAME="DAGTransformer"
NUM_LAYERS=3
NUM_HEADS=1
HIDDEN_DIM=64
INPUT_DIM=13
DROPOUT=0.0
DROPOUT_MLP=0.2
WEIGHT_DECAY="1e-5"

MODEL_ARCHITECTURE_ARGS="\
    --model_name ${MODEL_NAME} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --hidden_dim ${HIDDEN_DIM} \
    --input_dim ${INPUT_DIM} \
    --dropout ${DROPOUT}"

# Function: Print a formatted header
print_header() {
    echo ""
    echo -e "\n\n################################################################"
    echo -e "######    $1"
    echo -e "################################################################\n"
}


# --- Start Executing Experiments ---
for task in "${TASKS[@]}"; do

  # [Structural Adjustment] The outer loop iterates through all label normalization methods.
  for scaling_method in "${PRETRAIN_LABEL_SCALING_METHODS[@]}"; do

    # ==============================================================================
    #  Phase 1: Pre-training (Task: ${task}, Label Scaling: ${scaling_method})
    # ==============================================================================
    print_header "Phase 1: Pre-training | Task: [${task}] | Label Scaling: [${scaling_method}]"

    for pt_bs in "${PRETRAIN_BS_LIST[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "======> Pre-training | Task: ${task} | PT BS: ${pt_bs} | Seed: ${seed} | Label Scaling: ${scaling_method}"

            python ${PYTHON_SCRIPT} \
                --mode pretrain \
                --task_name ${task} \
                --seed ${seed} \
                ${MODEL_ARCHITECTURE_ARGS} \
                --sample_lines ${PRETRAIN_SAMPLES} \
                --epochs_pretrain ${EPOCHS_PRETRAIN} \
                --pretrain_bs ${pt_bs} \
                --lr_pretrain ${PRETRAIN_LR} \
                --loss_func_pretrain ${PRETRAIN_LOSS} \
                --pretrain_label_scaling ${scaling_method} # Pass the label scaling parameter

            if [ $? -ne 0 ]; then
                echo "Error: Pre-training failed (Task: ${task}, PT BS: ${pt_bs}, Seed: ${seed}, Label Scaling: ${scaling_method}). Aborting script."
                exit 1
            fi
        done
    done


    # ==============================================================================
    #  Phase 2 (A): Downstream Tasks Depending on Pre-training (Task: ${task}, Label Scaling: ${scaling_method})
    # ==============================================================================
    print_header "Phase 2(A): Two-stage Fine-tuning | Task: [${task}] | Loading model trained with [${scaling_method}]"

    # --- Strategy A: Two-stage Fine-tuning (Load the just pre-trained model) ---
    for pt_bs in "${PRETRAIN_BS_LIST[@]}"; do
        PRETRAIN_LOAD_ARGS="\
            --sample_lines ${PRETRAIN_SAMPLES} \
            --pretrain_bs ${pt_bs} \
            --lr_pretrain ${PRETRAIN_LR} \
            --loss_func_pretrain ${PRETRAIN_LOSS}"

        for ft_bs in "${FINETUNE_BS_LIST[@]}"; do
            FINETUNE_COMMON_ARGS="\
                --num_val_samples ${NUM_VAL_SAMPLES} \
                --batch_size ${ft_bs} \
                --dropout_mlp ${DROPOUT_MLP} \
                --weight_decay ${WEIGHT_DECAY}"

            for ft_samples in "${FINETUNE_SAMPLES_LIST[@]}"; do
                for ft_loss in "${FINETUNE_LOSS_LIST[@]}"; do
                    print_header "[Strategy A] Task: ${task} | Two-stage Fine-tuning | PT BS: ${pt_bs}, FT BS: ${ft_bs} | FT Samples: ${ft_samples} | Loss: ${ft_loss} | PT Label Scaling: ${scaling_method}"
                    for seed in "${SEEDS[@]}"; do
                        echo "======> Fine-tuning (Two-stage) | Seed: ${seed}"

                        python ${PYTHON_SCRIPT} \
                            --mode finetune \
                            --task_name ${task} \
                            --seed ${seed} \
                            ${MODEL_ARCHITECTURE_ARGS} \
                            ${FINETUNE_COMMON_ARGS} \
                            ${PRETRAIN_LOAD_ARGS} \
                            --use_pretrained True \
                            --two_stage_finetune True \
                            \
                            --num_train_samples ${ft_samples} \
                            --loss_func ${ft_loss} \
                            \
                            --lr ${TWO_STAGE_LR_HEAD} \
                            --lr_finetune_head ${TWO_STAGE_LR_HEAD} \
                            --lr_finetune_encoder ${TWO_STAGE_LR_ENCODER} \
                            --epochs_head_only ${TWO_STAGE_EPOCHS_HEAD} \
                            --epochs_full_finetune ${TWO_STAGE_EPOCHS_FULL} \
                            --pretrain_label_scaling ${scaling_method} # Pass the label scaling parameter

                        if [ $? -ne 0 ]; then
                            echo "Warning: Fine-tuning task failed (Strategy A, Task: ${task}, Seed: ${seed}, PT Label Scaling: ${scaling_method}). Continuing to the next task."
                        fi
                    done
                done
            done
        done
    done
  done # --- End of loop for label normalization methods ---

done # --- End of the outermost task loop ---

print_header "All experimental suites have been completed!"

echo "======> Starting to summarize experiment results"
python results_process_excel.py
echo "======> Experiment results summarization complete"