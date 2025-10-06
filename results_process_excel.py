import os
import pandas as pd
import glob
from tqdm import tqdm

# --- Configuration ---
# Main results directory, consistent with your shell script outputs
RESULTS_ROOT_DIR = "finetune_results"
# Output filename for the final summary
OUTPUT_FILE = "experiment_summary_score_normalization.xlsx"
# The expected number of random seeds per experiment group
EXPECTED_SEEDS = 5
# The list of tasks to be included in the summary
TARGET_TASKS = ["Heisenberg_8", "TFIM_8", "TFCluster_8"]

def get_strategy_name(row):
    """Returns the strategy name based on the boolean values in the configuration row."""
    if not row['use_pretrained']:
        return "From Scratch"
    elif row['two_stage_finetune']:
        return "2-Stage Finetune"
    # This can be extended if other strategies are added in the future
    else:
        # This case corresponds to single-stage full or frozen fine-tuning,
        # which might not be covered by the current shell scripts.
        return "Single-Stage Finetune"

def get_task_from_path(file_path, tasks_list):
    """Parses the task name from the file path."""
    for task in tasks_list:
        # Use os.sep to ensure cross-platform compatibility (handles both \ and /)
        if f"{os.sep}{task}{os.sep}" in file_path:
            return task
    return "Unknown"  # Return "Unknown" if no task is found

def main():
    """
    Main function to find, process, and summarize all experiment results.
    """
    print("--- Starting to summarize experiment results ---")

    # Construct the search path
    search_pattern = os.path.join(RESULTS_ROOT_DIR, "**", "summary.xlsx")
    excel_files = glob.glob(search_pattern, recursive=True)

    if not excel_files:
        print(f"Error: No 'summary.xlsx' files found in the '{RESULTS_ROOT_DIR}' directory.")
        print("Please ensure this script is in the same directory as the results folder and that experiments have been run successfully.")
        return

    print(f"Found {len(excel_files)} experiment result files. Starting processing...")

    all_summaries = []

    for file_path in tqdm(excel_files, desc="Processing Excel files"):
        try:
            # Filter out tasks we are not interested in
            task_name = get_task_from_path(file_path, TARGET_TASKS)
            if task_name == "Unknown" or task_name not in TARGET_TASKS:
                continue

            df = pd.read_excel(file_path)

            # We are only interested in the validation set ('val') results
            val_df = df[df['dataset_split'] == 'val'].copy()

            if val_df.empty:
                tqdm.write(f"Warning: No validation set ('val') data in file {file_path}. Skipping.")
                continue

            # Check if the number of seeds matches the expectation
            if len(val_df) != EXPECTED_SEEDS:
                tqdm.write(f"Warning: The number of validation rows ({len(val_df)}) in {file_path} "
                           f"does not match the expected number of seeds ({EXPECTED_SEEDS}). Will still calculate the average.")

            # --- 1. Extract configuration information ---
            # The configuration is the same for all rows, so we can get it from the first one
            config_row = val_df.iloc[0]

            summary_data = {
                'task': task_name,
                'strategy': get_strategy_name(config_row),
                'pt_samples': config_row.get('pretrain_samples', 'N/A'),
                'bs_pretrain': config_row.get('bs_pretrain', 'N/A'),
                'pretrain_label_scaling': config_row.get('pretrain_label_scaling', 'N/A'),
                'ft_samples': config_row.get('downstream_train_samples', 'N/A'),
                'ft_batch_size': config_row['batch_size'],
                'downstream_loss': config_row['downstream_loss']
            }

            # --- 2. Calculate the average of metrics ---
            metric_cols = ['spearman', 'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20',
                           'ndcg@50', 'ndcg@100',
                           'top1', 'top5', 'top10', 'top20', 'top50', 'top100',
                           'loss']

            # Ensure all metric columns exist, excluding non-existent ones before calculation
            existing_metric_cols = [col for col in metric_cols if col in val_df.columns]
            avg_metrics = val_df[existing_metric_cols].mean()

            # --- 3. Merge the data ---
            for metric, avg_value in avg_metrics.items():
                summary_data[f"{metric}_avg"] = avg_value

            all_summaries.append(summary_data)

        except Exception as e:
            tqdm.write(f"Error: An exception occurred while processing file {file_path}: {e}")

    if not all_summaries:
        print("No data was successfully processed. Cannot generate a summary.")
        return

    # --- 4. Create and organize the final DataFrame ---
    final_df = pd.DataFrame(all_summaries)

    # Define the final column order for better readability
    final_column_order = [
        'task',
        'strategy',
        'pt_samples',
        'bs_pretrain',
        'pretrain_label_scaling',
        'ft_samples',
        'ft_batch_size',
        'downstream_loss',
        'spearman_avg',
        'ndcg@1_avg',
        'ndcg@5_avg',
        'ndcg@10_avg',
        'ndcg@20_avg',
        'ndcg@50_avg',
        'ndcg@100_avg',
        'top1_avg',
        'top5_avg',
        'top10_avg',
        'loss_avg'
    ]

    # Filter out columns that may not exist
    final_column_order = [col for col in final_column_order if col in final_df.columns]
    final_df = final_df[final_column_order]

    # Sort the results for clarity
    final_df = final_df.sort_values(
        by=['task', 'strategy', 'pt_samples', 'bs_pretrain', 'ft_samples', 'ft_batch_size', 'downstream_loss'],
        ascending=[True, False, True, True, True, True, True]
    ).reset_index(drop=True)

    # --- 5. Save to Excel ---
    try:
        final_df.to_excel(OUTPUT_FILE, index=False, float_format="%.4f")
        print("\n" + "=" * 60)
        print("           Experiment Results Summary (Average of 5 Seeds)")
        print("=" * 60)
        print(final_df.to_string())
        print("\n" + "=" * 60)
        print(f"\nAggregated results have been successfully saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError: Failed to save to Excel file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()