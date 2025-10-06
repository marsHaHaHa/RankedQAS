"""
Utilities for logging experiment information, including experiment name, timestamp, results, etc.
"""

import os
from typing import Dict, Any, List
import pandas as pd

from utils.common import ensure_dir


def save_metrics_to_excel(
    metrics: dict,
    hyperparameters: dict,
    result_path: str,
    timestamp,
    loss_config=None,
    model_config=None,
    memo: str = "",
    excel_filename: str = "experiment_results.xlsx",
    sheet_name: str = "Results"
):
    """
    Save given metrics and configurations into an Excel file.

    Parameters:
    - metrics (dict): Dictionary of metric names and their values.
    - hyperparameters (dict): Dictionary of general hyperparameters.
    - result_path (str): Path where the results directory will be created.
    - timestamp (str): Timestamp of the experiment run.
    - loss_config (dict, optional): Dictionary of loss function configurations.
    - model_config (dict, optional): Dictionary of model architecture configurations.
    - memo (str, optional): Optional notes to include in the log.
    - excel_filename (str, optional): Name of the Excel file to save the data.
    - sheet_name (str, optional): Name of the Excel sheet.

    Example usage:
    save_metrics_to_excel(
        metrics=metrics,
        hyperparameters=hparams,
        result_path="results/my_task",
        timestamp="2023-10-27_10-30-00",
        memo="Initial test run",
        excel_filename="GraphTransformer_results.xlsx"
    )
    """
    excel_save_path = os.path.join(result_path, "result_excel")
    ensure_dir(excel_save_path)
    excel_file = os.path.join(excel_save_path, excel_filename)

    df_existing = pd.read_excel(excel_file) if os.path.exists(excel_file) else pd.DataFrame()

    # Ensure loss_config, model_config, and hyperparameters are dictionaries, using empty dicts if they are None.
    loss_config = loss_config or {}
    model_config = model_config or {}
    hyperparameters = hyperparameters or {}

    metrics_with_info = {
        **hyperparameters,
        **model_config,
        **loss_config,
        **metrics,
        "time": timestamp,
        "memo": memo,
    }

    df_new = pd.DataFrame([metrics_with_info])
    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Experiment results have been saved to {excel_file}")


def log_to_excel(
        filepath: str,
        run_data: Dict[str, Any],
        column_order: List[str] = None
):
    """
    Appends data from a single experiment run (config + metrics) to a specified Excel file.
    If the file or sheet does not exist, it will be created.

    Args:
        filepath (str): The full path to the Excel file (e.g., 'results/my_experiments.xlsx').
        run_data (Dict[str, Any]): A single dictionary containing all information for the current run.
                                   Example: {'lr': 0.01, 'seed': 0, 'loss': 0.5, 'spearman': 0.8}.
        column_order (List[str], optional):
            Specifies the desired column order in the final Excel file for better organization.
            If None, the default dictionary order is used.
    """
    # 1. Ensure the directory exists.
    ensure_dir(os.path.dirname(filepath))

    # 2. Convert the current run's data into a DataFrame.
    # Using [run_data] makes it a single-row DataFrame.
    new_df = pd.DataFrame([run_data])

    # 3. Read existing data; create an empty DataFrame if the file doesn't exist.
    try:
        existing_df = pd.read_excel(filepath)
    except FileNotFoundError:
        existing_df = pd.DataFrame()

    # 4. Combine the old and new data.
    # ignore_index=True resets the index to maintain continuity.
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # 5. [CRITICAL] Unify column order.
    if column_order:
        # Get all current columns to handle cases where new data has columns not in column_order.
        all_columns = existing_df.columns.union(new_df.columns).tolist()
        # Create an ordered list of columns based on the provided template.
        ordered_columns = [col for col in column_order if col in all_columns]
        # Append any new columns not in the template to the end to prevent data loss.
        remaining_columns = [col for col in all_columns if col not in ordered_columns]
        final_columns = ordered_columns + remaining_columns

        # Reindex the DataFrame; non-existent columns will be filled with NaN.
        combined_df = combined_df.reindex(columns=final_columns)

    # 6. Write to the file.
    try:
        # Use ExcelWriter for better control over the writing process.
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            combined_df.to_excel(writer, index=False, sheet_name='Results')
        print(f"Results successfully logged to: {filepath}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary. If the value is a list, take the last element.

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Prefix for the keys.
        sep (str): Separator between parent and child keys.

    Returns:
        dict: Flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = v[-1] if v else None
        else:
            items[new_key] = v
    return items


# Example usage
if __name__ == "__main__":
    import random
    import time

    result_path = "testlog"
    # Example hyperparameters
    hyperparameters = {
        "Seed": random.randint(0, 10000),
        "lr": round(random.uniform(1e-4, 1e-2), 6),
        "batch_size": random.choice([16, 32, 64]),
    }
    # Example model config
    model_config = {
        "transformer_layers": random.randint(2, 6),
        "num_heads": random.choice([2, 4, 8]),
        "mlp_hidden_dim": random.choice([64, 128, 256]),
        "dropout": round(random.uniform(0.1, 0.5), 2),
    }
    # Example loss config
    loss_config = {
        "temperature": round(random.uniform(0.5, 2.0), 2),
        "k": random.choice([5, 10, 20]),
        "metric_function": "ndcg",
    }
    # Example metrics
    metrics = {
        "top_1_acc": round(random.uniform(0.6, 1.0), 4),
        "top_5_acc": round(random.uniform(0.6, 1.0), 4),
        "top_10_acc": round(random.uniform(0.6, 1.0), 4),
        "Spearman": round(random.uniform(0.6, 1.0), 4),
        "NDCG@10": round(random.uniform(0.6, 1.0), 4),
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    save_metrics_to_excel(
        metrics=metrics,
        hyperparameters=hyperparameters,
        model_config=model_config,
        loss_config=loss_config,
        result_path=result_path,
        timestamp=timestamp,
        memo="GraphTransformer+NDCG_Test",
        excel_filename="GraphTransformer_ndcg_experiment_results.xlsx"
    )

    print("Experiment results have been saved to the Excel file.")