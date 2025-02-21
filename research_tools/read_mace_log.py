# warning, LLM generated code

import pandas as pd
import re
import warnings
import glob
import dataclasses
import json 
import os 
import numpy as np


colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]




def parse_last_two_tables_from_log(log_path, return_all=False):
    """
    Parse the last two ASCII error tables from the specified log file.
    
    The function collects blocks of lines bounded by lines that start with '+' or '|'
    (i.e., ASCII table boundary lines or row lines). Each block is treated as one
    table once we hit a line that doesn't start with '+' or '|'. It then parses each
    such table into a pandas DataFrame. Finally, it returns the last two tables found
    in the file (or fewer if the file contains fewer than two).

    This approach handles tables that have multiple lines with +--- (e.g. top, header
    separator, and bottom).
    
    Parameters
    ----------
    log_path : str
        Path to the log file.
    
    Returns
    -------
    (df_second_to_last, df_last) : tuple of (pandas.DataFrame or None, pandas.DataFrame or None)
        DataFrames corresponding to the last two tables found in the file.
        If no tables are found, returns (None, None).
        If exactly one table is found, returns (None, single_table_df).
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    table_blocks = []
    current_block = []
    in_table = False
    
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('+') or stripped.startswith('|'):
            # This line belongs to a table (boundary or data line)
            current_block.append(line)
            in_table = True
        else:
            # We've reached a line that doesn't start with '+' or '|'
            # If we were in a table, finalize the current block
            if in_table:
                table_blocks.append(current_block)
                current_block = []
                in_table = False
            # Otherwise, it's just a line outside any table, so we ignore it
    
    # If the file ends while we are still in a table, finalize the last table
    if in_table and current_block:
        table_blocks.append(current_block)
    
    dataframes = []
    for block in table_blocks:
        # block is a list of lines that belong to one ASCII table
        row_lines = [ln for ln in block if ln.strip().startswith('|')]
        
        if len(row_lines) < 2:
            # Not a complete table or no data
            print(row_lines)
            continue
        
        # The first row_lines entry should be the header
        header_line = row_lines[0]
        header_parts = [col.strip() for col in header_line.strip('| \n').split('|')]
        
        data = []
        for row_line in row_lines[1:]:
            row_parts = [col.strip() for col in row_line.strip('| \n').split('|')]
            if len(row_parts) == len(header_parts):
                data.append(row_parts)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header_parts)
        
        # Attempt to convert numeric columns
        for col in df.columns:
            numeric_count = 0
            for val in df[col]:
                # Simple float pattern
                if re.match(r'^[+-]?(\d+(\.\d+)?|\.\d+)$', val):
                    numeric_count += 1
            if numeric_count >= len(df[col]) / 2:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass
        
        dataframes.append(df)
    
    # Issue warnings if fewer than two tables are found\
    if return_all:
        return dataframes
    
    if len(dataframes) == 0:
        warnings.warn(
            f"No ASCII tables found in log file '{log_path}'. Returning None, None.",
            category=UserWarning
        )
        return None, None
    elif len(dataframes) == 1:
        warnings.warn(
            f"Only one table found in log file '{log_path}'. Returning None for the second table.",
            category=UserWarning
        )
        return None, dataframes[0]
    else:
        # Return the last two tables
        return dataframes[-2], dataframes[-1]


@dataclasses.dataclass
class RunInfo:
    name: str
    seed: int


def parse_path(path: str) -> RunInfo:
    name_re = re.compile(r"(?P<name>.+)_run-(?P<seed>\d+)_train.txt")
    match = name_re.match(os.path.basename(path))
    if not match:
        raise RuntimeError(f"Cannot parse {path}")

    return RunInfo(name=match.group("name"), seed=int(match.group("seed")))



def parse_training_results(path: str):
    run_info = parse_path(path)
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["name"] = run_info.name
            d["seed"] = run_info.seed
            results.append(d)

    return results


def plot_test_in_training(ax, data: pd.DataFrame, min_epoch, test_name, property, **kwargs) -> None:
    data = data[data["epoch"] > min_epoch]
    for_test = data.groupby(["name", "mode", "epoch", "test_name"]).agg([np.mean, np.std]).reset_index()
    test_data = data[data["mode"] == "eval_test"]
    test_data = test_data[test_data["test_name"] == test_name]
    
    ax.plot(
        test_data["epoch"],
        test_data[property],
        **kwargs,
        zorder=1,
        label=test_name,
    )
    ax.set_xlabel("Epoch")
    ax.legend()



def plot_training_run(axes, data: pd.DataFrame, min_epoch, **kwargs) -> None:
    data = data[data["epoch"] > min_epoch]

    if "test_name" in data.columns:
        for_test = data.groupby(["name", "mode", "epoch", "test_name"]).agg([np.mean, np.std]).reset_index()
        data = data.drop(columns="test_name")
        data = data.groupby(["name", "mode", "epoch"]).agg([np.mean, np.std]).reset_index()
    else:
        data = data.groupby(["name", "mode", "epoch"]).agg([np.mean, np.std]).reset_index()

    valid_data = data[data["mode"] == "eval"]
    train_data = data[data["mode"] == "opt"]


    ax = axes[0]
    ax.plot(
        valid_data["epoch"],
        valid_data["loss"]["mean"],
        **kwargs,
        zorder=1,
        label="Validation",
    )
    ax.plot(
        train_data["epoch"],
        train_data["loss"]["mean"],
        **kwargs,
        linestyle='--',
        zorder=1,
        label="Training",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(
        valid_data["epoch"],
        valid_data["mae_e"]["mean"],
        **kwargs,
        zorder=1,
        label="MAE Energy [eV]",
    )
    ax.plot(
        valid_data["epoch"],
        valid_data["mae_f"]["mean"],
        **kwargs,
        linestyle='--',
        zorder=1,
        label="MAE Forces [eV/Å]",
    )
    ax.set_xlabel("Epoch")
    ax.legend()

