# warning, LLM generated code

import pandas as pd
import re
import warnings

def parse_last_two_tables_from_log(log_path):
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
    
    # Issue warnings if fewer than two tables are found
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