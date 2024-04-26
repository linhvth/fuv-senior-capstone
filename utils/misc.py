"""
Helper functions
"""

import os
import datetime
import numpy as np
import pandas as pd

def write_as_txt(results: dict, save_result_path: str) -> None:
    action = 'a' if os.path.exists(save_result_path) else 'w'
    with open(save_result_path, action) as f:
        f.write('\n----------------')
        f.write('\n' + get_current_time_string() + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def write_as_csv(result: pd.DataFrame, save_result_path: str) -> None:
    result.to_csv(save_result_path, index=False)
    return

def check_path_file(path):
    # Create the folder if it doesn't exist
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            raise OSError(f"Error creating folder '{path}': {e}")
        
def get_current_time_string():
    """
    Gets the current time in YYYY-MM-DD_HH-MM-SS format as a string.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("Date: %Y-%m-%d\nTime: %H-%M-%S")
    return timestamp