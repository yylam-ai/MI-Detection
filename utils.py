import re
import os
import numpy as np
import pandas as pd
from tabulate import tabulate


def extract_metrics(file_path):
    metrics = {
        "Sensitivity": [],
        "Specificity": [],
        "Precision": [],
        "F1-Score": [],
        "F2-Score": [],
        "Accuracy": []
    }
    
    with open(file_path, 'r') as file:
        content = file.read()
    
    for metric in metrics.keys():
        pattern = rf"{metric}:(\d+\.?\d*)"
        values = [float(match) for match in re.findall(pattern, content)]
        metrics[metric] = values
    
    averages = {metric: np.mean(values) for metric, values in metrics.items()}
    df = pd.DataFrame([averages])
    model_view = os.path.splitext(os.path.basename(file_path))[0]
    model, view = model_view.split('_')
    df["Model"] = model
    df["View"] = view

    # Reorder columns to make "Model" and "View" the first two
    column_order = ["Model", "View"] + [col for col in df.columns if col not in ["Model", "View"]]
    df = df[column_order]
    
    return df

# calculate average metric score across folds
if __name__ == "__main__":
    dir_path = "output"
    file_paths = []
    for f_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f_name)
        file_extension = file_path.split('.')[-1]
        if file_extension == "txt":
            file_paths.append(file_path)

    all_results = pd.concat([extract_metrics(fp) for fp in file_paths], ignore_index=True)

    print(f"{'='*20} Average peformance results computed from 5-fold {'='*20}")
    print(tabulate(all_results, headers='keys', tablefmt='grid'))
