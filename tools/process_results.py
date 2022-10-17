import click
import os
import os.path
import re

import pandas as pd
import numpy as np


def summary_files(basepath, subdir):
    """Returns a list of all summary files from the subdirectory."""
    summary_files = []
    summary_dir = os.path.join(basepath, subdir)
    pattern = re.compile('[0-9_]*_summary.csv$')
    for root, directories, files in os.walk(summary_dir):
        for file in files:
            if pattern.match(file):
                summary_files.append(os.path.join(root, file))
    return summary_files


def merge_sa_summary(summary_files):
    """Merge the results from a set of summary files into a single table."""
    sa_res_data = []
    for sa_file in summary_files:
        sa_res_data.append(
            pd.read_csv(
                sa_file,
                sep=';',
                usecols=([
                    'n', 'r', 'a', 'i', 'init_score', 'time', 'best', 'slack'
                ])
            )
        )
    sa_data = pd.concat(sa_res_data)
    sa_data.rename(columns={'time': 'sa_time', 'best': 'sa_best', 'slack': 'sa_slack'}, inplace=True)
    return sa_data


def collect_results():
    topleveldir = os.path.join(os.getcwd(), 'results', '20220928allresults')
    param_dirs = [
        '20220711sa_original',
        '20220825sa_cutoff',
        '20220826sa_cutoff',
        '20220828sa_cutoff',
        '20220829sa_cutoff',
        '20220830sa_cutoff',
        '20220901sa_cutoff',
        '20220902sa_cutoff',
        '20220903sa_cutoff',
        '20220904sa_cutoff',
        '20220905sa_cutoff'
    ]
    best_dir = param_dirs[5]
    instance_file = os.path.join(os.getcwd(), 'data', '20220607finaltestset', '20220607_instances_overview.csv')

    # Collect instance information
    combined_data = pd.read_csv(
        instance_file,
        sep=';',
        usecols=([
            'n', 'r', 'adversarial', 'serialID', 'feasible'
        ])
    )
    combined_data.rename(columns={'adversarial': 'a', 'serialID': 'i'}, inplace=True)
    flow = combined_data.copy()
    flow.sort_values(by=['n', 'r', 'a', 'i'], inplace=True)
    
    # Collect results from the MIP
    mip_files = summary_files(topleveldir, 'mip')
    mip_res_data = []
    for mip_file in mip_files:
        mip_res_data.append(
            pd.read_csv(
                mip_file,
                sep=';',
                usecols=([
                    'n', 'r', 'a', 'i', 'time', 'objective'
                ])
            )
        )
    mip_data = pd.concat(mip_res_data)
    mip_data.rename(columns={'time': 'mip_time', 'objective': 'mip_best'}, inplace=True)
    combined_data = pd.merge(combined_data, mip_data, how='left', on=['n', 'r', 'a', 'i'])
    
    # Get results from the chosen parameter setting
    sa_files = summary_files(topleveldir, best_dir)
    sa_data = merge_sa_summary(sa_files)
    combined_data = pd.merge(combined_data, sa_data, how='left', on=['n', 'r', 'a', 'i'])
    
    # Set best and find if a better score exists.
    # Initial best is MIP score
    combined_data['best_score'] = combined_data.loc[:, 'mip_best']
    combined_data['best_feasible'] = True
    combined_data = combined_data.fillna(-1)
    combined_data.sort_values(by=['n', 'r', 'a', 'i'], inplace=True)
    for param_dir in param_dirs:
        # Collect summary files from sub dir
        summ_files = summary_files(topleveldir, param_dir)
        
        # Put results in a single table (selecting for n)
        result_table = merge_sa_summary(summ_files)
        result_table.sort_values(by=['n', 'r', 'a', 'i'], inplace=True)
        # print(f'{flow.dtypes["r"]} <> {result_table.dtypes["r"]}')
        result_table = pd.merge(flow, result_table, how='left', on=['n', 'r', 'a', 'i'])
        result_table = result_table.fillna(-1)
        
        if combined_data.shape[0] != result_table.shape[0]:
            print(f'{combined_data.shape[0]} != {result_table.shape[0]}')
        
        for idx in range(combined_data.shape[0]):
            if combined_data.iloc[idx, combined_data.columns.get_loc('best_score')] > result_table.iloc[idx, result_table.columns.get_loc('sa_best')] or combined_data.iloc[idx, combined_data.columns.get_loc('best_score')] < 0:
                combined_data.iat[idx, combined_data.columns.get_loc('best_score')] = result_table.iloc[idx, result_table.columns.get_loc('sa_best')]
                combined_data.iat[idx, combined_data.columns.get_loc('best_feasible')] = (result_table.iloc[idx, result_table.columns.get_loc('sa_slack')] <= 0)
    
    results = {
        "5" : [0, 0, 0, 0, 0],
        "10": [0, 0, 0, 0, 0],
        "15": [0, 0, 0, 0, 0]
    }
    for idx, row in combined_data.iterrows():
        if int(row["n"]) > 15:
            continue
        if float(row["mip_best"]) < 0 or float(row["sa_slack"]) > 0:
            continue
        results[str(row["n"])][0] += 1
        results[str(row["n"])][1] += (float(row["mip_best"]) - float(row["best_score"])) / float(row["best_score"])
        results[str(row["n"])][2] += float(row["mip_time"])
        results[str(row["n"])][3] += (float(row["sa_best"]) - float(row["best_score"])) / float(row["best_score"])
        results[str(row["n"])][4] += float(row["sa_time"])
    
    latex_string = ''
    for key in ["5", "10", "15"]:
        latex_string += f'{key} & {results[key][2] / results[key][0]:.2f} & {results[key][1] / results[key][0]:.2f} & {results[key][4] / results[key][0]:.2f} & {results[key][3] / results[key][0]:.2f} \\\\\n'
    with open(os.path.join(topleveldir, 'latex_table2.tex'), 'w') as f:
        f.write(latex_string)
    
    return
    
    latex_string = ''

    for idx, row in combined_data.iterrows():
        # Labels
        latex_string += f'{row["n"]:.0f} & {row["r"]:.0f} & {row["a"]:.0f} & {row["i"]:.0f} & '
        
        # Flow
        if int(row['feasible']) == 1:
            latex_string += 'yes & '
        else:
            latex_string += '\\textit{no} & '

        # Best
        if not row["best_feasible"]:
            latex_string += '\\textit{'
        latex_string += f'{float(row["best_score"]):.2f}'
        if not row["best_feasible"]:
            latex_string += '}'
        
        # MIP
        if pd.isna(row["mip_time"]) or row["mip_time"] < 0:
            latex_string += ' & - & -'
        else:
            if float(row["mip_time"]) > 3600:
                latex_string += ' & \\texttt{LIMIT} & '
            else:
                latex_string += f' & {float(row["mip_time"]):.2f} & '
            if float(row["mip_best"]) < 0:
                latex_string += '\\textit{'
            if np.isclose(float(row["mip_best"]), float(row["best_score"]), atol=0.001):
                latex_string += '\\textbf{'
            latex_string += f'{float(row["mip_best"]):.2f}'
            if np.isclose(float(row["mip_best"]), float(row["best_score"]), atol=0.001):
                latex_string += '}'
            if float(row["mip_best"]) < 0:
                latex_string += '}'
        
        # SA
        latex_string += f' & {float(row["sa_time"]):.2f} & '
        if float(row["sa_slack"]) > 0:
            latex_string += '\\textit{'
        if np.isclose(float(row["sa_best"]), float(row["best_score"]), atol=0.001):
            latex_string += '\\textbf{'
        latex_string += f'{float(row["sa_best"]):.2f}'
        if float(row["sa_slack"]) > 0:
            latex_string += '}'
        if np.isclose(float(row["sa_best"]), float(row["best_score"]), atol=0.001):
            latex_string += '}'
        latex_string += f' & {float(row["init_score"]):.2f}'
        latex_string += '\\\\\n'

    with open(os.path.join(topleveldir, 'latex_table.tex'), 'w') as f:
        f.write(latex_string)

if __name__ == "__main__":
    collect_results()