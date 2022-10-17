import click
import os
import os.path
import re

import pandas as pd


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--instance-dir',
    '-i',
    'instance_dir',
    default=os.getcwd(),
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to directory with the required instance files."
)
@click.option(
    '--result-dir',
    '-r',
    'result_dir',
    default=os.getcwd(),
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to directory with the required result files."
)
@click.option(
    '--output-dir',
    '-o',
    'output_dir',
    default=os.getcwd(),
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to the output directory."
)
def main(instance_dir, result_dir, output_dir):
    """Generate a LaTeX table from some output data.

    Parameters
    ----------
    instance_dir : str
        Path to directory with the required instance files.
    output_dir : str
        Path to the output directory.
    """
    # Find instance data
    flow_data = ''
    pattern = re.compile('[0-9]*_instances_overview.csv$')
    for root, directories, files in os.walk(instance_dir):
        for file in files:
            if pattern.match(file):
                flow_data = os.path.join(root, file)
                break

    pattern2 = re.compile('[0-9_]*_summary.csv$')

    # Get list of MIP summary files
    mip_files = []
    mip_dir = os.path.join(result_dir, 'mip')
    for root, directories, files in os.walk(mip_dir):
        for file in files:
            if pattern2.match(file):
                mip_files.append(os.path.join(root, file))
    
    # Get list of SA summary files
    sa_files = []
    sa_dir = os.path.join(result_dir, 'sa')
    for root, directories, files in os.walk(sa_dir):
        for file in files:
            if pattern2.match(file):
                sa_files.append(os.path.join(root, file))

    # Get list of SA2 summary files
    sa2_files = []
    sa2_dir = os.path.join(result_dir, 'sa2')
    for root, directories, files in os.walk(sa2_dir):
        for file in files:
            if pattern2.match(file):
                sa2_files.append(os.path.join(root, file))

    # Collect instance information
    combined_data = pd.read_csv(
        flow_data,
        sep=';',
        usecols=([
            'n', 'r', 'adversarial', 'serialID', 'feasible'
        ])
    )
    combined_data.rename(columns={'adversarial': 'a', 'serialID': 'i'}, inplace=True)

    # Collect MIP results
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

    # Collect SA data
    sa_res_data = []
    for sa_file in sa_files:
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
    combined_data = pd.merge(combined_data, sa_data, how='left', on=['n', 'r', 'a', 'i'])

    # Collect SA2 data
    sa2_res_data = []
    for sa_file in sa2_files:
        sa2_res_data.append(
            pd.read_csv(
                sa_file,
                sep=';',
                usecols=([
                    'n', 'r', 'a', 'i', 'time', 'best', 'slack'
                ])
            )
        )
    sa2_data = pd.concat(sa2_res_data)
    sa2_data.rename(columns={'time': 'sa2_time', 'best': 'sa2_best', 'slack': 'sa2_slack'}, inplace=True)
    combined_data = pd.merge(combined_data, sa2_data, how='left', on=['n', 'r', 'a', 'i'])

    combined_data.sort_values(by=['n', 'r', 'a', 'i'], inplace=True)

    latex_string = ''

    for idx, row in combined_data.iterrows():
        latex_string += f'{row["n"]:.0f} & {row["r"]:.0f} & {row["a"]:.0f} & {row["i"]:.0f} & '
        if int(row['feasible']) == 1:
            latex_string += '\cellcolor{green}'
        else:
            latex_string += '\cellcolor{red}'
        if pd.isna(row["mip_time"]):
            latex_string += ' & - & -'
        else:
            cellcolor = '\cellcolor{red}'
            if float(row["mip_best"]) < 0:
                cellcolor = '\cellcolor{red!25}'
            else:
                cellcolor = '\cellcolor{green!25}'
            mip_time_str = f'{float(row["mip_time"]):.2f}'
            if float(row["mip_time"]) > 3600:
                mip_time_str = '\\texttt{LIMIT}'
            latex_string += f' & {cellcolor} {mip_time_str} & {cellcolor} {float(row["mip_best"]):.2f}'
        latex_string += f' & {float(row["init_score"]):.2f}'
        cellcolor = '\cellcolor{red}'
        if float(row["sa_slack"]) > 0:
            cellcolor = '\cellcolor{red!25}'
        else:
            cellcolor = '\cellcolor{green!25}'
        latex_string += f' & {cellcolor} {float(row["sa_time"]):.2f} & {cellcolor} {float(row["sa_best"]):.2f}'
        cellcolor = '\cellcolor{red}'
        if float(row["sa2_slack"]) > 0:
            cellcolor = '\cellcolor{red!25}'
        else:
            cellcolor = '\cellcolor{green!25}'
        latex_string += f' & {cellcolor} {float(row["sa2_time"]):.2f} & {cellcolor} {float(row["sa2_best"]):.2f}'
        latex_string += '\\\\\n'

    with open(os.path.join(output_dir, 'latex_table.tex'), 'w') as f:
        f.write(latex_string)


if __name__ == "__main__":
    main()