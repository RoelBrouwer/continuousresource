import click
import os
import os.path

import pandas as pd


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    '--path',
    '-p',
    'path',
    default=os.getcwd(),
    required=False,
    type=click.Path(
        exists=True,
        resolve_path=True
    ),
    help="Path to directory with the required input files."
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
def main(path, output_dir):
    """Generate a LaTeX table from some output data.

    Parameters
    ----------
    path : str
        Path to directory with the required input files.
    output_dir : str
        Path to the output directory.
    """
    flow_data = '20220425_instances_overview.csv'
    mip_data = ''
    onlyswap_data = ('onlyswap.csv', 'SP')
    onlymovesingle_data = ('onlymovesingle.csv', 'MS')
    onlymovepair_data = ('onlymovepair.csv', 'MP')
    sa_variants = [onlyswap_data, onlymovesingle_data, onlymovepair_data]

    # Collect instance information
    combined_data = pd.read_csv(
        os.path.join(path, flow_data),
        sep=';',
        usecols=([
            'n', 'r', 'adversarial', 'serialID', 'feasible'
        ])
    )
    combined_data.rename(columns={'adversarial': 'a', 'serialID': 'i'}, inplace=True)

    # Collect MIP results
    if mip_data != '':
        mip_res_data = pd.read_csv(
            os.path.join(path, mip_data),
            sep=';',
            usecols=([
                'n', 'r', 'a', 'i', 'time', 'objective'
            ])
        )
        mip_res_data.rename(columns={'time': 'mip_time', 'objective': 'mip_best'}, inplace=True)
        combined_data = pd.merge(combined_data, mip_res_data, on=['n', 'r', 'a', 'i'])

    # Collect SA data
    for file in sa_variants:
        data = pd.read_csv(
            os.path.join(path, file[0]),
            sep=';',
            usecols=([
                'n', 'r', 'a', 'i', 'time', 'best', 'slack'
            ])
        )
        data.rename(columns={'time': file[1] + '_time', 'best': file[1] + '_best', 'slack': file[1] + '_slack'}, inplace=True)
        combined_data = pd.merge(combined_data, data, on=['n', 'r', 'a', 'i'])

    combined_data.sort_values(by=['n', 'r', 'a', 'i'], inplace=True)

    latex_string = '\begin{tabular}{|l|l|l|l|l|'
    if mip_data != '':
        latex_string += 'rr|'
    for _ in sa_variants:
        latex_string += 'rr|'
    latex_string += '}\hline\multicolumn{4}{c|}{} & \multicolumn{1}{|c|}{Flow}'
    if mip_data != '':
        latex_string += ' & \multicolumn{2}{|c|}{MIP}'
    for sav in sa_variants:
        latex_string += '& \multicolumn{2}{|c|}{SA (' + sav[1] + ')}'
    latex_string += '\\\\ \hline $n$ & $P$ & adv.? & \# & Feas.?'
    if mip_data != '':
        latex_string += '& time & obj'
    for _ in sa_variants:
        latex_string += '& time & obj'
    latex_string += '\\\\ \hline'

    for idx, row in combined_data.iterrows():
        latex_string += f'{row["n"]:.0f} & {row["r"]:.0f} & {row["a"]:.0f} & {row["i"]:.0f} & '
        if int(row['feasible']) == 1:
            latex_string += '\cellcolor{green}'
        else:
            latex_string += '\cellcolor{red}'
        if mip_data != '':
            latex_string += f' &  {float(row["mip_time"]):.2f} & {float(row["mip_best"]):.2f}'
        for sav in sa_variants:
            cellcolor = '\cellcolor{red}'
            if float(row[sav[1] + "_slack"]) > 0:
                cellcolor = '\cellcolor{red!25}'
            else:
                cellcolor = '\cellcolor{green!25}'
            latex_string += f' & {cellcolor} {float(row[sav[1] + "_time"]):.2f} & {cellcolor} {float(row[sav[1] + "_best"]):.2f}'
        latex_string += '\\\\'

    latex_string += '\hline \end{tabular}'

    with open(os.path.join(output_dir, 'latex_table.tex'), 'w') as f:
        f.write(latex_string)


if __name__ == "__main__":
    main()