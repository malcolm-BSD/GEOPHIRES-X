#! python
"""
Framework for running Monte Carlo simulations using GEOPHIRES v3.0 & HIP-RA 1.0
build date: September 2023
Created on Wed November  16 10:43:04 2017
@author: Malcolm Ross V3
@author: softwareengineerprogrammer
"""

import argparse
import ast
import concurrent.futures
import json
import logging
import logging.config
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from pylocker import Locker
from rich.console import Console
from rich.table import Table
from scipy.stats import binom, norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from geophires_x.GeoPHIRESUtils import InsertImagesIntoHTML, render_default

logger = logging.getLogger('root')  # TODO should be getting __name__ logger instead of root
logger.setLevel(logging.INFO)
pb: Optional[Any] = None


def parse_value(value_str: str) -> Union[None, bool, int, float, str, List[Any]]:
    """
    Parses a string value into a proper Python type (None, bool, int, float, list, or str).
    """
    value_str = value_str.strip()

    if value_str.lower() in {"none", "null", "nil"}:
        return None

    if ',' in value_str:
        parts = [v.strip() for v in value_str.split(',')]
        return [parse_value(v) for v in parts]

    if value_str.lower() in {"true", "false"}:
        return value_str.lower() == "true"

    try:
        return int(value_str)
    except ValueError:
        pass

    try:
        return float(value_str)
    except ValueError:
        pass

    return value_str


def extract_output_value(result_lines: List[str], output: str) -> Union[bool, int, float, str, List[Any]]:
    """
    Extracts the value corresponding to a given output key from a list of strings.
    Ignores units or extra text after the value. Returns '' if not found.

    Args:
        result_lines: List of strings containing key-value pairs.
        output: The key to search for.

    Returns:
        Parsed value (int, float, bool, etc.), or '' if not found.
    """
    pattern = rf'^\s*{re.escape(output)}\s*:\s*(.+)$'

    for line in result_lines:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            raw_value_section = match.group(1).strip()

            token_match = re.match(
                r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|true|false|null|none|nil|\w+(?:,\s*\w+)*)',
                raw_value_section,
                re.IGNORECASE,
            )
            if token_match:
                return parse_value(token_match.group(1))

    return ""


def wait_for_file_stable(
    filepath: str,
    timeout_sec: Optional[float] = 60,
    check_interval_sec: float = 0.5,
    stable_period_sec: float = 2,
) -> bool:
    """
    Waits for a file to appear and then for its modification time to remain unchanged
    for a specified stable period, indicating that the file is no longer being written to.
    """
    start_time = time.time()

    while True:
        if os.path.exists(filepath):
            try:
                last_mtime = os.path.getmtime(filepath)
            except OSError:
                time.sleep(check_interval_sec)
                continue

            stable_start = time.time()
            while True:
                time.sleep(check_interval_sec)
                try:
                    current_mtime = os.path.getmtime(filepath)
                except OSError:
                    break

                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_period_sec:
                    return True

                if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                    return False
        else:
            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                return False
            time.sleep(check_interval_sec)


def calculate_binomial_value_complementary(
    v1: float,
    trials_v1: float,
    prob_v1: float,
    trials_v2: float,
    prob_v2: float,
) -> float:
    """
    Calculate the complementary v2 such that it falls in the complementary position in its binomial distribution
    relative to v1's position in its binomial distribution.
    """
    p_v1 = binom.cdf(v1, trials_v1, prob_v1)
    complementary_p_v1 = 1 - p_v1
    v2 = binom.ppf(complementary_p_v1, trials_v2, prob_v2)
    return v2


def calculate_binomial_value(
    v1: float,
    trials_v1: float,
    prob_v1: float,
    trials_v2: float,
    prob_v2: float,
) -> float:
    """
    Calculate v2 such that it falls in the same relative position in its binomial distribution
    as v1 falls in its binomial distribution.
    """
    p_v1 = binom.cdf(v1, trials_v1, prob_v1)
    v2 = binom.ppf(p_v1, trials_v2, prob_v2)
    return v2


def calculate_lognormal_value_complementary(
    v1: float,
    mean_v1: float,
    std_v1: float,
    mean_v2_log: float,
    std_v2_log: float,
) -> float:
    """
    Calculate v2 such that it is in the complementary position of v1 in their respective distributions
    with v2 being log-normal.
    """
    p_v1 = norm.cdf(v1, loc=mean_v1, scale=std_v1)
    p_v2 = 1 - p_v1
    v2 = np.exp(norm.ppf(p_v2, loc=mean_v2_log, scale=std_v2_log))
    return v2


def calculate_lognormal_value(
    v1: float,
    mean_v1: float,
    std_v1: float,
    mean_v2_log: float,
    std_v2_log: float,
) -> float:
    """
    Calculate v2 such that it falls in the same relative position in its log-normal distribution
    as v1 falls in its normal distribution.
    """
    z_score = (v1 - mean_v1) / std_v1
    log_v2 = mean_v2_log + z_score * std_v2_log
    v2 = np.exp(log_v2)
    return v2


def calculate_normal_complementary(
    v1: float,
    mean_v1: float,
    std_v1: float,
    mean_v2: float,
    std_v2: float,
) -> float:
    """
    Calculate v2 such that it is in the complementary position of v1 in their respective normal distributions.
    """
    p_v1 = norm.cdf(v1, loc=mean_v1, scale=std_v1)
    p_v2 = 1 - p_v1
    v2 = norm.ppf(p_v2, loc=mean_v2, scale=std_v2)
    return v2


def calculate_normal_value(
    v1: float,
    mean_v1: float,
    std_v1: float,
    mean_v2: float,
    std_v2: float,
) -> float:
    """
    Calculate v2 such that it falls in the same relative position in its normal distribution
    as v1 falls in its own distribution.
    """
    z_score = (v1 - mean_v1) / std_v1
    v2 = mean_v2 + z_score * std_v2
    return v2


def calculate_scaled_value_complementary(
    v1_value: float,
    v1_range: tuple[float, float],
    v2_range: tuple[float, float],
) -> float:
    """
    Calculate the complementary scaled value of v2 based on the position of v1 within its range.
    """
    min_v1, max_v1 = v1_range
    min_v2, max_v2 = v2_range

    if v1_value < min_v1 or v1_value > max_v1:
        raise ValueError('v1_value is out of the specified range for v1.')

    v1_percentage = ((v1_value - min_v1) / (max_v1 - min_v1)) * 100
    v2_percentage = 100 - v1_percentage
    v2_scaled = (v2_percentage / 100) * (max_v2 - min_v2) + min_v2
    return v2_scaled


def calculate_scaled_value(v1: float, v1_range: tuple[float, float], v2_range: tuple[float, float]) -> float:
    """Scales a value from one range to another."""
    min_v1, max_v1 = v1_range
    min_v2, max_v2 = v2_range

    v2 = (v1 - min_v1) / (max_v1 - min_v1) * (max_v2 - min_v2) + min_v2
    return v2


def Write_HTML_Output(
    html_path: str,
    df: pd.DataFrame,
    outputs: list,
    mins: list,
    maxs: list,
    medians: list,
    averages: list,
    means: list,
    std: list,
    full_names: set,
    short_names: set,
) -> None:
    """
    Write_HTML_Output - write the results of the Monte Carlo simulation to an HTML file
    """

    results_table = Table(title='GEOPHIRES/HIR-RA Monte Carlo Results')
    results_table.add_column('Iteration #', no_wrap=True, justify='center')
    for output in df.axes[1]:
        results_table.add_column(output.replace(',', ''), no_wrap=True, justify='center')

    statistics_table = Table(title='GEOPHIRES/HIR-RA Monte Carlo Statistics')
    statistics_table.add_column('Output Parameter Name', no_wrap=True, justify='center')
    statistics_table.add_column('minimum', no_wrap=True, justify='center')
    statistics_table.add_column('maximum', no_wrap=True, justify='center')
    statistics_table.add_column('median', no_wrap=True, justify='center')
    statistics_table.add_column('average', no_wrap=True, justify='center')
    statistics_table.add_column('mean', no_wrap=True, justify='center')
    statistics_table.add_column('standard deviation', no_wrap=True, justify='center')

    for index, row in df.iterrows():
        data = row.values[0 : len(outputs)]
        str_to_parse = str(row.values[len(outputs)]).strip().replace('(', '').replace(')', '')
        fields = str_to_parse.split(';')
        for field in fields:
            if len(field) > 0:
                key, value = field.split(':')
                data = np.append(data, float(value))

        results_table.add_row(str(int(index)), *[render_default(d) for d in data])

    for i in range(len(outputs)):
        statistics_table.add_row(
            outputs[i],
            render_default(mins[i]),
            render_default(maxs[i]),
            render_default(medians[i]),
            render_default(averages[i]),
            render_default(means[i]),
            render_default(std[i]),
        )

    console = Console(style='bold white on black', force_terminal=True, record=True, width=500)
    console.print(results_table)
    console.print(' ')
    console.print(statistics_table)
    console.save_html(html_path)

    InsertImagesIntoHTML(html_path, full_names, short_names)


def check_and_replace_mean(input_value, args) -> list:
    """Check if the user requested a mean value replacement by specifying '#'."""

    i = 0
    for input_x in input_value:
        if '#' in input_x:
            vari_name = input_value[0]
            with open(args.Input_file) as f:
                ss = f.readlines()
            for s in ss:
                if str(s).startswith(vari_name):
                    s2 = s.split(',')
                    input_value[i] = s2[1]
                    break
            break
        i += 1

    return input_value


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def evaluate_expression(expression: str, variable_value: float) -> float:
    try:
        x = sp.symbols('x')
        parsed_expr = sp.sympify(expression)
        evaluated_expr = parsed_expr.subs(x, variable_value)
        result = evaluated_expr.evalf()
        return float(result)
    except Exception:
        return -99999


def extract_values(data: List[str]) -> List[float]:
    seen: dict[str, float] = {}
    order: List[str] = []

    for item in data:
        name, value = item.split(':')
        seen[name] = float(value)
        if name not in order:
            order.append(name)

    return [seen[name] for name in order]


def make_tornado_plots_stacked(
    df: pd.DataFrame,
    input_df: pd.DataFrame,
    ins: List[str],
    outs: List[List[str]],
    output_file: str,
    html_path: str,
    full_names: set,
    short_names: set,
) -> None:
    df.columns = df.columns.str.strip()
    input_df.columns = input_df.columns.str.strip()

    if isinstance(outs, list) and len(outs) > 0 and isinstance(outs[0], list):
        clean_outs = [col.strip() for col in outs[0]]
    else:
        raise ValueError("outs must be a list of column name lists. Example: [['Input A', 'Input B']]")

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(input_df[clean_outs])

    coefficients: dict[str, np.ndarray] = {}
    for output in ins:
        if output:
            y = df[output].values.reshape(-1, 1)
            y_scaled = StandardScaler().fit_transform(y).ravel()
            model = LinearRegression().fit(X, y_scaled)
            coefficients[output] = model.coef_

    coef_df = pd.DataFrame(coefficients, index=clean_outs)
    coef_df = coef_df.reindex(coef_df[ins[0]].abs().sort_values().index)

    ax = coef_df.plot(kind='barh', figsize=(10, 6))
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Standardized Regression Coefficients by Output')
    plt.xlabel('Standardized Coefficient')
    plt.tight_layout()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.legend(title='Output Metric')
    save_path = Path(Path(output_file).parent, 'stacked_tornado.png')
    if html_path:
        save_path = Path(Path(html_path).parent, 'stacked_tornado.png')
    plt.savefig(save_path)
    plt.close()
    full_names.add(save_path)
    short_names.add('stacked_tornado')


def make_tornado_plots(
    df: pd.DataFrame,
    input_df: pd.DataFrame,
    ins: List[str],
    outs: List[List[str]],
    output_file: str,
    html_path: str,
    full_names: set,
    short_names: set,
) -> None:
    for tornado_in in ins:
        for tornado_outs in outs:
            if tornado_in and tornado_outs:
                tornado_outs = [col.strip() for col in tornado_outs]
                input_df.columns = input_df.columns.str.strip()
                X = input_df[tornado_outs].values

                tornado_in_clean = tornado_in.strip()
                df.columns = df.columns.str.strip()
                y = df[tornado_in_clean].values.reshape(-1, 1)

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y).ravel()

                model = LinearRegression()
                model.fit(X_scaled, y_scaled)

                regression_df = pd.DataFrame({'Input': tornado_outs, 'Coefficient': model.coef_})
                regression_df['AbsCoefficient'] = regression_df['Coefficient'].abs()
                regression_df = regression_df.sort_values('AbsCoefficient', ascending=True)

                plt.figure(figsize=(8, 6))
                plt.barh(regression_df['Input'], regression_df['Coefficient'])
                plt.xlabel('Standardized Regression Coefficient')
                plt.title('Sensitivity Analysis (Regression) on ' + tornado_in_clean)
                plt.grid(True)
                plt.tight_layout()
                save_path = Path(Path(output_file).parent, f'{tornado_in_clean}_tornado.png')
                if html_path:
                    save_path = Path(Path(html_path).parent, f'{tornado_in_clean}_tornado.png')
                plt.savefig(save_path)
                plt.close()
                full_names.add(save_path)
                short_names.add(tornado_in_clean)


def parse_random_args(expression: str) -> str:
    patterns = {
        'choice': r'random\\.choice\\(\\[(.*?)\\]\\)',
        'random': r'random\\.random\\(\\)',
        'uniform': r'random\\.uniform\\((.*?),(.*?)\\)',
    }

    def replace_choice(match: re.Match[str]) -> str:
        args_str = match.group(1).replace(';', ',')
        args_list = ast.literal_eval(f'[{args_str}]')
        return str(random.choice(args_list))

    def replace_random(match: re.Match[str]) -> str:
        return str(random.random())

    def replace_uniform(match: re.Match[str]) -> str:
        a = float(match.group(1).strip())
        b = float(match.group(2).strip())
        return str(random.uniform(a, b))

    if '.choice' in expression:
        expression = re.sub(patterns['choice'], replace_choice, expression)
    elif '.random' in expression:
        expression = re.sub(patterns['random'], replace_random, expression)
    elif '.uniform' in expression:
        expression = re.sub(patterns['uniform'], replace_uniform, expression)
    return expression


def work_package(pass_list: list) -> None:
    """Function that is called by the executor. It does the work of running the simulation."""

    log = logging.getLogger('root')
    log.setLevel(logging.INFO)

    input_values: List[List[Any]] = pass_list[0]
    outputs: List[str] = pass_list[1]
    links_ratio: List[List[Any]] = pass_list[2]
    links_equal: List[List[Any]] = pass_list[3]
    links_reverse: List[List[Any]] = pass_list[4]
    links_math: List[List[Any]] = pass_list[5]
    args: argparse.Namespace = pass_list[6]
    output_file: str = pass_list[7]
    working_dir: str = pass_list[8]  # noqa: F841
    python_path: str = pass_list[9]

    input_file_entries = ''
    randomized_values: List[List[Any]] = []

    for input_value in input_values:
        input_name = input_value[0].strip()
        distribution = input_value[1].strip()
        f2: Union[str, float] = 0.0
        f3: Union[str, float] = 0.0
        f4: Union[str, float] = 0.0
        if len(input_value) > 2:
            f2 = input_value[2].strip()
            if is_number(str(f2)):
                f2 = float(f2)
        if len(input_value) > 3:
            f3 = input_value[3].strip()
            if is_number(str(f3)):
                f3 = float(f3)
        if len(input_value) > 4:
            f4 = input_value[4].strip()
            if is_number(str(f4)):
                f4 = float(f4)

        rando = 0.0
        if distribution.startswith('normal'):
            rando = np.random.normal(float(f2), float(f3))
        elif distribution.startswith('uniform'):
            rando = np.random.uniform(float(f2), float(f3))
        elif distribution.startswith('triangular'):
            rando = np.random.triangular(float(f2), float(f3), float(f4))
        elif distribution.startswith('lognormal'):
            rando = np.random.lognormal(float(f2), float(f3))
        elif distribution.startswith('binomial'):
            rando = np.random.binomial(int(float(f2)), float(f3))
        elif distribution.startswith('nominal'):
            values = str(f2).split(':')
            rando = random.choice(values)
        else:
            raise ValueError(f'Unsupported distribution: {distribution}')

        randomized_values.append([input_name, rando, distribution, f2, f3, f4])
        input_file_entries += input_name + ', ' + str(rando) + '\n'

    for link in links_equal:
        master_input = str(link[0]).strip()
        slave_input = str(link[1]).strip()
        master_value = None
        for rv in randomized_values:
            if rv[0] == master_input:
                master_value = rv[1]
                break
        if master_value is not None:
            input_file_entries += slave_input + ', ' + str(master_value) + '\n'

    for link2 in links_ratio:
        master_input = str(link2[0]).strip()
        slave_input = str(link2[1]).strip()
        master_value = None
        master_distribution = ''
        master_params = [0.0, 0.0, 0.0]
        for rv3 in randomized_values:
            if rv3[0] == master_input:
                master_value = float(rv3[1])
                master_distribution = str(rv3[2]).strip()
                master_params = [rv3[3], rv3[4], rv3[5] if len(rv3) > 5 else 0.0]
                break
        for rv4 in randomized_values:
            if rv4[0] == slave_input and master_value is not None:
                slave_params = [rv4[3], rv4[4], rv4[5] if len(rv4) > 5 else 0.0]
                if master_distribution.startswith('uniform') or master_distribution.startswith('triangular'):
                    rv4[1] = calculate_scaled_value(
                        master_value,
                        (float(master_params[0]), float(master_params[1])),
                        (float(slave_params[0]), float(slave_params[1])),
                    )
                elif master_distribution.startswith('normal'):
                    rv4[1] = calculate_normal_value(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                elif master_distribution.startswith('lognormal'):
                    rv4[1] = calculate_lognormal_value(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                elif master_distribution.startswith('binomial'):
                    rv4[1] = calculate_binomial_value(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                input_file_entries += slave_input + ', ' + str(rv4[1]) + '\n'
                break

    for link3 in links_reverse:
        master_input = str(link3[0]).strip()
        slave_input = str(link3[1]).strip()
        master_value = None
        master_distribution = ''
        master_params = [0.0, 0.0, 0.0]
        for rv5 in randomized_values:
            if rv5[0] == master_input:
                master_value = float(rv5[1])
                master_distribution = str(rv5[2]).strip()
                master_params = [rv5[3], rv5[4], rv5[5] if len(rv5) > 5 else 0.0]
                break
        for rv6 in randomized_values:
            if rv6[0] == slave_input and master_value is not None:
                slave_params = [rv6[3], rv6[4], rv6[5] if len(rv6) > 5 else 0.0]
                if master_distribution.startswith('uniform') or master_distribution.startswith('triangular'):
                    rv6[1] = calculate_scaled_value_complementary(
                        master_value,
                        (float(master_params[0]), float(master_params[1])),
                        (float(slave_params[0]), float(slave_params[1])),
                    )
                elif master_distribution.startswith('normal'):
                    rv6[1] = calculate_normal_complementary(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                elif master_distribution.startswith('lognormal'):
                    rv6[1] = calculate_lognormal_value_complementary(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                elif master_distribution.startswith('binomial'):
                    rv6[1] = calculate_binomial_value_complementary(
                        master_value,
                        float(master_params[0]),
                        float(master_params[1]),
                        float(slave_params[0]),
                        float(slave_params[1]),
                    )
                input_file_entries += slave_input + ', ' + str(rv6[1]) + '\n'
                break

    for link4 in links_math:
        master_input = str(link4[0].strip())
        slave_input = str(link4[1].strip())
        equation = str(link4[2].strip().replace(';', ',')).casefold()
        equation = parse_random_args(equation)
        for rv7 in randomized_values:
            if rv7[0] == master_input:
                master_value = float(rv7[1])
                new_val = evaluate_expression(equation, master_value)
                input_file_entries += slave_input + ', ' + str(new_val) + '\n'
                break

    tmp_input_file: str = str(Path(tempfile.gettempdir(), f'{uuid.uuid4()!s}.txt'))
    tmp_output_file: str = tmp_input_file.replace('.txt', '_result.txt')

    shutil.copyfile(args.Input_file, tmp_input_file)

    with open(tmp_input_file, 'a') as f:
        f.write(input_file_entries)

    if os.name != 'nt':
        from geophires_monte_carlo.common import _get_logger  # noqa: F401
        from geophires_x_client import GeophiresInputParameters, GeophiresXClient, GeophiresXResult
        from hip_ra import HipRaClient, HipRaInputParameters, HipRaResult
        from hip_ra_x import HipRaXClient

        if args.Code_File.endswith('GEOPHIRESv3.py'):
            geophires_client: GeophiresXClient = GeophiresXClient()
            result: GeophiresXResult = geophires_client.get_geophires_result(
                GeophiresInputParameters(from_file_path=Path(tmp_input_file))
            )
            shutil.copyfile(result.output_file_path, tmp_output_file)
        elif args.Code_File.endswith('HIP_RA.py'):
            hip_ra_client: HipRaClient = HipRaClient()
            result: HipRaResult = hip_ra_client.get_hip_ra_result(
                HipRaInputParameters(file_path_or_params_dict=Path(tmp_input_file))
            )
            shutil.copyfile(result.output_file_path, tmp_output_file)
        elif args.Code_File.endswith('hip_ra_x.py'):
            hip_ra_x_client: HipRaXClient = HipRaXClient()
            result: HipRaResult = hip_ra_x_client.get_hip_ra_result(
                HipRaInputParameters(file_path_or_params_dict=Path(tmp_input_file))
            )
            shutil.copyfile(result.output_file_path, tmp_output_file)
    else:
        if os.cpu_count() is not None and os.cpu_count() < 4:
            log.info(
                'Python indicates that we are running on a local laptop/desktop so we will use local processes,'
                ' but the CPU/thread count is too small'
            )
            sys.exit(-74385)

        sprocess = subprocess.Popen(
            [python_path, args.Code_File, tmp_input_file, tmp_output_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sprocess.wait()

    s1 = ''
    result_s = ''
    local_outputs = outputs

    if not wait_for_file_stable(tmp_output_file, timeout_sec=10):
        logger.warning(f'output file does not exist: {tmp_output_file}')
        sys.exit(-33)

    with open(tmp_output_file) as f:
        result_lines = f.readlines()

    try:
        for out in local_outputs:
            s1 = extract_output_value(result_lines, out)
            if s1 in (None, ''):
                raise ValueError(f"No value found for required output key: '{out}'")
            result_s += str(s1) + ', '
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        raise

    result_s += '(' + input_file_entries.replace('\n', ';', -1).replace(', ', ':', -1) + ')'
    result_s = result_s.strip(' ').strip(',')
    result_s += '\n'

    Path.unlink(Path(tmp_input_file))
    Path.unlink(Path(tmp_output_file))

    lock_pass = str(uuid.uuid1())
    FL = Locker(filePath=output_file, lockPass=lock_pass, timeout=10, mode='a')
    with FL as r:
        acquired, code, fd = r
        if fd is not None:
            fd.write(result_s)


def main(command_line_args=None, enable_geophires_monte_carlo_logging_config: bool = False) -> None:
    r"""
    main - this is the main function that is called when the program is run
    """

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)
    logger.info(f'Init {__name__!s}')

    tic = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('Code_File', help='Code File')
    parser.add_argument('Input_file', help='Input file')
    parser.add_argument('MC_Settings_file', help='MC Settings file')
    parser.add_argument('MC_OUTPUT_FILE', help='Output file', nargs='?')

    if command_line_args is None:
        logger.warning('Command line args were not passed explicitly, falling back to sys.argv')
        command_line_args = sys.argv[1:]

    args = parser.parse_args(command_line_args)

    with open(args.MC_Settings_file, encoding='UTF-8') as f:
        flist = f.readlines()

    inputs: List[List[str]] = []
    outputs: List[str] = []
    links_ratio: List[List[str]] = []
    links_reverse: List[List[str]] = []
    links_equal: List[List[str]] = []
    links_math: List[List[str]] = []
    iterations = 0
    output_file = (
        args.MC_OUTPUT_FILE
        if 'MC_OUTPUT_FILE' in args and args.MC_OUTPUT_FILE is not None
        else str(Path(Path(args.Input_file).parent, 'MC_Result.txt').absolute())
    )
    code_file_name = Path(args.Code_File).name
    python_path = sys.executable
    html_path = ''
    tornado1_in = tornado2_in = tornado3_in = ''
    tornado1_outs: List[str] = []
    tornado2_outs: List[str] = []
    tornado3_outs: List[str] = []

    for line in flist:
        clean = line.strip()
        pair = clean.split(',')
        if len(pair) < 2:
            continue
        pair[1] = pair[1].strip()
        if pair[0].startswith('INPUT'):
            inputs.append(pair[1:])
            continue
        if pair[0].startswith('OUTPUT'):
            outputs.append(pair[1])
            continue
        if pair[0].startswith('ITERATIONS'):
            iterations = int(pair[1])
            continue
        if pair[0].startswith('MC_OUTPUT_FILE'):
            output_file = pair[1]
            continue
        if pair[0].startswith('PYTHON_PATH'):
            python_path = pair[1]
            continue
        if pair[0].startswith('HTML_PATH'):
            html_path = pair[1]
            continue
        if pair[0].casefold() == '=LINK'.casefold():
            links_equal.append(pair[1:])
            continue
        if pair[0].casefold() == 'RLINK'.casefold():
            links_reverse.append(pair[1:])
            continue
        if pair[0].casefold() == 'LINK'.casefold():
            links_ratio.append(pair[1:])
            continue
        if pair[0].casefold() == '+LINK'.casefold():
            links_math.append(pair[1:])
            continue
        if pair[0].casefold() == 'TORNADO1'.casefold():
            tornado1_in = str(pair[1]).strip()
            tornado1_outs = pair[2:]
            continue
        if pair[0].casefold() == 'TORNADO2'.casefold():
            tornado2_in = str(pair[1]).strip()
            tornado2_outs = pair[2:]
            continue
        if pair[0].casefold() == 'TORNADO3'.casefold():
            tornado3_in = str(pair[1]).strip()
            tornado3_outs = pair[2:]
            continue

    for input_value in inputs:
        check_and_replace_mean(input_value, args)

    header = ''

    for output in outputs:
        header += output + ', '

    for inp in inputs:
        header += inp[0] + ', '

    for link in links_math:
        header += link[1] + ', '

    header = ''.join(header.rsplit(' ', 1))
    header = ''.join(header.rsplit(',', 1))
    header += '\n'

    with open(output_file, 'w') as f:
        f.write(header)

    working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(working_dir)
    working_dir = working_dir + os.sep

    pass_list = [
        inputs,
        outputs,
        links_ratio,
        links_equal,
        links_reverse,
        links_math,
        args,
        output_file,
        working_dir,
        python_path,
    ]

    executor_args = [pass_list for _ in range(iterations)]

    with tqdm(total=iterations, desc='Finished processes', unit='iteration') as pbar:
        max_workers = max(1, (os.cpu_count() or 1) - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(work_package, arg): arg for arg in executor_args}
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    print('\n')
    logger.info('Done with calculations! Summarizing...')

    with open(output_file) as f:
        header_line = f.readline()
        all_results = f.readlines()

    result_count = 0
    results: List[List[float]] = []
    for line in all_results:
        result_count += 1
        if '-9999.0' not in line and len(header_line) > 1:
            line = line.strip()
            if len(line) > 10:
                line, sep, tail = line.partition(', (')
                line = line.replace('(', '').replace(')', '')
                results.append([float(y) if is_number(y) else -999.999 for y in line.split(',')])
        else:
            logger.warning(f'-9999.0 or space found in line {result_count!s}')

    actual_records_count = len(results)
    if len(results) < 1:
        raise RuntimeError(
            'No MC results generated, '
            f'this is likely caused by {code_file_name} throwing an exception '
            f'when run with your input file.'
        )

    mins = np.nanmin(results, 0)
    maxs = np.nanmax(results, 0)
    medians = np.nanmedian(results, 0)
    averages = np.average(results, 0)
    means = np.nanmean(results, 0)
    std = np.nanstd(results, 0)

    results_pd = pd.read_csv(output_file)
    df = pd.DataFrame(results_pd)

    input_df = pd.DataFrame()

    input_row = df[df.columns[len(outputs)]].tolist()[0]
    input_row = input_row.replace('(', '').replace(')', '')
    input_row = input_row.strip().strip(';')
    input_columns_data = input_row.split(';')
    for input_column_data in input_columns_data:
        input_column_name, _ = input_column_data.split(':')
        input_df[input_column_name] = []

    for i in range(actual_records_count):
        input_row = str(df[df.columns[len(outputs)]].tolist()[i])
        if len(input_row) < 10:
            continue
        input_row = input_row.replace('(', '').replace(')', '')
        input_row = input_row.strip().strip(';')
        input_columns_data = input_row.split(';')
        data = extract_values(input_columns_data)
        input_df.loc[i] = data[: input_df.columns.size]

    logger.info(f'Calculation Time: {time.time() - tic:10.3f} sec')
    logger.info(f'Calculation Time per iteration: {(time.time() - tic) / actual_records_count:10.3f} sec')
    if iterations != actual_records_count:
        msg = f'NOTE: {actual_records_count!s} iterations finished successfully and were used to calculate the statistics.'
        logger.warning(msg)

    annotations = ''
    outputs_result: dict[str, dict[str, float]] = {}

    full_names: set = set()
    short_names: set = set()
    with open(output_file, 'a') as f:
        for i in range(len(inputs)):
            input_name = inputs[i][0]
            plt.figure(figsize=(8, 6))
            ax = plt.subplot()
            ax.set_title(input_name)
            ax.set_xlabel('Random Values')
            ax.set_ylabel('Probability')

            plt.figtext(0.11, 0.74, annotations, fontsize=8)
            ret = plt.hist(input_df[input_df.columns[i]].tolist(), bins=50, density=True)
            fname = input_df.columns[i].strip().replace('/', '-')
            save_path = Path(Path(output_file).parent, f'{fname}.png')
            if html_path:
                save_path = Path(Path(html_path).parent, f'{fname}.png')
            plt.savefig(save_path)
            plt.close()
            full_names.add(save_path)
            short_names.add(fname)

        for i in range(len(outputs)):
            output = outputs[i]
            f.write(f'{output}:\n')
            outputs_result[output] = {}
            outputs_result[output]['minimum'] = mins[i]
            outputs_result[output]['maximum'] = maxs[i]
            outputs_result[output]['median'] = medians[i]
            outputs_result[output]['average'] = averages[i]
            outputs_result[output]['mean'] = means[i]
            outputs_result[output]['standard deviation'] = std[i]

            for k, v in outputs_result[output].items():
                display = f'     {k}: {v:,.2f}\n'
                f.write(display)
                annotations += display

            plt.figure(figsize=(8, 6))
            ax = plt.subplot()
            ax.set_title(output)
            ax.set_xlabel('Output units')
            ax.set_ylabel('Probability')

            plt.figtext(0.11, 0.74, annotations, fontsize=8)
            ret = plt.hist(df[df.columns[i]].tolist(), bins=50, density=True)
            f.write(f'bin values (as percentage): {ret[0]!s}\n')
            f.write(f'bin edges: {ret[1]!s}\n')
            fname = df.columns[i].strip().replace('/', '-')
            save_path = Path(Path(output_file).parent, f'{fname}.png')
            if html_path:
                save_path = Path(Path(html_path).parent, f'{fname}.png')
            plt.savefig(save_path)
            plt.close()
            full_names.add(save_path)
            short_names.add(fname)
            annotations = ''

    if tornado1_in:
        make_tornado_plots_stacked(
            df,
            input_df,
            [tornado1_in, tornado2_in, tornado3_in],
            [tornado1_outs, tornado2_outs, tornado3_outs],
            output_file,
            html_path,
            full_names,
            short_names,
        )
        make_tornado_plots(
            df,
            input_df,
            [tornado1_in, tornado2_in, tornado3_in],
            [tornado1_outs, tornado2_outs, tornado3_outs],
            output_file,
            html_path,
            full_names,
            short_names,
        )

        if html_path:
            Write_HTML_Output(
                html_path,
                df,
                outputs,
                mins,
                maxs,
                medians,
                averages,
                means,
                std,
                full_names,
                short_names,
            )

    with open(Path(output_file).with_suffix('.json'), 'w') as json_output_file:
        json_output_file.write(json.dumps(outputs_result))
        logger.info(f'Wrote JSON results to {json_output_file.name}')

    logger.info(f'Complete {__name__!s}: {sys._getframe().f_code.co_name}')


if __name__ == '__main__':
    logger.info(f'Init {__name__!s}')

    main(command_line_args=sys.argv[1:])
