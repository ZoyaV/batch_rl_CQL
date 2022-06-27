import argparse
import json
import os
from collections import defaultdict
from random import shuffle
from typing import Union
import matplotlib.pyplot as plt
import pathlib

import seaborn as sns

import pandas as pd
import torch
from pandas import DataFrame
from sample_factory.utils.utils import log
from tabulate import tabulate

import wandb
import yaml
# from evaluation.policies import run_algo
from agents.assistant_switcher import AssistantSwitcher
from agents.epom import EPOM
from agents.heuristic_switcher import HeuristicSwitcher
from agents.learnable_switcher import LearnableSwitcher
from agents.random_switcher import RandomSwitcher
from agents.replan import RePlan
from agents.prioritzied import Prioritized
from evaluation.eval_settings import EvaluationSettings, PlotView, TabularView, MultiPlotView
from evaluation.eval_utils import ResultsHolder
from learning.epom_config import Environment
from training_run import make_env
from utils.deal_with_wandb import fix_wandb_git
from utils.files import select_free_dir_name
from utils.gs2dict import generate_variants
from concurrent.futures import ProcessPoolExecutor as Pool

from utils.hashable_dict import HashableDict


def to_pandas(eval_configs):
    data = {}
    for config in eval_configs:
        data[config['id']] = {**config['results'], **config['resolved_vars']}

    return pd.DataFrame.from_dict(data, orient='index')


def preprocess_table(eval_configs, view):
    eval_configs.sort(key=lambda x: x['id'])

    df = to_pandas(eval_configs)

    for key_to_drop in view.drop_keys:
        if key_to_drop in df.head():
            df = df.drop(key_to_drop, axis=1)

    group_by = [x for x in df.head() if x not in eval_configs[0]['results'].keys()]
    df = df.groupby(by=group_by, as_index=False).mean()

    df: DataFrame = df.round(view.round_digits)

    if view.sort_by:
        df = df.sort_values(view.sort_by)

    if view.rename_fields:
        df = df.rename(columns=view.rename_fields)

    return df


def add_results_to_dict(results, resolved_vars, config_results, algo):
    if algo not in config_results:
        config_results[algo] = []
    info = {'.'.join(key): value for key, value in resolved_vars.items()}
    config_results[algo].append({'arguments': info, 'results': results})


def split_on_chunks(data, num_chunks):
    offset = int(1.0 * len(data) / num_chunks + 0.5)
    for i in range(0, num_chunks - 1):
        yield data[i * offset:i * offset + offset]
    yield data[num_chunks * offset - offset:]


def run_algo(eval_configs):
    algo_cfg = eval_configs[0].algo
    results = []

    if algo_cfg.name == 'EPOM':
        algo = EPOM(algo_cfg)
    elif algo_cfg.name == 'RePlan':
        algo = RePlan(algo_cfg)
    elif algo_cfg.name == 'Prioritized':
        algo = Prioritized(algo_cfg)
    elif algo_cfg.name == 'ASwitcher':
        algo = AssistantSwitcher(algo_cfg)
    elif algo_cfg.name == 'HSwitcher':
        algo = HeuristicSwitcher(algo_cfg)
    elif algo_cfg.name == 'RSwitcher':
        algo = RandomSwitcher(algo_cfg)
    elif algo_cfg.name == 'LSwitcher':
        algo = LearnableSwitcher(algo_cfg)
    else:
        raise KeyError(f"No algo with type {algo_cfg.name}")

    for idx, cfg in enumerate(eval_configs):
        env_cfg = Environment(**cfg.environment.dict())
        log.debug(f'[{idx + 1}/{len(eval_configs)}] ' + env_cfg.json(exclude_defaults=True))
        env = make_env(env_cfg)

        obs = env.reset()
        algo.after_reset()
        results_holder = ResultsHolder()

        dones = [False for _ in range(len(obs))]
        infos = [{'is_active': True} for _ in range(len(obs))]
        rew = [0 for _ in range(len(obs))]
        with torch.no_grad():
            while True:
                obs, rew, dones, infos = env.step(algo.act(obs, rew, dones, infos))
                infos[0] = {**infos[0], **algo.get_additional_info()}
                results_holder.after_step(infos)
                algo.after_step(dones)

                if all(dones):
                    break

        results.append(results_holder.get_final())
        env.close()

    return results


def run_in_parallel(configs):
    algo = configs[0].algo
    results = []

    log.debug(f'starting: {algo}')
    with Pool(algo.num_process) as executor:
        future_to_stuff = []
        for split in split_on_chunks(configs, algo.num_process):
            if not split:
                continue
            future_to_stuff.append(executor.submit(run_algo, split))
        for future in future_to_stuff:
            results += future.result()
    out = []
    for index in range(len(configs)):
        out.append({'results': results[index], 'id': configs[index].id, 'resolved_vars': configs[index].resolved_vars})
    log.debug(f'finished: {algo}')
    return out


def pop_key(key, d):
    to_extract = d
    for part in key[:-1]:
        if part not in to_extract:
            return None
        to_extract = to_extract[part]
    if key[-1] not in to_extract:
        return None
    return to_extract.pop(key[-1])


def save_summary(configs_to_save, summary_path):
    log.debug(f'Saving results to: {summary_path}')
    with open(summary_path, 'w') as f:
        json.dump(configs_to_save, f)


def prepare_plt(view: PlotView):
    plt.style.use(view.plt_style)
    plt.rcParams['figure.dpi'] = view.figure_dpi
    plt.rcParams['font.size'] = view.font_size
    plt.rcParams['legend.fontsize'] = view.legend_font_size
    plt.rcParams['figure.facecolor'] = view.figure_face_color


def prepare_plot_fields(view):
    x = view.x if view.x not in view.rename_fields else view.rename_fields[view.x]
    y = view.y if view.y not in view.rename_fields else view.rename_fields[view.y]
    hue = view.by if view.by not in view.rename_fields else view.rename_fields[view.by]
    return x, y, hue


def process_view(results, view: Union[TabularView, PlotView, MultiPlotView], save_path=None):
    if view.type == 'tabular':
        df = preprocess_table(results, view)
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            table = tabulate(df, headers=df.head(), tablefmt=view.table_format)
        if save_path:
            with open(str(save_path) + '.txt', "w") as f:
                f.write(table)
        if view.print_results:
            log.debug('\n' + table)
    elif view.type == 'plot':
        df = preprocess_table(results, view)

        prepare_plt(view)
        x, y, hue = prepare_plot_fields(view)
        sns.lineplot(x=x, y=y, data=df, ci=view.confidence_intervals, hue=hue, style=hue if view.line_types else None,
                     markers=view.markers, palette=view.palette)
        plt.tight_layout()
        plt.plot()
        path = str(
            save_path) + f'-{view.y}.{view.extension}' if save_path else f'{view.y.replace(" ", "-")}.{view.extension}'
        plt.savefig(path)
        plt.close()

    elif view.type == 'multi-plot':
        df = preprocess_table(results, view)

        over_keys = df[view.over].unique()
        num_cols = view.num_cols
        num_rows = len(over_keys) // num_cols

        prepare_plt(view)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))

        x, y, hue = prepare_plot_fields(view)

        for idx, over in enumerate(over_keys):
            tdf = df[df[view.over] == over]
            col = idx % num_cols
            row = idx // num_cols
            ax = axs[row, col]
            g = sns.lineplot(x=x, y=y, data=tdf, ci=view.confidence_intervals, hue=hue, ax=ax,
                             style=hue if view.line_types else None, markers=view.markers, palette=view.palette)
            ax.set_title(over)
            # if col == 0 and row == 0:
            #     continue
            # if col == num_cols - 1 and row == num_rows - 1:
            #     continue
            # g.get_legend().remove()

        plt.tight_layout()
        plt.plot()
        path = str(save_path) + f'-{view.y}.{view.extension}' if save_path else f'{view.y.replace(" ", "-")}'
        path += f'-{view.over}'
        path += f'.{view.extension}'
        plt.savefig(path)
        plt.close()


def evaluate(evaluation_config):
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    os.environ['OMP_NUM_THREADS'] = str(1)
    os.environ['MKL_NUM_THREADS'] = str(1)

    if 'eval_dir' in evaluation_config:
        path_for_saving_results = evaluation_config['eval_dir']
    else:
        path_for_saving_results = EvaluationSettings().eval_dir

    path_for_saving_results = pathlib.Path(path_for_saving_results)
    path_for_saving_results /= select_free_dir_name(path_for_saving_results)
    path_for_saving_results.mkdir(exist_ok=True, parents=True)

    with open(path_for_saving_results / 'config.yaml', "w") as f:
        yaml.dump(evaluation_config, f)

    use_wandb = evaluation_config.get('use_wandb', False)
    if use_wandb:
        wandb.init(project='pogema-eval-' + evaluation_config.get('name', ""), anonymous="allow", )

    grouped_by_algo = defaultdict(lambda: [])
    id_ = 0

    # temporary remove map to speedup config generation, so don't use grid_search with map!
    map_key = ['environment', 'grid_config', 'map']
    map_value = pop_key(map_key, evaluation_config)
    for resolved_vars, eval_config in generate_variants(evaluation_config):
        # noinspection Pydantic
        c = EvaluationSettings(**eval_config)
        shorted_resolved_vars = {key[-1]: value for key, value in resolved_vars.items()}
        shorted_resolved_vars['algo'] = c.algo.name
        if c.environment.grid_config.map_name:
            shorted_resolved_vars['map_name'] = c.environment.grid_config.map_name

        c.resolved_vars = shorted_resolved_vars

        c.environment.grid_config.map = map_value
        c.id = id_
        id_ += 1

        grouped_by_algo[HashableDict(c.algo.dict())].append(c)

    resulted_configs = []
    for key, configs in grouped_by_algo.items():
        shuffle(configs)

        results = run_in_parallel(configs)

        resulted_configs += results
        resulted_configs = list(sorted(resulted_configs, key=lambda x: x['id']))
        if len(grouped_by_algo.values()) > 1:
            algo_summary_path = path_for_saving_results / f'{configs[0].algo.name}.json'
            save_summary(results, summary_path=algo_summary_path)

            if use_wandb:
                wandb.save(str(algo_summary_path))

    full_summary_path = path_for_saving_results / 'results.json'
    save_summary(resulted_configs, full_summary_path)

    for idx, view in enumerate(evaluation_config.get('results_views', [])):
        if view['type'] == 'tabular':
            view = TabularView(**view)
        elif view['type'] == 'plot':
            view = PlotView(**view)
        else:
            raise KeyError
        process_view(resulted_configs, view, path_for_saving_results / f'view-{str(idx).zfill(2)}')

    if use_wandb:
        wandb.save(str(path_for_saving_results) + '/*')
        wandb.finish()

    return path_for_saving_results


def main():
    parser = argparse.ArgumentParser(description='Parallel evaluation over group of EvaluationConfigs.')
    parser.add_argument('--config_path', type=str, action="store", default='configs/eval-debug.yaml',
                        help='path folder with *.yaml EvaluationConfigs', required=False)
    args = parser.parse_args()

    with open(args.config_path) as f:
        evaluation_config = yaml.safe_load(f)

    evaluate(evaluation_config)


if __name__ == '__main__':
    fix_wandb_git()
    main()
