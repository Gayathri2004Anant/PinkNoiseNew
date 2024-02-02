import collections
import json
import pickle
import warnings
from glob import glob
from math import ceil, floor

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def get_runs(runs, conds=None, **settings):
    data = []
    confs = []
    for s in settings:
        if isinstance(settings[s], tuple):
            settings[s] = list(settings[s])
    for run in runs:
        for path in glob(f'dat/runs/{run}/working_directories/*/'):
            with open(path + 'settings.json') as f:
                conf = json.load(f)
                if 'conf' in conf:
                    conf.update(dict(conf['conf']))
                if not all(v is None and k not in conf or k in conf and conf[k] == v for k, v in settings.items()):
                    continue
                if conds and not all(conf[k] in v for k, v in conds.items()):
                    continue
            try:
                with open(path + 'state.pkl', 'rb') as f:
                    data.append(pickle.load(f))
                confs.append(conf)
            except:  # noqa E722
                pass
    return data, confs


def plot(ax, runs, color='C0', n_runs=5, label=None, conds=None, fmt='-', fill=True, **settings):
    data, _ = get_runs(runs, conds, **settings)
    if len(data) != n_runs:
        print(f"{len(data)}/{n_runs} runs found: {settings}")
    if len(data) == 0:
        return
    t = [t for t, _ in data[0]['evaluation_returns']]
    x = np.vstack([np.vstack([x for _, x in d['evaluation_returns']]).mean(1) for d in data])
    if fill:
        ax.fill_between(t, np.quantile(x, 0.25, 0), np.quantile(x, 0.75, 0), alpha=0.3, color=color)
    ax.plot(t, np.median(x, 0), fmt, color=color, label=label)


def get_env_norms(runs, envs, conds=None, **settings):
    norms = {}
    for env in tqdm(envs):
        data, _ = get_runs(runs, conds, env=env, **settings)
        x = np.array([np.mean([x for _, x in d['evaluation_returns']]) for d in data])
        norms[env] = (x.mean(), x.std())
    return norms


def get_box(runs, norms=None, n_runs=5, conds=None, a=0, b=1, **settings):
    data, _ = get_runs(runs, conds, **settings)
    if len(data) != n_runs:
        print(f"{len(data)}/{n_runs} runs found: {settings}")
    x = np.array([(e := np.array([x for _, x in d['evaluation_returns']]))[floor(a*len(e)):ceil(b*len(e))].mean()
                  for d in data])
    if norms:
        return (x - norms[settings['env']][0]) / norms[settings['env']][1]
    return x


env_code2name = {
    "('ball_in_cup', 'catch')": "Ball-In-Cup",
    "('cartpole', 'balance_sparse')": "Cartpole (b.)",
    "('cartpole', 'swingup_sparse')": "Cartpole (s.)",
    "('cheetah', 'run')": "Cheetah",
    "('hopper', 'hop')": "Hopper",
    "('pendulum', 'swingup')": "Pendulum",
    "('reacher', 'hard')": "Reacher",
    "('walker', 'run')": "Walker",
    'MountainCarContinuous-v0': "MountainCar",
    'door-v0': "Door",
}
env_name2code = {v: k for k, v in env_code2name.items()}


def bootstrap_average(df, N=1000, group=['env', 'agent'], var=['seed'], value='perf'):
    pt = df.pivot(index=group, columns=var, values=value)
    E, S = pt.shape
    samples = pt.values[np.arange(E), np.random.randint(S, size=(N, E))]
    return samples.mean(-1)


def get_env2best_beta(dataframe, aggregation_method="mean"):
    seed_aggregated = dataframe.groupby(["env", "beta"]).agg(aggregation_method).reset_index()
    best_betas = seed_aggregated.sort_values(by="perf").groupby(["env"]).tail(1)
    env_to_best_beta = best_betas[["env", "beta"]].set_index("env").T.to_dict()
    return env_to_best_beta


def get_env2worst_beta(dataframe, aggregation_method="mean"):
    seed_aggregated = dataframe.groupby(["env", "beta"]).agg(aggregation_method).reset_index()
    best_betas = seed_aggregated.sort_values(by="perf").groupby(["env"]).head(1)
    env_to_best_beta = best_betas[["env", "beta"]].set_index("env").T.to_dict()
    return env_to_best_beta


def ou_gaussian_as_beta(data, set_noise_constant=False):
    data = pd.DataFrame(data, copy=True)  # deep copy
    data.loc[data.noise == 'ou', 'beta'] = 'ou'
    data.loc[data.noise == 'wn', 'beta'] = 'wn'
    if set_noise_constant:
        data.loc[data.noise == 'ou', 'noise'] = 'constant'
        data.loc[data.noise == 'wn', 'noise'] = 'constant'
    return data


def show_latex(latex, preamble=""):
    # from jakobs code
    try:
        import tempfile
        import tex2pix
        from PIL import Image
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            textext = r"""
            \documentclass[convert]{{standalone}}

            \usepackage[inline]{{enumitem}}
            \usepackage{{graphicx}}
            \usepackage{{amsmath}}
            \usepackage{{amssymb}}
            \usepackage{{soul}}
            \usepackage{{verbatim}}
            \usepackage{{xcolor}}
            \usepackage{{colortbl}}
            \usepackage{{booktabs}}
            \usepackage{{multirow}}
            \usepackage{{pifont}}
            \usepackage{{bm}}
            \newcommand{{\cmark}}{{\ding{{51}}}}%
            \newcommand{{\xmark}}{{\ding{{55}}}}%
            {preamble}

            \begin{{document}}
            {latex}
            \end{{document}}
            """.format(preamble=preamble, latex=latex)
            r = tex2pix.Renderer(tex=textext)
            r.mkpng(tmp.name)
            #output = pnglatex.pnglatex(latex, 'output.png')
            image = Image.open(tmp.name)
        image.show()
        return image
    except ImportError:
        warnings.warn("Cannot show latex table, because packages are missing")
        print(latex)


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def format_pvalue(pvalue, minimal=False):
    # from jakobs code
    pvalue = np.abs(pvalue)
    if minimal:
        if pvalue >= 0.01:
            return f"${pvalue:0.2f}$"
        # elif pvalue >= 0.001:
        #     return f"${pvalue:0.3f}$"
        else:
            exponent = int(np.ceil(np.log10(pvalue)))
            return f"$\\texttt{{<}}10^{{{exponent}}}$"
    if pvalue >= 0.06:
        return f"$p={pvalue:0.2f}$"
    # elif pvalue >= 0.001:
    #     return f"$p={pvalue:0.3f}$"
    else:
        exponent = int(np.ceil(np.log10(pvalue)))
        return f"$p < 10^{{{exponent}}}$"

def normalize_group(df, columns_to_group, ignore_column=None):
        """
        This function allows perform a normalization over groups, i.e. the
        mean and std within a group, for each float(64|32) columns
        will be mean=0, std=1.
        :returns: pd.DataFrame
        """
        df = df.copy(deep=True)
        # we need to duplicate the columns first because they will be removed in the process
        group1 = [col + "_NORMALIZE_GROUP" for col in columns_to_group]
        for column in columns_to_group:
            df.loc[:, column + "_NORMALIZE_GROUP"] = df[column]

        if ignore_column is None:
            def false_function(*args):
                return False
            ignore_column = false_function
        elif isinstance(ignore_column, collections.abc.Iterable):
            set_of_columns_to_ignore = set(ignore_column)

            def ignore_if_in_set(column_name):
                return column_name in set_of_columns_to_ignore
            ignore_column = ignore_if_in_set
        unnormalized_group = df.groupby(by=group1)

        def transformer(x: pd.Series) -> pd.Series:
            if ignore_column(x.name):
                return x
            if len(x) <= 0:
                return x
            if x.dtype not in (np.dtype("float64"), np.dtype("float32")):
                return x
            std = x.std()
            if np.isclose(std, 0.):
                std = 1.0
            return (x - x.mean()) / std
        normalized_group = unnormalized_group.transform(transformer)
        return normalized_group
