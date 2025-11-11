#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Analysis and plotting utilities for regression data.
"""
import itertools
import json
import re

from pathlib import Path, PurePosixPath
from enum import IntEnum

import pandas as pd
import numpy as np

Islc = pd.IndexSlice

KernelCategory = IntEnum("KernelCategory", ["GEO", "PHYS", "GP", "USER"], start=0)

RESULT_LEVELS = ('problem', 'geo', 'arch', 'instance')
GEO_COLORS = {
    'orange': '#F6A75E',
    'vecgeom': '#5785B7',
    'geant4': '#9F239D',
}
ARCH_SHAPES = {
    'gpu': 'x',
    'cpu': 's',
    'gpu+g4': '+',
    'cpu+g4': 'd',
    'g4': 'o',
}
KERNEL_CATEGORY_LABELS = ["Geometry", "Physics", "Geo&Phys", "User"]
KERNEL_ORDERING = {
    'along-step-neutral': KernelCategory.GEO,
    'along-step-general-linear': KernelCategory.GP,
    'along-step-cylmap-msc': KernelCategory.GP,
    'along-step-uniform-msc': KernelCategory.GP,
    'initialize-tracks': KernelCategory.GP,
    'extend-from-primaries': KernelCategory.GP,
    'extend-from-secondaries': KernelCategory.GP,
    'geo-boundary': KernelCategory.GEO,
    'physics-discrete-select': KernelCategory.PHYS,
    'pre-step': KernelCategory.PHYS,
    'tracking-cut': KernelCategory.PHYS,
    'step-gather-post': KernelCategory.USER,
    'step-gather-post': KernelCategory.USER,
}

CPU_POWER_PER_TASK= {
    "frontier": 225 / 8, # 64-core AMD “Optimized 3rd Gen EPYC”
    "perlmutter": 280 / 4, # AMD EPYC 7453
}
GPU_POWER_PER_TASK = {
    "wildstyle": 250, # V100
    #"frontier": 500 / 2, # MI250x
    "frontier": 100, # estimated
    # "perlmutter": 250, # A100
    "perlmutter": 100, # based on real-world usage
}
CPU_PER_TASK = {
    "wildstyle": 32,
    "frontier": 7, # 64 total, 8 reserved
    "perlmutter": 16,
}
TASK_PER_NODE = {
    "wildstyle": 2,
    "frontier": 8,
    "perlmutter": 4,
}

BYTES_PER_REG = 4 # 32-bit registers

def unstack_subdict(df):
    result = pd.DataFrame(list(df.values), index=df.index)
    result.columns.name = df.name
    return result


def groupby_notinstance(obj):
    """
    Return an object suitable for ``describe``, ``sum``, etc.
    """
    return obj.groupby(level=RESULT_LEVELS[:-1])


def summarize_instances(obj):
    descr = groupby_notinstance(obj).describe()
    if isinstance(obj, pd.Series):
        return descr[['count', 'mean', 'std']]
    return descr.loc[Islc[:], Islc[:, ['count', 'mean', 'std']]]


def inverse_summary(summary):
    result = summary.copy()
    mean = 1/summary['mean']
    rel = summary['std'] / summary['mean']
    std = rel * mean
    return pd.DataFrame({'count': summary['count'], 'mean': mean, 'std': std})


def _calc_re(summary):
    return summary['std'] / summary['mean']


def calc_summary_ratio(num, denom):
    ratio = num['mean'] / denom['mean']
    std = ratio * np.hypot(_calc_re(denom), _calc_re(num))
    return pd.DataFrame({'mean': ratio, 'std' : std})


def calc_ratio(summary, num, denom):
    mean = summary['mean'].unstack()
    re = summary['std'].unstack() / mean
    ratio = mean[num] / mean[denom]
    std = ratio * np.hypot(re[denom], re[num])
    return pd.DataFrame({'mean': ratio, 'std' : std})


def get_cpugpu_ratio(summary):
    return calc_ratio(summary, 'cpu', 'gpu')


def calc_event_rate(results, summary=None):
    if summary is None:
        summary = results.summed
    ppe = summary[('num_primaries', 'mean')] / summary[('num_events', 'mean')]
    event_rate = inverse_summary(summary['avg_time_per_primary'])
    event_rate['mean'] /= ppe
    event_rate['std'] /= ppe
    return event_rate


class ProblemAbbreviator:
    def __init__(self):
        input_dir = Path(__file__).parent / "input"
        try:
            with open(input_dir / "problem-abbr.json") as f:
                self.geo_abbrev = json.load(f)
        except OSError as e:
            print("Failed to load problem abbreviations:", e)
            self.geo_abbrev = {}

    def __call__(self, inp):
        geo, *_ = inp['geometry_name'].partition('.')
        bits = [self.geo_abbrev[geo]]
        if inp.get('field') and any(inp['field']):
            bits.append('F$_\\mathrm{U}$')
        if not inp['enable_msc']:
            bits.append('M̃')

        return "".join(bits)

abbreviate_problem = ProblemAbbreviator()

def get_failed_problem_set(failures):
    failed_probs = failures.groupby(level='problem').any()
    return set(failed_probs.index[failed_probs])

def _to_nametuple(namelist):
    return tuple(namelist)

def _is_celersim(result):
    arch = result.index.get_level_values('arch')
    good_arch = (arch == 'cpu') | (arch == 'gpu') | (arch == 'gpu+sync')
    return pd.Series(good_arch, index=result.index)

def _calc_power(idx, system):
    power = pd.Series(index=idx)
    arch = power.index.get_level_values("arch")
    is_g4 = lambda arch_key: "g4" in arch_key
    is_gpu = {'gpu', 'gpu+g4', 'gpu+sync'}.__contains__
    is_cpu = {'cpu', 'cpu+g4', 'g4'}.__contains__

    gpu_power = GPU_POWER_PER_TASK.get(system)
    cpu_power = CPU_POWER_PER_TASK.get(system)

    if gpu_power is not None:
        # Real-world usage plus the CPU driving it
        frac_cpu = pd.Series(index=idx)
        frac_cpu[:] = 1.0 / CPU_PER_TASK[system]
        frac_cpu[arch.map(is_g4)] = 1.0
        power[arch.map(is_gpu)] = gpu_power + cpu_power * frac_cpu
    if cpu_power is not None:
        power[arch.map(is_cpu)] = cpu_power

    return power

class Analysis:
    def __init__(self, basedir):
        basedir = Path(basedir)
        with open(basedir / 'index.json') as f:
            index = {_to_nametuple(name): dirname
                     for (dirname, name) in json.load(f)}
        with open(basedir / 'summaries.json') as f:
            summaries = {_to_nametuple(v.pop('name')): v
                         for v in json.load(f).values()}

        input = pd.DataFrame([v.get('input') or {} for v in summaries.values()],
                             index=summaries.keys())
        input.index.names = RESULT_LEVELS[:-1]
        try:
            # Try renaming < v3
            geo_name = input.pop('geometry_filename')
        except KeyError:
            pass
        else:
            input['geometry_name'] = geo_name

        # Result MAY BE a string if all failed; otherwise, instance list.
        # Convert to a series first to check
        result = pd.Series([v['result'] for v in summaries.values()],
                            index=summaries.keys())
        all_invalid = result.map(lambda v: not isinstance(v, list))
        if np.any(all_invalid):
            # Drop "no successful problem" entries and save for later
            failed_idx = []
            failed_vals = []
            for i, v in result[all_invalid].items():
                failed_idx.append(i + (0,))
                failed_vals.append({'failed': v})
            failed = pd.Series(failed_vals, index=failed_idx)
            result = result[~all_invalid]
        else:
            failed = None

        # Convert from Series of list to DataFrame
        result = result.apply(pd.Series)
        # Stack instances
        result = result.stack()
        if failed is not None:
            result = pd.concat([result, failed])
        # Name the indices: prob, geo, arch, instance
        result.index.names = RESULT_LEVELS
        result = unstack_subdict(result)

        self.basedir = basedir
        self.index = index
        self.input = input
        self.result = result
        if 'failure' in result:
            self.valid = result['failure'].isna()
        else:
            self.valid = pd.Series(True, index=result.index)
        self.celersim = _is_celersim(result)
        self.version = self._load_version(summaries)

        self.failed_problems = get_failed_problem_set(~self.successful)
        self.failed_pga = ~self.successful.unstack('instance').any(axis=1)

        self.summed = summarize_instances(self.result[self.successful].dropna(how='all'))
        self.power = _calc_power(self.summed.index, self.system)

    def _load_version(self, summaries):
        versions = set()
        def _lstripv(text):
            if text.startswith("v"):
                return text[1:]
            return text

        for (k, s) in summaries.items():
            try:
                temp_sys = s["system"]
            except KeyError:
                pass
            else:
                if isinstance(temp_sys, list):
                    versions.update(_lstripv(ts["version"]) for ts in temp_sys)
                elif temp_sys is None:
                    print("WARNING: missing version for summary", k)
                    continue
                else:
                    versions.add(_lstripv(temp_sys["version"]))
        if len(versions) > 1:
            print("WARNING: multiple versions present in same summary:",
                  versions)
        elif not versions:
            print("WARNING: no version found")
        return " or ".join(versions)

    def load_results(self, name, instance):
        subdir = self.basedir / self.index[name]
        with open(subdir / f"{instance:d}.json") as f:
            result = json.load(f)

        try:
            version = result['system']['build']['version']
        except KeyError:
            version = self.version
        result['_metadata'] = {
            'version': version,
            'system': self.system,
            'name': name,
            'instance': instance,
        }
        return result

    def failures(self):
        try:
            failures = self.result['failure'].dropna()
        except KeyError:
            return
        if not failures.size:
            return
        return unstack_subdict(failures)

    def action_times(self):
        result = self.result

        return summarize_instances(
            unstack_subdict(result['action_times'][self.valid & self.celersim]))

    def active_hwm(self):
        hwm = self.result['active_hwm']
        return summarize_instances(unstack_subdict(hwm[self.valid & self.celersim]))

    def problems(self):
        """Get an ordered list of problem names.
        """
        skip = set()
        result = []
        for (p, *_) in self.index:
            if p not in skip:
                result.append(p)
                skip.add(p)
        return result

    def problem_to_abbr(self, problems=None, index=None):
        if problems is None:
            problems = self.problems()
        if index is None:
            failed_problems = self.failed_problems
        else:
            failed = self.failed_pga.groupby(index.names).any()
            try:
                failed_idx = failed[index]
            except KeyError as e:
                print(f"All problems seem to have failed: {e}")
                failed_idx = index
            failed_problems = get_failed_problem_set(failed_idx)

        result = {}
        for p in problems:
            inputs = self.input.xs(p, level='problem')
            inputs = inputs.dropna(subset='geometry_name')
            if len(inputs):
                value = abbreviate_problem(inputs.iloc[0])
                if p in failed_problems:
                    value += "*"
            else:
                print(f"WARNING: no inputs available for {p}")
                result[p] = "*"
            result[p] = value
        return result

    def __str__(self):
        return f"Analysis for Celeritas v{self.version} on {self.system}"

    @property
    def system(self):
        return self.basedir.name

    @property
    def cpu_per_task(self):
        return CPU_PER_TASK[self.system]

    @property
    def invalid(self):
        return ~self.valid

    @property
    def successful(self):
        # Protect against NaN for celer-g4 run
        unconverged_celersim = self.result['unconverged'] > 0
        return self.valid & ~(self.celersim & unconverged_celersim)

    def plot_results(self, ax, df):
        if df.empty:
            print("Warning: empty dataframe (will not plot results)")
            return []
        get_levels = df.index.get_level_values
        _idx_problems = set(get_levels('problem'))
        problems = [p for p in self.problems() if p in _idx_problems]
        problem_to_abbr = self.problem_to_abbr(problems, df.index)
        p_to_i = dict(zip(problems, itertools.count()))

        # One data point for each row, with geometries close to each other
        index = np.array([p_to_i[p] for p in get_levels('problem')], dtype=float)
        index += [(0.1 if g == 'orange' else -0.05) for g in get_levels('geo')]
        color = np.array([GEO_COLORS[g] for g in get_levels('geo')])

        if 'arch' in df.index.names:
            slc_mark = [(a.upper(), get_levels('arch') == a, ARCH_SHAPES[a])
                        for a in ARCH_SHAPES]
        else:
            slc_mark = [(None, slice(None), 's')]

        result = []
        for lab, slc, mark in slc_mark:
            temp_idx = index[slc]
            if not len(temp_idx):
                continue
            temp_sum = df.loc[slc]
            if np.all(np.isnan(temp_sum['mean'])):
                print(f"Warning: all NaN for {lab!s}")
                continue
            ax.errorbar(temp_idx, temp_sum['mean'], temp_sum['std'],
                        capsize=0, fmt='none', ecolor=(0.2,)*3)
            scat = ax.scatter(temp_idx, temp_sum['mean'], c=color[slc],
                              marker=mark, label=lab)
            result.append(scat)

        xax = ax.get_xaxis()
        xax.set_ticks(np.arange(len(problems)))
        xax.set_ticklabels(list(problem_to_abbr.values()), rotation=90)
        ax.set_axisbelow(True)
        return result

def CountGetter(out, stream):
    result = out['result']
    try:
        result = result['runner']
    except KeyError:
        # v0.2 and before #774
        def get_counts(key):
            return result[key][-1]
    else:
        def get_counts(key):
            return result[key][stream]

    return get_counts

def StepTimeGetter(out, stream):
    result = out['result']
    try:
        result = result['runner']
    except KeyError:
        # v0.2 and before #774
        time = result['time']
        def get_step_time():
            return time['steps']
    else:
        time = result['time']
        def get_step_time():
            return time['steps'][stream]

    return get_step_time

def _calc_scale_and_label(arr):
    dtype = type(arr[0])
    scale10 = np.floor(np.log10(np.max(arr)))
    if scale10 == 0:
        return (dtype(1), "")

    return (dtype(10**scale10), " [$\\times 10^{{{n}}}$]".format(n =
                                                                 int(scale10)))


def plot_counts(ax, out):
    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    lines = []
    def plot(ax, *args, **kwargs):
        line, = ax.plot(*args, **kwargs)
        lines.append(line)

    get_counts = CountGetter(out, stream=0)
    active = np.asarray(get_counts('active'), dtype=float)
    alive = np.asarray(get_counts('alive'), dtype=float)
    (norm, scale_label) = _calc_scale_and_label(active)

    plot(ax, active / norm, '-', color=(blue + (0.5,)), label='Active')
    plot(ax, alive / norm, '-', color=blue, label='Alive')
    ax.set_xlabel('Step iteration')
    ax.set_ylabel('Tracks' + scale_label, color=blue)
    ax.set_axisbelow(True)
    ax.grid()

    oax = ax.twinx()
    inits = np.asarray(get_counts('initializers'))
    plot(oax, inits / norm, '--', color=red, label='Queued')
    oax.axhline(out['input']['initializer_capacity'] / norm, linestyle='--',
                color=(red + (0.25,)))
    oax.set_ylabel("Initializers" + scale_label, color=red)

    max_init_idx = np.argmax(inits)
    max_init_val = inits[max_init_idx]
    text = re.sub(r'([-+.0-9]+)e\+?(-)?0*([0-9]+)', r'$\1\\times 10^{\2\3}$',
                  f'{max_init_val:.2g}')
    oax.annotate(text + ' queued', xy=(max_init_idx, max_init_val / norm), xycoords='data',
                 xytext=(30, 10), textcoords='offset points',
                 size='x-small',
                 bbox=dict(boxstyle="round,pad=.2", fc=(0.9, 0.9, 0.9, 0.8) , ec=(0.2,)*3),
                 arrowprops=dict(arrowstyle="->", ec=red, lw=1,
                                 connectionstyle="arc3,rad=0.2"))

    oax.spines['left'].set_color(blue)
    oax.spines['right'].set_color(red)
    oax.set_axisbelow(True)

    legend = ax.legend(lines, [l.get_label() for l in lines])

    return {
        'ax': ax,
        'oax': oax,
        'legend': legend,
    }


def plot_time_per_step(ax, outp, scale=1):
    get_counts = CountGetter(outp, stream=0)
    get_step_time = StepTimeGetter(outp, stream=0)

    active = np.asarray(get_counts('active'), dtype=float)
    stime = np.asarray(get_step_time()) * 1000
    def _xy(idx):
        return np.array([active[idx], stime[idx]])

    hq, hi = max((q, i) for (i, q) in enumerate(active))
    hq //= 2
    _, filling = min((q, i) for (i, q) in enumerate(active[:hi]) if q > hq)
    _, draining = min((q, i) for (i, q) in enumerate(active[hi:], start=hi) if q > hq)

    alpha = np.ones_like(active, dtype=float)
    alpha[active == outp['input']['num_track_slots']] = .05

    (norm, active_label) = _calc_scale_and_label(active)
    active /= norm

    lw = 0.5 * scale

    ax.set_axisbelow(True)
    ax.axhline(0, linestyle='-', lw=lw, color=(0.75,)*3, zorder=-2)
    ax.axvline(0, linestyle='-', lw=lw, color=(0.75,)*3, zorder=-2)

    ax.plot(active, stime, marker='', color="0.9", zorder=-1, lw=lw)
    scat = ax.scatter(active, stime, c=np.arange(len(active)), alpha=0.8, s=(3 * scale),
                      edgecolors='none')
    ax.annotate('All primaries active', xy=_xy(0), xycoords='data',
                xytext=(30, 0), textcoords='offset points',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=lw))
    ax.annotate('Filling', xy=(_xy(filling)), xycoords='data',
                xytext=(_xy(filling) * [1.1, 0.7]), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=lw))
    ax.annotate('Draining', xy=(_xy(draining)), xycoords='data',
                xytext=(_xy(draining) * [0.7, 1.3]), textcoords='data',
                size='x-small', color=".2",
                arrowprops=dict(arrowstyle="->", ec=".2", lw=lw))
    ax.set_xlabel("Number of active tracks" + active_label)
    ax.set_ylabel("Time per step [ms]")

    cb = ax.get_figure().colorbar(scat)
    cb.set_label('Step iteration')

    return {
        'ax': ax,
        'cb': cb,
    }


def plot_accum_time_inv(ax, outp):
    get_counts = CountGetter(outp, 0)
    get_step_time = StepTimeGetter(outp, 0)

    active = np.asarray(get_counts('active'), dtype=float)
    stime = np.asarray(get_step_time())

    accum_time = np.cumsum(stime)
    accum_steps = np.cumsum(active)

    blue = (.1, .1, .9)
    red = (.7, .1, .1)

    ax.plot(np.arange(len(accum_steps)), accum_time, marker='', color=blue)
    ax.set_xlabel("Step iteration")
    ax.set_ylabel("Total wall time [s]", color=blue)
    ax.grid()
    ax.set_axisbelow(True)

    oax = ax.twinx()
    oax.set_axisbelow(True)

    (norm, accum_steps_label) = _calc_scale_and_label(accum_steps)
    accum_steps /= norm

    oax.plot(np.arange(len(accum_steps)), accum_steps, linestyle='-.',
             marker='', color=(red + (0.5,)))
    oax.set_ylabel("Total number of steps" + accum_steps_label, color=red)
    oax.spines['left'].set_color(blue)
    oax.spines['right'].set_color(red)

    return {
        'ax': ax,
        'oax': oax,
    }


def annotate_metadata(obj, md, **kwargs):
    """Draw a little caption on a figure or axis with result metadata.
    """
    if isinstance(md, Analysis):
        s = f"v{md.version} on {md.system}"
    else:
        # Assume data from a single result
        name = "/".join(md['name'])
        s = f"{name}.{md['instance']}\nv{md['version']} on {md['system']}"

    try:
        # Assume obj is axes to get layout coordinates
        transform = obj.transAxes
    except AttributeError:
        # Hope obj is a figure
        transform = None

    text_kwargs = dict(va='bottom', ha='right',
        fontstyle='italic', color=(0.5,)*3, size='xx-small',
        transform=transform,
        zorder=-100
    )
    text_kwargs.update(kwargs)

    return obj.text(0.98, 0.02, s, **text_kwargs)


def make_failure_table(failures):
    if failures is None:
        failures = pd.DataFrame(dtype=object)
    flist = []
    idx = []
    for key, err in failures.iterrows():
        tp = err.get("type")

        if tp == "DebugError":
            f = PurePosixPath(err["file"])
            err["file"] = f.name
            text = "{which}: `{condition}` at `{file}:{line}`".format(**err)
        elif tp == "RuntimeError":
            f = PurePosixPath(err["file"])
            err["where"] = f.name
            if "line" in err and not np.isnan(err["line"]):
                err["where"] = "{}:{:d}".format(f.name, int(err["line"]))
            text = "{which} error: `{what}` at `{where}`".format(**err)
        elif "stdout" in err and isinstance(err["stdout"], list) and err["stdout"]:
            text = "`{}`".format(err["stdout"][-1])
        elif "stderr" in err and isinstance(err["stderr"], list) and err["stderr"]:
            for line in err["stderr"]:
                if line.startswith("celeritas: CUDA error"):
                    text = line
                    break
            else:
                # Use final line
                text = line
            text = "`{}`".format(line)
        else:
            text = "(unknown failure)"
        idx.append("{}/{}+{} ({:d})".format(*key))
        flist.append(text)
    return pd.Series(flist, index=idx, name="Failure", dtype=object)


def float_fmt_transform(digits):
    assert int(digits) == digits
    format = "{{:.{}f}}".format(digits).format
    def transform(val):
        if np.isnan(val):
            return "—"
        return format(val)
    return transform


def get_action_priority(k):
    try:
        return KERNEL_ORDERING[k]
    except KeyError:
        # Physics model
        return KernelCategory.PHYS


class AutoPercentFormatter:
    def __init__(self, threshold):
        """Give threshold in percent"""
        self.threshold = threshold

    def __call__(self, pctvalue):
        if pctvalue < self.threshold:
            return ""
        return "{:.0f}%".format(pctvalue)


class ActionFractionCalculator:
    def __init__(self, times):
        self.times = times.dropna()

        actions = list(self.times.index)
        ca = sorted([(get_action_priority(a), a) for a in actions])
        (category, actions) = zip(*ca)
        self.actions = np.array(actions)

        cat = np.array([int(c) for c in category])
        self.bounds = np.concatenate([cat[:-1] != cat[1:], [True]])
        self.labels = [KERNEL_CATEGORY_LABELS[c] for c in cat[self.bounds]]

    def __call__(self, arch):
        series = self.times[arch]
        inner = np.array([series[t] for t in self.actions])
        outer = np.cumsum(inner)[self.bounds]
        outer = np.concatenate([[outer[0]], np.diff(outer)])
        return (inner, outer)

class PiePlotter:
    def __init__(self, times):
        import matplotlib.pyplot as plt
        self._colormaps = plt.colormaps
        self.get_actions = ActionFractionCalculator(times)

        cmap = self._colormaps["tab20c"] # 5 groups of 4 shades
        def _get_color(color, shade = 0):
            shade = shade % 4
            return cmap((color % 5) * 4 + shade)

        self.outer_colors = _get_color(np.arange(len(KernelCategory)))

    @property
    def arch(self):
        return self.get_actions.times.columns

    @property
    def actions(self):
        return self.get_actions.actions

    @property
    def catlabels(self):
        return self.get_actions.labels

    def __call__(self, ax, arch):
        width = 0.3
        angle = 90.0 # start angle in degrees
        legend_thresh = 0.01

        (inner, outer) = self.get_actions(arch)

        # Plot outer ring (categories)
        (wedges, texts, autotexts) = ax.pie(
            outer,
            autopct=AutoPercentFormatter(5), pctdistance=0.85,
            radius=1, colors=self.outer_colors,
            wedgeprops=dict(width=width, edgecolor='w'), startangle=angle,
            textprops=dict(fontweight="bold")
        )
        outer_legend = ax.legend(wedges, self.catlabels,
                  loc="upper right",
                  bbox_to_anchor=(0.5, 0, 0.5, 1))
        ax.add_artist(outer_legend)

        # Determine kernels to higlight
        inner_frac = inner / np.sum(inner)
        inner_cmap = self._colormaps["plasma"]

        slc = inner_frac > legend_thresh
        num_inner = np.count_nonzero(slc)
        ascending_idx = np.argsort(np.argsort(inner_frac[slc]))
        inner_colors = np.zeros((inner.size, 4))
        inner_colors[slc, :] = inner_cmap(
                np.linspace(0.0, 1.0, num_inner)[ascending_idx])
        # Small kernels are grey
        inner_colors[~slc, :] = [0.5, 0.5, 0.5, 1.0]

        (wedges, texts, autotexts) = ax.pie(
            inner,
            radius=(1 - width), colors=inner_colors,
            autopct=AutoPercentFormatter(10), pctdistance=0.75,
            wedgeprops=dict(width=width, edgecolor='w'), startangle=angle,
            textprops=dict(size='x-small')
        )

        # Generate legend with all real kernels and one stand-in gray kernel
        wedges = np.array(wedges)
        actions = self.actions[slc].tolist()
        if not np.all(slc):
            wedges = wedges[slc].tolist() + [wedges[~slc][0]]
            actions += ["Fast kernel (<{:.0f}%)".format(legend_thresh * 100)]

        inner_legend = ax.legend(
            wedges, actions,
            loc="center",
            fontsize='xx-small'
        )
        ax.add_artist(inner_legend)



def _update_gpusync_tuple(t):
    if t[2] == "gpu+sync":
        t = t[:2] + ("gpu",) + t[3:]
    return t


def calc_geo_frac(analysis):
    action_times = analysis.result["action_times"][analysis.valid]
    _arch = action_times.index.get_level_values("arch")
    # Only get CPU and GPU+sync values
    _valid_arch = (_arch == "cpu") | (_arch == "gpu+sync")
    action_times = unstack_subdict(action_times[_valid_arch])
    # Get a mask for action categories
    _cat = np.vectorize(get_action_priority)(action_times.columns)
    geo_actions = ((_cat == KernelCategory.GEO)
                   | (_cat == KernelCategory.GP))

    # Replace "gpu+sync" with "gpu"
    geo_action_times = action_times.loc[:, geo_actions]
    geo_action_times.index = pd.MultiIndex.from_tuples(
        [_update_gpusync_tuple(r) for r in geo_action_times.index],
        names=geo_action_times.index.names
    )

    geo_frac = summarize_instances(
        geo_action_times.sum(axis=1) / analysis.result["total_time"]
    )
    geo_frac.dropna(inplace=True)
    return geo_frac


def dump_markdown(f, headers, table, alignment=None):
    widths = np.vectorize(len)(table)
    widths = np.concatenate([widths, [[len(t) for t in headers]]])
    col_widths = np.max(widths, axis=0)
    if alignment is None:
        alignment = ['<'] * len(headers)
    col_fmt = " | ".join(f"{{:{a}{c}}}" for (a, c) in zip(alignment, col_widths))
    fmt = ("| " + col_fmt + " |\n").format

    f.write(fmt(*headers))
    f.write(fmt(*["-"*w for w in col_widths]))
    for i in range(table.shape[0]):
        f.write(fmt(*table[i,:].tolist()))



def dump_rate(f, analysis, rate, units, prec=3):
    assert prec == int(prec)
    fmt = "{{:.{:d}f}} (±{{:.{:d}f}})".format(prec, prec).format
    rate = rate.loc[:, ['mean', 'std']]
    rate = rate.loc[rate.index.get_level_values('arch') != 'gpu+sync']
    rate = rate.dropna(how='all', axis=0).unstack('arch')

    _avail_arch = set(rate.columns.get_level_values(1))
    arches = [_a for _a in ARCH_SHAPES if _a in _avail_arch]
    tp_out = np.full((len(rate), 2 + len(arches)), "", dtype=object)
    _abbrev = analysis.problem_to_abbr()
    prev_prob = None
    for (i, ((prob, geo), row)) in enumerate(rate.iterrows()):
        if prob != prev_prob:
            abbr = _abbrev[prob]
            tp_out[i, 0] = f"{prob} [{abbr}]"
        prev_prob = prob
        tp_out[i, 1] = geo
        unstacked_row = row.unstack(0)
        for (j, a) in enumerate(arches, start=2):
            try:
                row2 = unstacked_row.loc[a]
            except KeyError:
                # Arch not applicable for this geometry
                continue
            if np.any(np.isnan(row2)):
                continue
            tp_out[i, j] = fmt(*row2)

    dump_markdown(f,
                  ["Problem", "Geometry"]
                  + [a.upper() + " " + units for a in arches],
                  tp_out,
                  alignment=("<<" + ">"*len(arches)))


def dump_speedup(f, results, prec=1):
    assert prec == int(prec)
    fmt = "{{:.{:d}f}}× (±{{:.{:d}f}})".format(prec, prec).format

    speedup = get_cpugpu_ratio(results.summed['total_time']).dropna(how='all', axis=0)
    speedup_out = np.full((len(speedup), 3), "", dtype=object)
    _abbrev = results.problem_to_abbr()
    prev_prob = None
    for (i, ((prob, geo), row)) in enumerate(speedup.iterrows()):
        if prob != prev_prob:
            abbr = _abbrev[prob]
            speedup_out[i, 0] = f"{prob} [{abbr}]"
        speedup_out[i, 1] = geo
        speedup_out[i, 2] = fmt(*row)
        prev_prob = prob

    dump_markdown(f,
                  ["Problem", "Geometry", "Speedup"],
                  speedup_out,
                  alignment="<<>")


def get_device_properties(analysis):
    problem = analysis.problems()[0]
    results = analysis.load_results((problem, "orange", "gpu"), 0)
    return results["system"]["device"]

def load_kernels(analysis, problem, geo):
    return analysis.load_results((problem, geo, "gpu"), 0)["system"]["kernels"]


def kernel_stats_dataframe(kernel_stats):
    values = []
    index = []
    for (instance, kernels) in kernel_stats.items():
        arch, _, geo = instance.partition("/")
        for (ki, stats) in enumerate(kernels):
            stats.pop("stack_size", None) # Unavailable with HIP
            row = list(stats.values())
            row.append(ki)
            values.append(row)
            name = stats["name"]
            if name == "extend-from-secondaries":
                # Fixup duplicate name
                name = f"{name}-{ki}"
            index.append((arch, geo, name))
    index=pd.MultiIndex.from_tuples(index, names=("arch", "geo", "name"))
    columns = pd.Index(list(stats.keys()) + ["kernel_index"])
    result = pd.DataFrame(values, index=index, columns=columns)
    del result["name"]
    del result["print_buffer_size"]
    result["register_mem"] = result["num_regs"] * BYTES_PER_REG
    return result



def main():
    # Generate table from wildstyle failures
    pass
