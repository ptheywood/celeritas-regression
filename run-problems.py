#!/usr/bin/env python3
# Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
- Loop over all problems
- Launch simultaneously on multiple cores (different seed per run!)
- Save overall times from all runs, and output from one run
- Catch failure message and save

Requires Python 3.7+.
"""

import asyncio
import itertools
import math
import json
import re
from pathlib import Path, PurePath
from pprint import pprint
from os import environ
import shutil
from signal import SIGINT, SIGTERM, SIGKILL
import subprocess
import sys
import time

try:
    import numpy as np
    from scipy.integrate import simpson
except ImportError as e:
    print("Can't load numpy/scipy:", e)

from summarize import inp_to_nametuple, summarize_all, exception_to_dict, get_num_events_and_primaries

systems = {}

class System:
    name = None
    build_dirs = {}
    num_jobs = None # Number of simultaneous jobs to run
    gpu_per_job = None
    cpu_per_job = None
    power_sample_interval = 1.0  # seconds

    def get_runtime_environ(self, inp):
        env = {}

        omp_threads = self.cpu_per_job
        if not inp['use_device']:
            # No device ative
            env['CELER_DISABLE_DEVICE'] = "1"
        elif inp['merge_events']:
            # Single stream: merge onto one CPU
            omp_threads = 1

        if not inp['_use_celeritas']:
            assert inp['_exe'] == "celer-g4"
            env['CELER_DISABLE'] = "1"

        if inp['_exe'] == "celer-g4":
            # Let Geant4 handle the threading
            omp_threads = 1
            env['G4FORCE_RUN_MANAGER_TYPE'] = "MT"
            env['G4FORCENUMBEROFTHREADS'] = str(self.cpu_per_job)
        else:
            assert inp['_exe'] == "celer-sim"

        env['OMP_NUM_THREADS'] = str(omp_threads)
        return env

    async def compute_gpu_energy(self, power_monitor_subprocess: asyncio.subprocess.Process):
        """
        Terminate the power monitor subprocess and compute the total energy consumed

        Returns:
            energy_wh: total energy consumed in watt-hours
            gpu_power: array of GPU power samples in watts (average power draw over 1s)
        """
        if power_monitor_subprocess is None:
            return 0, np.array([])

        power_monitor_subprocess.terminate()
        out, _ = await communicate_with_timeout(power_monitor_subprocess, 5)

        if power_monitor_subprocess.returncode:
            print(f"Power monitor exited with code {power_monitor_subprocess.returncode}")
            return 0, np.array([])

        lines = out.decode().splitlines()
        gpu_power = []
        for line in lines:
            if re.match('^ [0-9]+', line):
                line_cols = line.split()
                try:
                    power = float(line_cols[3])
                    sm_use = int(line_cols[6])
                except (IndexError, ValueError):
                    print(f"Failed to parse power sample: {line}")
                    continue
                else:
                    if sm_use > 40:  # 80 threshold lead to some jobs not capturing gpu usage on GH200, testing a lower threshold
                        gpu_power.append(power)
        gpu_power = np.array(gpu_power, dtype=np.float32)
        if gpu_power.size == 0:
            print("No GPU power samples found")
            return 0, np.array([])
        energy_ws = np.sum(np.multiply(gpu_power, self.power_sample_interval))
        print(f"{energy_ws} watts-secs {simpson(gpu_power)}")
        energy_wh = energy_ws / 3600
        return energy_wh, gpu_power

    def create_gpu_power_monitor_subprocess(self, inp):
        return None

    def create_celer_subprocess(self, inp):
        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")
        cmd = build / "bin" / inp['_exe']
        print("cmd: {cmd}")

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))
        if inp['use_device']:
            env['CUDA_VISIBLE_DEVICES'] = str(inp['_instance'])

        return asyncio.create_subprocess_exec(
            cmd, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    def get_monitoring_coro(self):
        return []

    def filter_problems(self, inputs):
        return inputs

class Wildstyle(System):
    build_dirs = {
        'orange': Path("/home/s3j/Code/celeritas/build-reldeb"),
        'vecgeom': Path("/home/s3j/Code/celeritas/build-reldeb-vecgeom"),
    }
    name = "wildstyle"
    num_jobs = 2
    gpu_per_job = 1
    cpu_per_job = 32


class Local(System):
    build_dirs = {
        "orange": Path("/Users/seth/Code/celeritas/build-reldeb"),
        "vecgeom": Path("/Users/seth/Code/celeritas/build-vecgeom-reldeb"),
    }
    name = "testing"
    num_jobs = 1
    gpu_per_job = 0
    cpu_per_job = 1


class Frontier(System):
    _CELER_ROOT = Path(environ['HOME']) / 'Code' / 'celeritas-frontier'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug'
    }
    name = "frontier"
    # FIXME: srun behavior has changed, 'main' srun counts as 1 so resource is
    # limited
    num_jobs = 7
    gpu_per_job = 1
    cpu_per_job = 7

    # NOTE: layout multi-gpu run
    # num_jobs = 4
    # gpu_per_job = 2
    # cpu_per_job = 14

    def get_runtime_environ(self, inp):
        env = super().get_runtime_environ(inp)
        env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] = "0"
        return env

    def create_celer_subprocess(self, inp):
        cmd = "srun"

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))

        args = [
            f"--cpus-per-task={self.cpu_per_job}",
            "--cpu-bind=threads,verbose",
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
            args.append("--gpu-bind=verbose,closest")
        else:
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            raise RuntimeError("Geometry type unavailable")

        exe = build / "bin" / inp['_exe']
        if not exe.exists():
            raise FileNotFoundError(exe)
        args.extend([str(exe), "-"])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    def filter_problems(self, inputs):
        return [i for i in inputs if i['_geometry'] != "vecgeom"]


class Perlmutter(Frontier):
    # System details:
    # https://portal.nersc.gov/cfs/mpccc/sleak/userdocs4/systems/perlmutter/system_details/#system-specification-phase-1
    _CELER_ROOT = Path(environ.get('CFS', '')) / 'atlas' / 'esseivaj' / 'devel' / 'celeritas'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug-novg',
        "vecgeom": _CELER_ROOT / 'build-ndebug',
    }
    name = "perlmutter"
    num_jobs = 4  # Nvidia A100 per node
    gpu_per_job = 1
    cpu_per_job = 16  # 1/4 of AMD EPYC with no hyperthreading
    power_sample_interval = 1.0  # seconds

    def create_gpu_power_monitor_subprocess(self, inp):
        """
        Create a subprocess that monitors GPU power usage using nvidia-smi

        Must be using ampere GPUs, each sample measures the average power draw over 1s

        Returns:
            subprocess: the power monitor subprocess
        """
        cmd = "nvidia-smi"
        args = ['dmon', '-i', str(inp['_instance']), '--select', 'pu', '--options', 'DT']
        return asyncio.create_subprocess_exec(cmd, *args, stdout=subprocess.PIPE)

    def create_celer_subprocess(self, inp):
        cmd = "srun"

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))
        if inp['geometry_file'].endswith('cms2018.gdml') \
           and inp['use_device'] and inp['_exe'] == 'celer-g4':
            env["CUDA_HEAP_SIZE"] = "10000000"
            env["CUDA_STACK_SIZE"] = "32000"
        # number of virtual CPUS
        n_cpus = int(2 * (64 / self.num_jobs))

        # 2 hyperthreads per core on Perlmutter
        assert self.cpu_per_job * 2 == n_cpus
        args = [
            f"--cpus-per-task={n_cpus}",
            "--ntasks=1",
            "--cpu-bind=verbose,cores"
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
            args.append("--gpu-bind=verbose,closest")
        else:
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            build = PurePath("nonexistent")

        exe = build / "bin" / inp['_exe']
        if not exe.exists():
            raise FileNotFoundError(exe)
        args.extend([str(exe), "-"])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    def filter_problems(self, inputs):
        return inputs


class Bede(System):
    #  User must be a member of the bdshe19 project on N8CIR Bede & have activated the spack env
    build_dirs = {
        'orange': Path("/nobackup/projects/bdshe19/aarch64/celeritas-project/celeritas/build-ndebug-novg"),
        'vecgeom': Path("/nobackup/projects/bdshe19/aarch64/celeritas-project/celeritas/build-ndebug"),
    }
    name = "bede"
    num_jobs = 1 # 1 GH200 480GB per gh partition node
    gpu_per_job = 1
    cpu_per_job = 72 # 72 CPU cores per GH200
    cpu_per_job_g4 = int(cpu_per_job // 2) # celer-g4 encounters cuda out of memory errors with high cpu per gpu count
    power_sample_interval = 1.0  # seconds

    def create_gpu_power_monitor_subprocess(self, inp):
        """
        Create a subprocess that monitors GPU power usage using nvidia-smi

        Must be using ampere+ GPUs, each sample measures the average power draw over 1s

        Returns:
            subprocess: the power monitor subprocess
        """
        cmd = "nvidia-smi"
        args = ['dmon', '-i', str(inp['_instance']), '--select', 'pu', '--options', 'DT']
        return asyncio.create_subprocess_exec(cmd, *args, stdout=subprocess.PIPE)

    def get_runtime_environ(self, inp):
        # Get the environment for the generic System
        env = super().get_runtime_environ(inp)

        # celer-g4 encounters cuda out of mem errors on GH200 with 72 CPU threads but a single 96GiB GPU.
        # use a different cpu count in this case.
        if inp['_exe'] == "celer-g4" and inp['use_device']:
            env['G4FORCENUMBEROFTHREADS'] = str(self.cpu_per_job_g4)

        # return the updated environment
        return env


class JadeARC(System):
    _CELER_ROOT = Path(environ['HOME']) / 'celeritas-project' / 'celeritas'
    build_dirs = {
        "orange": _CELER_ROOT / 'build-ndebug'
    }
    name = "jadearc"
    num_jobs = 1 # 8 MI300X per node
    gpu_per_job = 1
    cpu_per_job = 16 # 128 core per node
    power_sample_interval = 1.0  # seconds

    def get_runtime_environ(self, inp):
        env = super().get_runtime_environ(inp)
        # env["HSA_OVERRIDE_CPU_AFFINITY_DEBUG"] = "0"
        return env

    # def create_gpu_power_monitor_subprocess(self, inp):
    #     """
    #     Create a subprocess that monitors GPU power usage using amd-smi

    #     @todo - unclear if this is instantaneous or time-sampled power consumption from the very sparse amd docs

    #     Returns:
    #         subprocess: the power monitor subprocess
    #     """
    #     cmd = "amd-smi"
    #     args = ['monitor', '-g', str(inp['_instance']), '-ptum', '-w', str(self.power_sample_interval), "--csv"]
    #     return asyncio.create_subprocess_exec(cmd, *args, stdout=subprocess.PIPE)

    # async def compute_gpu_energy(self, power_monitor_subprocess: asyncio.subprocess.Process):
    #     """
    #     Terminate the power monitor subprocess and compute the total energy consumed

    #     Returns:
    #         energy_wh: total energy consumed in watt-hours
    #         gpu_power: array of GPU power samples in watts (average power draw over 1s)
    #     """
    #     if power_monitor_subprocess is None:
    #         return 0, np.array([])

    #     power_monitor_subprocess.terminate()
    #     out, _ = await communicate_with_timeout(power_monitor_subprocess, 5)

    #     if power_monitor_subprocess.returncode:
    #         print(f"Power monitor exited with code {power_monitor_subprocess.returncode}")
    #         return 0, np.array([])

    #     lines = out.decode().splitlines()
    #     gpu_power = []
    #     for line in lines:
    #         if re.match('^[0-9]+', line):
    #             line_cols = line.split(",")
    #             try:
    #                 power = float(line_cols[2])
    #                 gfx_use = int(line_cols[5])
    #             except (IndexError, ValueError):
    #                 print(f"Failed to parse power sample: {line}")
    #                 continue
    #             else:
    #                 if gfx_use > 0: # sometimes this seems to be stuck at 100 when gpu is not being used and idles at 200W...
    #                     gpu_power.append(power)
    #     gpu_power = np.array(gpu_power, dtype=np.float32)
    #     if gpu_power.size == 0:
    #         print("No GPU power samples found")
    #         return 0, np.array([])
    #     energy_ws = np.sum(np.multiply(gpu_power, self.power_sample_interval))
    #     print(f"{energy_ws} watts-secs {simpson(gpu_power)}")
    #     energy_wh = energy_ws / 3600
    #     return energy_wh, gpu_power

    def create_celer_subprocess(self, inp):
        cmd = "srun"

        env = dict(environ)
        env.update(self.get_runtime_environ(inp))

        args = [
            f"--cpus-per-task={self.cpu_per_job}",
        ]
        if inp['use_device']:
            args.append("--gpus-per-task=1")
            args.append("--gpu-bind=verbose,closest")
        else:
            args.append("--gpus=0")

        try:
            build = self.build_dirs[inp["_geometry"]]
        except KeyError:
            raise RuntimeError("Geometry type unavailable")

        exe = build / "bin" / inp['_exe']
        if not exe.exists():
            raise FileNotFoundError(exe)
        args.extend([str(exe), "-"])

        return asyncio.create_subprocess_exec(
            cmd, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    def filter_problems(self, inputs):
        return [i for i in inputs if i['_geometry'] != "vecgeom"]

class Awe(System):
    build_dirs = {
        'orange': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas/build-ndebug-novg"),
    }
    name = "awe"
    num_jobs = 1 # 2 * 7900XTX but only using 1
    gpu_per_job = 1 # 1 GPU per job
    cpu_per_job = 12 # 24 cores total, 2 gpus. 

    # no vecgeom builds on AMD
    def filter_problems(self, inputs):
        return [i for i in inputs if i['_geometry'] != "vecgeom"]

class Blackmass(System):
    build_dirs = {
        'orange': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas-cu126/build-ndebug-novg"),
        'vecgeom': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas-cu126/build-ndebug"),
    }
    name = "blackmass"
    num_jobs = 1 # 1 3080
    gpu_per_job = 1 # 1 GPU per job
    cpu_per_job = 8 # 16c 32t CPU

class BlackmassCU130(System):
    build_dirs = {
        'orange': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas/build-ndebug-novg"),
        'vecgeom': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas/build-ndebug"),
    }
    name = "blackmass_cu130"
    num_jobs = 1 # 1 3080
    gpu_per_job = 1 # 1 GPU per job
    cpu_per_job = 8 # 16c 32t CPU 

class Mavericks(System):
    build_dirs = {
        'orange': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas/build-ndebug-novg"),
        'vecgeom': Path("/home/ptheywood/code/shareing-r1-celeritas/celeritas/build-ndebug"),
    }
    name = "mavericks"
    num_jobs = 1 # 3 titan v, only uisng 1
    gpu_per_job = 1 # 1 GPU per job
    cpu_per_job = 6 # 6c12t (old) CPU (i7-5930K)

regression_dir = Path(__file__).parent
input_dir = regression_dir / "input"

base_input = {
    "_geometry": "orange",
    "_exe": "celer-sim",
    "_timeout": 600.0,
    "_use_celeritas": True,
    "use_device": False,
    "merge_events": False, # Separate streams
    "action_times": True,
    "write_track_counts": False,
    "initializer_capacity": 2**20,
    "num_track_slots": 2**12,
    "track_order": "unsorted",
    "max_steps": 2**21,
    "secondary_stack_factor": 2.0,
    "brem_combined": False,
    "physics_options": {
        "coulomb_scattering": False,
        "rayleigh_scattering": False,
        "eloss_fluctuation": True,
        "lpm": True,
        "em_bins_per_decade": 56,
        "physics": "em_basic",
        "msc": "urban",
    },
    "primary_options": {
        "seed": 0,
        "pdg": 11,
        "energy": 10000,  # 10 GeV
        "position": [0, 0, 0],
        "direction": {"distribution": "isotropic"},
        "primaries_per_event": 1300,  # 13 TeV
    },
}

use_geant = {
    "_exe": "celer-g4",
    "physics_list": "geant_physics_list",
    "sd_type": "none",
    "output_file": "-",
}

pure_geant = {
    "_geometry": "geant4",
    "_use_celeritas": False,
    "physics_options": {
        # Since geant4 uses splines it doesn't need as many points
        "em_bins_per_decade": 14,
    }
}

no_msc = {"physics_options": {"msc": "none"}}
use_field = {
    "field": [0.0, 0.0, 1.0], # units: [T]
    "field_options": {"max_substeps": 1000},
}

use_gpu = {
    "action_times": False,
    "use_device": True,
    "merge_events": True,
    "write_track_counts": True,
    "num_track_slots": 2**20,
    "max_steps": 2**15,
    "initializer_capacity": 2**26,
}

use_gpu_streams = use_gpu.copy()
use_gpu_streams.update({
    "merge_events": False,
    "num_track_slots": 2**18,
    "max_steps": 2**15,
    "initializer_capacity": 2**24,
})

use_sync = {
    "sync": True, # Deprecated
    "action_times": True,
}

testem15 = {
    "geometry_file": "testem15.gdml",
    "primary_options": {
        "pdg": [11, -11],
    },
}

testem3 = {
    "geometry_file": "testem3-flat.gdml",
    "primary_options": {
        "position": [-22, 0, 0],
        "direction": [1, 0, 0],
        "_units": "cgs",
    },
    "environ": {
        "ORANGE_FORCE_INPUT": str(input_dir / "testem3-flat-manual.org.json")
    },
}

testem3_composite = {
    "geometry_file": "testem3-composite.gdml",
    "primary_options": testem3["primary_options"],
    "environ": {
        "ORANGE_MAX_FACE_INTERSECT": "12,12",
    },
}

testem3_expanded = {
    "geometry_file": "testem3-expanded.gdml",
    "primary_options": testem3["primary_options"],
    "environ": testem3_composite["environ"],
}

_tilecal_angle = 76 * (2 * math.pi / 360)
tilecal = {
    "geometry_file": "atlas-tilecal.gdml",
    "primary_options": {
        "position": [229.801, 0, 0],
        "direction": [math.sin(_tilecal_angle), 0, math.cos(_tilecal_angle)],
    },
}

hgcal = {
    "geometry_file": "cms-hgcal.gdml",
    "primary_options": {
        "position": [0, 0, -899.999],
        "direction": [0, 0, 1],
    },
}

full_cms = {
    "_geometry": "vecgeom",
    "geometry_file": "cms2018.gdml",
    "cuda_stack_size": 8192,
}

use_vecgeom = {"_geometry": "vecgeom"}

# List of list of setting dictionaries
problems = [
    [testem15, no_msc],
    [testem15, no_msc, use_field],
    [testem15, use_field],
    [testem15, use_field, use_vecgeom],
    [testem3, no_msc],
    [testem3, no_msc, use_vecgeom],
    [testem3, no_msc, use_field],
    [testem3],
    [testem3, use_field],
    [testem3, use_field, use_vecgeom],
    [testem3_composite],
    [testem3_composite, use_vecgeom],
    [testem3_composite, use_field],
    [testem3_composite, use_field, use_vecgeom],
    [testem3_expanded, use_field],
    [testem3_expanded, use_field, use_vecgeom],
    [tilecal, no_msc],
    [tilecal, no_msc, use_vecgeom],
    [hgcal, no_msc],
    [hgcal, no_msc, use_vecgeom],
    [full_cms, no_msc],
    [full_cms, use_field],
]

# Run again with sync on for detailed GPU timing
sync_problems = [
    [testem15, no_msc, use_field],
    [testem15, no_msc, use_field, use_vecgeom],
    [testem3, use_field],
    [testem3, use_field, use_vecgeom],
    [testem3_composite, use_field],
    [testem3_composite, use_field, use_vecgeom],
    [full_cms, use_field],
]

def recurse_updated(d, other):
    result = d.copy()
    result.update(other)
    for k, v in result.items():
        if isinstance(v, dict):
            try:
                orig = d[k]
            except KeyError:
                v = result[k]
            else:
                v = recurse_updated(orig, result[k])
            result[k] = v
    return result


def build_input(problem_dicts):
    """Construct an input dictionary by merging inputs.

    Later entries override earlier entries.
    """
    # Combine all dictionaries
    inp = base_input.copy()
    for d in problem_dicts:
        inp = recurse_updated(inp, d)

    # Make paths absolute
    for k in inp:
        if k.endswith('_file'):
            v = inp[k]
            if v != '-':
                inp[k] = str(input_dir / v)

    # Save name and output directory
    inp["_name"] = name = inp_to_nametuple(inp)
    inp["_outdir"] = "-".join(name)

    # Update 'maximum events' input entry
    (inp["max_events"], _) = get_num_events_and_primaries(inp)

    return inp


def build_instance(inp, instance):
    inp = inp.copy()
    inp["_instance"] = instance
    inp["seed"] = 20220904 + instance
    return inp

def patch_input(system, inp):
    # patch num_track_slots for celer-sim
    if inp['_exe'] == "celer-sim" and not inp['merge_events']:
        inp['num_track_slots'] *= system.cpu_per_job
        inp['initializer_capacity'] *= system.cpu_per_job


async def communicate_with_timeout(proc, interrupt, terminate=5.0, kill=1.0, input=None):
    """Interrupt, then terminate, then kill a process if it doesn't
    communicate.
    """
    try:
        result = await asyncio.wait_for(
            proc.communicate(input),
            timeout=interrupt)
    except asyncio.TimeoutError:
        print(f"Timed out after {interrupt} seconds: sending interrupt")
        proc.send_signal(SIGINT)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=terminate)
    except asyncio.TimeoutError:
        print(f"Timed out *AGAIN* after {terminate} seconds")
        proc.send_signal(SIGTERM)
    else:
        return result

    try:
        result = await asyncio.wait_for(proc.communicate(),
                    timeout=kill)
    except asyncio.TimeoutError:
        print(f"Set phasers to kill after {kill} seconds")
        proc.send_signal(SIGKILL)
    else:
        return result

    print("Awaiting communication")
    result = await proc.communicate()
    return result


async def run_celeritas(system: System, results_dir, inp):
    instance = inp['_instance']

    patch_input(system, inp)

    proc_gpu_power = None
    if inp["use_device"]:
        try:
            power_subprocess = system.create_gpu_power_monitor_subprocess(inp)
            if power_subprocess:
                proc_gpu_power = await power_subprocess
        except Exception as e:
            print("Problem creating power subprocess:", e)

    outdir = results_dir / inp['_outdir']
    outdir.mkdir(exist_ok=True)

    try:
        proc = await system.create_celer_subprocess(inp)
    except Exception as e:
        print("Problem creating subprocess:", e)
        with open(outdir / f"{instance:d}.inp.json", "w") as f:
            json.dump(inp, f, indent=0, sort_keys=True)
        return exception_to_dict(e, context="creating subprocess")

    start = time.monotonic()
    print(f"{instance}: awaiting communcation")
    failed = False
    out, err = await communicate_with_timeout(proc,
        input=json.dumps(inp).encode(),
        interrupt=inp['_timeout']
    )

    run_delta = time.monotonic() - start
    start = time.monotonic()

    if proc_gpu_power:
        energy_wh, gpu_power = await system.compute_gpu_energy(proc_gpu_power)

    try:
        result = json.loads(out)
    except json.decoder.JSONDecodeError as e:
        print(f"{instance}: failed to decode JSON")
        failed = True
        result = {
            'stdout': out.decode().splitlines(),
        }

    # if json decoding of result failed, and proc_gpu_power was enabled, jobs would fail here? as "result" was not a key
    if proc_gpu_power and "result" in result:
    # if proc_gpu_power:
        res = result["result"]
        if "runner" in res:
            res = res["runner"]
        # res['gpu_energy_wh'] = energy_wh
        # json.dump error 'Object of type float32 is not JSON serializable)
        res['gpu_energy_wh'] = float(energy_wh)
        res['gpu_power'] = gpu_power.tolist()

    if proc.returncode:
        print(f"{instance}: exit code {proc.returncode}")
        failed = True
        result['stderr'] = err.decode().splitlines()

    # Copy special inputs to output for later processing
    result.setdefault('input', {}).update(
        {k: v for k,v in inp.items() if k.startswith('_')}
    )

    try:
        with open(outdir / f"{instance:d}.json", "w") as f:
            json.dump(result, f, indent=0, sort_keys=True)
    except Exception as e:
        print(f"{instance}: failed to output:", repr(e))
        failed = True

    if proc.returncode:
        # Write input to reproduce later
        with open(outdir / f"{instance:d}.inp.json", "w") as f:
            json.dump(inp, f, indent=0, sort_keys=True)

    # Echo stderr
    with open(outdir / f"{instance:d}.err", "wb") as f:
        f.write(err)

    if not failed:
        process_delta = time.monotonic() - start
        print(f"{instance}: success (launch {run_delta:.1f}, process {process_delta:.1f})")

    return result

async def main():
    try:
        sysname = sys.argv[1]
    except IndexError:
        Sys = Local
    else:
        # TODO: use metaclass to build this list automatically
        _systems = {S.name: S for S in [Frontier, Perlmutter, Wildstyle, Bede, JadeARC, Awe, Blackmass, BlackmassCU130, Mavericks]}
        Sys = _systems[sysname]

    system = Sys()
    system.build_dirs['geant4'] = system.build_dirs['orange']

    # Copy build files
    buildfile_dir = regression_dir / 'build-files' / system.name
    buildfile_dir.mkdir(exist_ok=True)
    for k, v in system.build_dirs.items():
        dst = buildfile_dir / (k + '.txt')
        try:
            shutil.copyfile(v / 'CMakeCache.txt', dst)
        except OSError:
            dst.unlink(missing_ok=True)

    results_dir = regression_dir / 'results' / system.name
    results_dir.mkdir(exist_ok=True)

    device_mods = []
    if system.gpu_per_job:
        device_mods.append([use_gpu])
        device_mods.append([use_gpu_streams, use_geant])
    device_mods.append([]) # CPU celeritas
    device_mods.append([use_geant]) # CPU celeritas through celer-g4
    device_mods.append([use_geant, pure_geant]) # CPU geant4 for reference

    # Set number of events based on number of CPUs
    base_inputs = [
        base_input,
        {"primary_options": {"num_events": system.cpu_per_job}},
    ]

    inputs = [build_input(base_inputs + p + d)
              for p, d in itertools.product(problems, device_mods)]
    if system.gpu_per_job:
        inputs += [build_input(base_inputs + p + [use_gpu, use_sync])
                   for p in sync_problems]

    inputs = system.filter_problems(inputs)
    with open(results_dir / "index.json", "w") as f:
        json.dump([(inp['_outdir'], inp['_name'])
                   for inp in inputs], f, indent=0)

    summaries = {}
    allstart = time.monotonic()
    _num_inputs = len(inputs)
    for (i, inp) in enumerate(inputs, start=1):
        print("="*79)
        name = inp['_outdir']
        print(f"Running problem {i} of {_num_inputs}: {name}...")
        start = time.monotonic()
        tasks = [run_celeritas(system, results_dir, build_instance(inp, i))
                 for i in range(system.num_jobs)]
        if not summaries:
            # Only print monitoring for first instance
            tasks.extend(system.get_monitoring_coro())
        result = await asyncio.gather(*tasks)

        # Ignore results from monitoring tasks
        result = result[:system.num_jobs]

        try:
            summaries[name] = summary = summarize_all(result)
        except Exception as e:
            print("*"*79)
            print("FAILED input:")
            pprint(inp)
            print("*"*79)
            pprint(result)
            print("Failed to summarize result above")
            raise
        summary['name'] = inp['_name'] # name tuple
        pprint(summary)
        alldelta = time.monotonic() - allstart
        delta = time.monotonic() - start
        print(f"Elapsed time for {name}: {delta:.1f} (total: {alldelta:.0f})")

    with open(results_dir / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=1, sort_keys=True)
    print(f"Wrote summaries to {results_dir}")

asyncio.run(main())
