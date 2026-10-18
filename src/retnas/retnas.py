"""Command-line interface for the RTNS HRT artifact."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retnas import utils
from retnas.job import Job

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = {0: 'AlexNet', 1: 'InceptionV3', 3: 'ResNet50', 7: 'YOLOv2'}


def load_config(filename, overrides=None):
    with open(filename) as stream:
        entries = yaml.safe_load(stream)['config']
    entries.update(overrides or {})
    config = utils.Config(**entries)
    config.config_path = str(Path(filename).resolve())
    if not config.hrt:
        raise ValueError('This artifact requires hrt: true')
    if config.policy_type != 'gru':
        raise ValueError('The paper artifact uses policy_type: gru')
    if config.next_state_mode not in ('nonewjobs', 'hrt_only_all_periods'):
        raise ValueError('Use nonewjobs or hrt_only_all_periods')
    if config.hrt_step_mode not in ('all', 'deployment'):
        raise ValueError('Use hrt_step_mode: all or deployment')
    if not config.possible_jobs or not set(config.possible_jobs) <= WORKLOADS.keys():
        raise ValueError('Workload IDs must be selected from 0, 1, 3, 7')
    if not config.sa_types or not set(config.sa_types) <= {0, 1, 2, 3}:
        raise ValueError('Accelerator type IDs must be selected from 0, 1, 2, 3')
    if config.scheduling_period <= 0 or not config.hrt_customers:
        raise ValueError('A positive scheduling period and HRT task set are required')
    ids = set()
    for customer, job, period, deadline in config.hrt_customers:
        if customer in ids or job not in WORKLOADS or period <= 0 or deadline <= 0:
            raise ValueError('Invalid HRT customer ID, workload, period or deadline')
        if int(period) != period or period % config.scheduling_period:
            raise ValueError('Each HRT period must be an integer multiple of scheduling_period')
        ids.add(customer)
    for _, jobs in config.tenant_workloads:
        if not set(jobs) <= WORKLOADS.keys():
            raise ValueError('Unknown tenant workload ID')
    # All configured paths are relative to the artifact root, never the caller cwd.
    for key in ('data_csv_path', 'runs_path', 'wcet_schedule_path', 'wcet_model_path', 'test_model_path'):
        value = getattr(config, key, None)
        if value:
            path = Path(value)
            setattr(config, key, str(path if path.is_absolute() else ROOT / path))
    config.num_sas = len(config.sa_types)
    config.num_sas_types = len(set(config.sa_types))
    config.all_jobs = [None] * 8
    for job_id, name in WORKLOADS.items():
        frame = pd.read_csv(Path(config.data_csv_path) / f'{job_id:02d}.csv')
        costs = frame[[f'latency_{i}' for i in config.sa_types]].to_numpy(dtype=float)
        if not len(costs) or not np.isfinite(costs).all() or (costs <= 0).any():
            raise ValueError(f'Invalid execution costs for {name}')
        config.all_jobs[job_id] = Job(name, costs.min(axis=1).sum(), [tuple(row) for row in costs])
    config.jobs = [config.all_jobs[j] for j in config.possible_jobs]
    utils.set_seed(config.seed)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='YAML configuration file')
    modes = parser.add_mutually_exclusive_group()
    for flag, help_text in [('--compute-wcet', 'Generate and validate an HRT reference'),
                            ('--test', 'Evaluate a learned policy'),
                            ('--simulation', 'Evaluate a dispatch heuristic'),
                            ('--prema', 'Evaluate PREMA'), ('--cdmsa', 'Evaluate CD-MSA'),
                            ('--hardsoft', 'Evaluate Tang RC/TC/TCSD'),
                            ('--slackstealing', 'Evaluate SSML')]:
        modes.add_argument(flag, action='store_true', help=help_text)
    parser.add_argument('--reference-method', choices=['edf', 'model'], default='edf')
    parser.add_argument('--wcet-model', help='Override wcet_model_path')
    parser.add_argument('--model', help='Override test_model_path')
    parser.add_argument('--wcet-schedule', help='Override wcet_schedule_path (input/output)')
    parser.add_argument('--steps', type=int, help='Override evaluation steps')
    parser.add_argument('--algorithm', choices=['EDF', 'SJF', 'SSJF', 'FCFS', 'LCFS'])
    parser.add_argument('--hardsoft-mode', choices=['rc', 'tc', 'tcsd'])
    args = parser.parse_args()
    overrides = {k: v for k, v in dict(wcet_model_path=args.wcet_model,
                 test_model_path=args.model, wcet_schedule_path=args.wcet_schedule,
                 num_steps=args.steps, algorithm=args.algorithm, hardsoft_mode=args.hardsoft_mode).items() if v is not None}
    config = load_config(args.config, overrides)
    from retnas.environment import Environment
    from retnas.admission_control import save_feasible_scheduling, test_wcet_schedule_with_edf, test_wcet_schedule_with_model
    evaluating = any((args.test, args.simulation, args.prema, args.cdmsa, args.hardsoft, args.slackstealing))
    offline = args.compute_wcet or (not evaluating and config.next_state_mode == 'hrt_only_all_periods')
    expected = ('all', 'hrt_only_all_periods', False) if offline else (
        'deployment' if evaluating else 'all',
        'nonewjobs', True)
    actual = (config.hrt_step_mode, config.next_state_mode, config.hrt_flag_provided_to_the_network)
    if actual != expected:
        parser.error(f'This stage requires hrt_step_mode, next_state_mode, hrt_flag_provided_to_the_network = {expected}')
    if config.num_steps <= 0:
        parser.error('num_steps must be positive')
    if args.compute_wcet:
        env = Environment(config, seed=config.seed)
        if args.reference_method == 'model':
            if not config.wcet_model_path:
                parser.error('Model reference generation requires wcet_model_path or --wcet-model')
            rows, feasible = test_wcet_schedule_with_model(config, config.wcet_model_path, env)
        else:
            rows, feasible = test_wcet_schedule_with_edf(config, env)
        if not feasible:
            raise SystemExit('Admission rejected: no feasible reference found; existing file unchanged.')
        save_feasible_scheduling(rows, config.wcet_schedule_path, config, env)
        return
    # A validated reference is mandatory before training the deployment policy.
    if not offline:
        env = Environment(config, seed=config.seed)
        env._validate_hrt_dispatch_period()
        env._ensure_hrt_reference()
    if args.test:
        if not config.test_model_path:
            parser.error('Deployment requires test_model_path or --model')
        from retnas.testing import test
        test(config)
    elif args.simulation:
        from retnas.simulation import simulate
        simulate(config)
    elif args.prema:
        from retnas.prema import prema
        prema(config)
    elif args.cdmsa:
        from retnas.cdmsa import cdmsa
        cdmsa(config)
    elif args.hardsoft:
        from retnas.hardsoft import hardsoft
        hardsoft(config)
    elif args.slackstealing:
        from retnas.ssml import ssml
        ssml(config)
    else:
        if not (config.num_episodes > 0 and config.num_timesteps > config.push_to_buffer_after >= 0
                and config.batch_size > 0 and config.update_iteration > 0 and config.buffer_size > 0):
            parser.error('Training requires positive episodes, batches, updates, buffer and steps beyond warm-up')
        from retnas.training_par import train
        train(config)


if __name__ == '__main__':
    main()
