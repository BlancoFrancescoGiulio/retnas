"""
Retnas HRT - Admission Control Module

Questo modulo gestisce:
1. Calcolo della schedulazione worst-case (WCET) per i job HRT
2. Admission control per nuovi customer HRT
3. Salvataggio/caricamento della schedulazione feasible da file
"""
import yaml
import numpy as np
import torch
import math
from retnas.environment import Environment
from retnas.job import SubJob
from retnas.networks import SchedulerNetworkActorRNN

def compute_lcm_periods(hrt_customers):
    """
    Calcola il minimo comune multiplo (LCM) di tutti i periodi HRT.

    Args:
        hrt_customers: Lista di HRT customers nella forma [customer_id, job_id, period, deadline]

    Returns:
        lcm_period: Il LCM di tutti i periodi
    """
    if not hrt_customers:
        return 0
    periods = [int(hrt_cust[2]) for hrt_cust in hrt_customers]

    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    result = periods[0]
    for period in periods[1:]:
        result = lcm(result, period)
    return result

def test_wcet_schedule_with_model(config, model_path, env=None, action_noise=None, rng=None):
    """Generate an independent reference, restoring caller settings on failure."""
    if env is None:
        raise ValueError('the WCET generator requires an Environment')
    original = (env.next_state_mode, env.hrt_step_mode, env.hrt_flag_provided_to_the_network, env._hrt_arrival_horizon)
    try:
        return _test_wcet_schedule_with_model(config, model_path, env, action_noise, rng)
    finally:
        (env.next_state_mode, env.hrt_step_mode, env.hrt_flag_provided_to_the_network, env._hrt_arrival_horizon) = original

def _test_wcet_schedule_with_model(config, model_path, env=None, action_noise=None, rng=None):
    """
    Testa la schedulazione worst-case usando un modello Retnas addestrato.
    Genera tutte le istanze HRT necessarie per il periodo LCM e simula la schedulazione
    per questo periodo usando il modello fornito.

    Args:
        config: Configurazione del sistema
        model_path: Path al modello actor da caricare
        env: Environment (opzionale, se None viene creato uno nuovo)

    Args (optional):
        action_noise: Standard deviation of a controlled perturbation applied
            only to the SA-assignment outputs. ``None`` disables perturbations.
        rng: NumPy random generator used for ``action_noise``.

    Returns:
        schedule: Lista di dict con schedulazione per ogni subjob HRT
        is_feasible: True se SLAST HRT == 100%, False altrimenti
    """
    print(f'[INFO] Testing WCET schedule with model: {model_path}')
    print('=' * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if env is None:
        print('Must pass the env to test_wcet_schedule_with_model, otherwise we cannot control the generation of HRT jobs! Exiting.')
        quit(1)
    lcm_period = compute_lcm_periods(env.hrt_customers)
    scheduling_period = config.scheduling_period
    if scheduling_period != env.scheduling_period:
        raise ValueError('config and Environment must use the same scheduling_period')
    env._validate_hrt_dispatch_period()
    num_periods = int(math.ceil(lcm_period / scheduling_period))
    print(f'[INFO] LCM of HRT periods: {lcm_period} ns')
    print(f'[INFO] Scheduling period: {scheduling_period} ns')
    print(f'[INFO] Number of scheduling periods to simulate: {num_periods}')
    print(f'[INFO] Total simulation time: {num_periods * scheduling_period} ns\n')
    num_layers = config.num_layers
    policy_type = getattr(config, 'policy_type', 'gru')
    actor_input_size = 4 + config.num_sas
    if config.hrt and config.hrt_flag_provided_to_the_network:
        actor_input_size += 1
    if getattr(config, 'num_tenants', 0) > 0 and getattr(config, 'enable_balancing_reward', False):
        actor_input_size += 2
    actor_output_size = 1 + config.num_sas
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as error:
        print(f'[ERROR] Failed to load model {model_path}: {error}')
        return ([], False)
    input_weight = next((checkpoint[key] for key in ('core.weight_ih_l0', 'fc0.weight') if key in checkpoint), None)
    if input_weight is not None and input_weight.shape[1] == actor_input_size - 1 and config.hrt and config.hrt_flag_provided_to_the_network:
        actor_input_size -= 1
        env.hrt_flag_provided_to_the_network = False
        print('[INFO] WCET checkpoint omits the HRT flag; adapting offline input only.')
    actor = SchedulerNetworkActorRNN(actor_input_size, config.actor_hidden_size, actor_output_size, num_layers=num_layers, use_packing=config.use_packing, type=policy_type).to(device)
    try:
        actor.load_state_dict(checkpoint)
        actor.eval()
    except Exception as e:
        print(f'[ERROR] Failed to load model {model_path}: {e}')
        return ([], False)
    env.reset(clear=True, no_new_jobs=True)
    env.next_state_mode = 'hrt_only_all_periods'
    env.hrt_step_mode = 'all'
    env._hrt_arrival_horizon = lcm_period
    env.new_jobs_arrival(hrt_only=True, hrt_all_periods=True)
    state = []
    hrt_unfeasibility_count = 0
    all_scheduled_sjs = [[] for _ in range(num_periods)]
    optional_schedule_keys = set()
    for period_idx in range(num_periods):
        print(f'  [Period {period_idx}] Generated {len(env.rq)} HRT subjobs')
        env.sort_ready_queue()
        state = env.encode_state()
        if len(env.rq) == 0:
            print(f'[Period {period_idx}] No queued work; waiting for periodic releases.')
        with torch.no_grad():
            input_state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            num_sjs = torch.tensor(input_state.shape[1])
            action = actor(input_state, num_sjs).squeeze(0).cpu().data.numpy()
        if action_noise is not None and action.shape[1] > 1:
            if rng is None:
                rng = np.random.default_rng()
            action[:, 1:] += rng.normal(loc=0.0, scale=float(action_noise), size=action[:, 1:].shape)
        rq_before = env.rq.copy()
        print(f'CURRENT TIME 1 {env.current_time}')
        (state, _, _, _, hrt_feasible) = env.step(action)
        print(f'CURRENT TIME 2 {env.current_time}')
        slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
        print(f'[Period {period_idx}/{num_periods}] SLAST_HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
        if not hrt_feasible:
            hrt_unfeasibility_count += 1
        for sj in rq_before:
            has_times = sj.start_time is not None and sj.finish_time is not None
            key = (sj.job_instance_id, sj.job_id, sj.layer_id)
            should_record = has_times and sj.start_time < env.current_time
            if key in optional_schedule_keys:
                should_record = False
            if should_record:
                optional_schedule_keys.add(key)
                all_scheduled_sjs[period_idx].append({'job_instance_id': sj.job_instance_id, 'job_id': sj.job_id, 'layer_id': sj.layer_id, 'sa_id': int(sj.sa_id), 'start_time': float(sj.start_time), 'finish_time': float(sj.finish_time), 'absolute_deadline': float(sj.absolute_deadline), 'arrival_time': float(sj.arrival_time), 'hrt_customer_id': sj.hrt_customer_id})
        if (period_idx + 1) % 100 == 0 or period_idx == num_periods - 1:
            slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
            print(f'[Period {period_idx + 1}/{num_periods}] SLAST_HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
    is_feasible = slast_hrt >= 0.9999 and hrt_unfeasibility_count == 0
    if is_feasible:
        from retnas.safe_prefix import PeriodicReference, ReferenceScheduleError
        recorded = [item for period_schedule in all_scheduled_sjs for item in period_schedule]
        try:
            PeriodicReference(recorded, env.hrt_customers, env.jobs, env.num_sas)
        except ReferenceScheduleError as error:
            print(f'[ERROR] WCET reference validation failed: {error}')
            is_feasible = False
    print(f"\n{'=' * 60}")
    print(f'FINAL RESULTS:')
    print(f'  SLAST_HRT: {slast_hrt * 100:.4f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    print(f'  Deadline misses HRT: {env.deadline_misses_hrt}')
    print(f'  Deadline misses BE: {env.deadline_misses_be}')
    print(f'  Unfeasible HRT scheduling periods: {hrt_unfeasibility_count} over {num_periods}')
    print(f'  Total scheduled subjobs: {sum((len(items) for items in all_scheduled_sjs))}')
    print(f"  Feasible: {('YES' if is_feasible else 'NO')}")
    print(f"{'=' * 60}\n")
    all_scheduled_sjs_flat = []
    for sj_entries_in_a_single_period in all_scheduled_sjs:
        all_scheduled_sjs_flat.extend(sj_entries_in_a_single_period)
    return (all_scheduled_sjs_flat, is_feasible)

def test_wcet_schedule_with_edf(config, env=None):
    """Construct EDF reservations without consulting the deployment fallback."""
    if env is None:
        env = Environment(config, seed=config.seed)
    original = (env.next_state_mode, env.hrt_step_mode)
    (env.next_state_mode, env.hrt_step_mode) = ('noarrivals', 'all')
    try:
        (rows, feasible) = _test_wcet_schedule_with_edf(config, env)
        if feasible:
            from retnas.safe_prefix import PeriodicReference
            PeriodicReference(rows, env.hrt_customers, env.jobs, env.num_sas)
        return (rows, feasible)
    finally:
        (env.next_state_mode, env.hrt_step_mode) = original

def _test_wcet_schedule_with_edf(config, env=None):
    """
    Calcola la schedulazione WCET usando EDF per un periodo pari al LCM
    di tutti i periodi HRT. Genera tutte le istanze HRT necessarie e verifica
    che raggiungano 100% SLAST HRT.

    Usa l'algoritmo EDF (Earliest Deadline First) con heuristic SA assignment
    come in simulation.py.

    Args:
        config: Configurazione del sistema
        env: Environment (opzionale, se None viene creato uno nuovo)

    Returns:
        schedule: Lista di dict con schedulazione per ogni subjob HRT
                  Formato: [{job_instance_id, job_id, layer_id, sa_id, start_time, finish_time, absolute_deadline}, ...]
        is_feasible: True se SLAST HRT == 100%, False altrimenti
    """
    if env is None:
        env = Environment(config, seed=config.seed)
    print('=' * 60)
    print('COMPUTING WCET SCHEDULE WITH EDF FOR LCM PERIOD')
    print('=' * 60)
    lcm_period = compute_lcm_periods(env.hrt_customers)
    scheduling_period = config.scheduling_period
    num_periods = int(math.ceil(lcm_period / scheduling_period))
    print(f'\n[INFO] LCM of HRT periods: {lcm_period} ns')
    print(f'[INFO] Scheduling period: {scheduling_period} ns')
    print(f'[INFO] Number of scheduling periods to simulate: {num_periods}')
    print(f'[INFO] Total simulation time: {num_periods * scheduling_period} ns\n')
    env.reset(clear=True, no_new_jobs=True)
    env.use_sa_assignement_heuristic = True
    env.prepend_sa_remaining_time = False
    print('[INFO] Generating HRT job instances for LCM period...')
    for hrt_cust in env.hrt_customers:
        (hrt_customer_id, job_id, period, hrt_deadline) = hrt_cust
        num_instances = lcm_period // period
        print(f'  [Customer {hrt_customer_id}] Job {job_id}: {num_instances} instances over LCM period')
        for instance in range(num_instances):
            arrival_time = instance * period
            adjusted_deadline = hrt_deadline - scheduling_period
            env.new_job_arrival(arrival_time=arrival_time, tenant_id=None, is_hrt=True, hrt_job_id=job_id, hrt_deadline=adjusted_deadline, hrt_customer_id=hrt_customer_id)
    if len(env.rq) == 0:
        print('[WARN] No HRT jobs to schedule')
        return ([], True)
    print(f'[INFO] Total HRT subjobs to schedule: {len(env.rq)}\n')
    schedule = []
    all_scheduled_subjobs = []
    hrt_unfeasibility_count = 0
    for period_idx in range(num_periods):
        if len(env.rq) == 0:
            print(f'[INFO] All jobs completed at period {period_idx}/{num_periods}')
            if period_idx <= num_periods:
                print(f'[INFO] [IMPORTANT] {period_idx}<={num_periods} --> this means that the period can be repeated indefinitely! Very good.')
            else:
                print(f'[ERROR] [IMPORTANT] {period_idx}>{num_periods} --> this means that the scheduling is NOT feasible.')
                hrt_unfeasibility_count += 1
                quit(1)
            break
        action = []
        for sj in env.rq:
            priority = sj.absolute_deadline
            sa_choice_feat = np.random.rand(config.num_sas)
            action.append(np.concatenate(([priority], sa_choice_feat)))
        action = np.array(action)
        rq_before = env.rq.copy()
        (_, _, _, _, hrt_feasible) = env.step(action)
        if not hrt_feasible:
            hrt_unfeasibility_count += 1
        for sj in rq_before:
            if sj.is_hrt and sj.start_time is not None and (sj.finish_time is not None):
                already_added = any((item['job_instance_id'] == sj.job_instance_id and item['layer_id'] == sj.layer_id for item in all_scheduled_subjobs))
                if not already_added:
                    job_instance_id = sj.job_instance_id
                    layer_id = sj.layer_id
                    sa = int(sj.sa_id)
                    start_t = sj.start_time
                    finish_t = sj.finish_time
                    all_scheduled_subjobs.append({'job_instance_id': job_instance_id, 'job_id': sj.job_id, 'layer_id': layer_id, 'sa_id': sa, 'start_time': float(start_t), 'finish_time': float(finish_t), 'absolute_deadline': float(sj.absolute_deadline), 'arrival_time': float(sj.arrival_time), 'hrt_customer_id': sj.hrt_customer_id})
        if (period_idx + 1) % 100 == 0 or period_idx == num_periods - 1:
            slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
            print(f'[Period {period_idx + 1}/{num_periods}] SLAST_HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    slast_hrt = env.deadline_hits_hrt / env.total_arrived_jobs_hrt if env.total_arrived_jobs_hrt > 0 else 0
    is_feasible = slast_hrt >= 0.9999
    print(f"\n{'=' * 60}")
    print(f'FINAL RESULTS:')
    print(f'  SLAST_HRT: {slast_hrt * 100:.4f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    print(f'  Deadline misses HRT: {env.deadline_misses_hrt}')
    print(f'  Deadline misses BE: {env.deadline_misses_be}')
    print(f'  Unfeasible HRT scheduling periods: {hrt_unfeasibility_count} over {num_periods}')
    print(f'  Total scheduled subjobs: {len(all_scheduled_subjobs)}')
    print(f"  Feasible: {('YES' if is_feasible else 'NO')}")
    print(f"{'=' * 60}\n")
    return (all_scheduled_subjobs, is_feasible)


def save_feasible_scheduling(schedule, filename='feasible_scheduling.yaml', config=None, env=None):
    """Validate both original and slack-shifted reservations before atomic publication."""
    import os
    import tempfile
    from pathlib import Path
    from retnas.safe_prefix import PeriodicReference, ReferenceScheduleError
    if env is None or not schedule:
        raise ReferenceScheduleError('A nonempty schedule and validation environment are required')
    PeriodicReference(schedule, env.hrt_customers, env.jobs, env.num_sas)
    slack = min(row['absolute_deadline'] - row['finish_time'] for row in schedule)
    shifted = [dict(row, start_time=row['start_time'] + slack,
                    finish_time=row['finish_time'] + slack) for row in schedule]
    PeriodicReference(shifted, env.hrt_customers, env.jobs, env.num_sas)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            tmp = stream.name
            yaml.safe_dump({'feasible_schedule': shifted}, stream, sort_keys=False)
        os.replace(tmp, path)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    print(f'Validated reference saved to {path} ({len(shifted)} subjobs; shift {slack:g})')
