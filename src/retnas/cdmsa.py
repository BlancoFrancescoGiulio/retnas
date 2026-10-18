"""
CD-MSA scheduler-core baseline adapted to the RETNAS environment.

This file implements the *scheduler* part of Wang et al., "CD-MSA:
Cooperative and Deadline-Aware Scheduling for Efficient Multi-Tenancy on DNN
Accelerators" (TPDS 2023), namely:

  * EDF-ordered dynamic aperiodic task set;
  * O(n)-style schedulability test based on Eq. (3):
        sum_{i<=k} C_i <= D_k;
  * exclusion of tasks when the current queue is not schedulable;
  * laxity-aware EDF reordering based on Eq. (4) and Eq. (5);
  * generation of an execution proposal for the current RETNAS ready queue.

Important adaptation notes
--------------------------
The original CD-MSA paper assumes a single MSA-based accelerator with a
specific fine-grained preemption/co-location architecture and an optional
Dynamic Task Fusion (DTF) mechanism. The RETNAS simulator does not expose that
hardware model, a fusion-latency table, or sub-tile checkpoints. Therefore this
implementation is a *strict scheduler-core adaptation*, not a cycle-accurate
reimplementation of the complete CD-MSA hardware/DTF stack.

Compared with the previous implementation, the critical fixes are:

  1. The schedulability test is performed at DNN-request level, not at isolated
     layer level. A CD-MSA "task" corresponds to one outstanding DNN request;
     its cost is the remaining exclusive-mode latency of all its remaining
     layers.
  2. Tasks excluded by the CD-MSA schedulability/laxity logic are not appended
     back to the execution order. BE excluded tasks are abandoned and counted as
     soft deadline misses. HRT tasks are never abandoned by this baseline.
  3. Laxity is computed from the EDF prefix as in Eq. (4), and the laxity-aware
     promotion rule follows Eq. (5).
  4. Accelerator assignment is kept consistent with the cost model by default:
     each layer is placed on its fastest accelerator. A greedy EFT assignment can
     be enabled with config.cdmsa_assignment = "eft".

Suggested config knobs
----------------------
cdmsa_cost_mode:       "best" (default), "mean", or "worst"
cdmsa_assignment:      "fastest" (default) or "eft"
cdmsa_drop_excluded:   True (default). If False, excluded BE tasks are deferred
                       with lowest priority instead of being abandoned.
cdmsa_debug:           False (default). Print per-step internal counters.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from tqdm.auto import trange
from tqdm import tqdm
from retnas.environment import Environment
EPS = 1e-09

@dataclass
class CDMSATask:
    """One outstanding DNN request, represented by its remaining sub-jobs."""
    key: int
    job_id: int
    rq_indices: List[int]
    absolute_deadline: float
    arrival_time: float
    is_hrt: bool
    owner: Optional[int]
    cost: float
    abandoned_count: int = 0

    def remaining_deadline(self, now: float) -> float:
        return float(self.absolute_deadline) - float(now)

def _layer_cost(config, sj, mode: str='best') -> float:
    lat = np.asarray(config.all_jobs[sj.job_id].layer_latencies[sj.layer_id], dtype=float)
    mode = mode.lower()
    if mode == 'best':
        return float(np.min(lat))
    if mode == 'worst':
        return float(np.max(lat))
    if mode == 'mean':
        return float(np.mean(lat))
    raise ValueError(f'Unsupported cdmsa_cost_mode={mode!r}; use best, mean, or worst')

def _fastest_sa(config, sj) -> int:
    lat = config.all_jobs[sj.job_id].layer_latencies[sj.layer_id]
    return int(np.argmin(lat))

def _one_hot(sa: int, num_sas: int) -> List[float]:
    v = [0.0] * num_sas
    v[int(sa)] = 1.0
    return v

def _empty_action(num_sas: int) -> np.ndarray:
    return np.zeros((0, 1 + num_sas), dtype=np.float32)

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def _group_ready_queue(env, config, abandon_counts: Dict[int, int]) -> Dict[int, CDMSATask]:
    """
    Group RETNAS sub-jobs by job_instance_id.

    CD-MSA schedules requests/tasks, not individual layers. Each task cost C_i
    is the remaining exclusive-mode latency of all sub-jobs of that request that
    are still in the ready queue.
    """
    mode = getattr(config, 'cdmsa_cost_mode', 'best')
    groups: Dict[int, List[int]] = {}
    for (idx, sj) in enumerate(env.rq):
        groups.setdefault(int(sj.job_instance_id), []).append(idx)
    tasks: Dict[int, CDMSATask] = {}
    for (key, indices) in groups.items():
        indices.sort(key=lambda i: env.rq[i].layer_id)
        first = env.rq[indices[0]]
        is_hrt = any((env.rq[i].is_hrt for i in indices))
        cost = sum((_layer_cost(config, env.rq[i], mode) for i in indices))
        tasks[key] = CDMSATask(key=key, job_id=int(first.job_id), rq_indices=indices, absolute_deadline=float(first.absolute_deadline), arrival_time=float(first.arrival_time), is_hrt=bool(is_hrt), owner=getattr(first, 'owner', None), cost=float(cost), abandoned_count=int(abandon_counts.get(int(first.job_id), 0)))
    return tasks

def _rebuild_dependencies_and_imminents(env) -> None:
    """
    Rebuild chain dependencies after removing complete DNN requests.

    RETNAS jobs are generated as a layer chain. After removing some jobs from
    the ready queue, the old dependency indices are no longer reliable. We
    rebuild dependencies by (job_instance_id, layer_id), which is consistent with
    Environment.new_job_arrival().
    """
    env.rq.sort(key=lambda sj: (sj.absolute_deadline, sj.job_instance_id, sj.layer_id))
    by_job: Dict[int, List[int]] = {}
    for (i, sj) in enumerate(env.rq):
        by_job.setdefault(int(sj.job_instance_id), []).append(i)
    env.imminents.clear()
    for (_, idxs) in by_job.items():
        idxs.sort(key=lambda i: env.rq[i].layer_id)
        for (pos, idx) in enumerate(idxs):
            sj = env.rq[idx]
            sj.dependencies = [idxs[pos - 1]] if pos > 0 else []
            sj.children = [idxs[pos + 1]] if pos + 1 < len(idxs) else []
            sj.is_last = pos + 1 == len(idxs)
            if not sj.dependencies:
                env.imminents.add(idx)

def _abandon_be_tasks(env, task_keys: Iterable[int], abandon_counts: Dict[int, int]) -> int:
    """
    Drop complete BE DNN requests selected by the CD-MSA exclusion rule.

    CD-MSA abandons excluded tasks to preserve global throughput. In RETNAS, a
    deadline miss is counted at DNN-request granularity, so we add one BE miss
    per abandoned BE job instance and remove all its remaining layers.
    """
    keys = set((int(k) for k in task_keys))
    if not keys:
        return 0
    droppable: Dict[int, int] = {}
    for sj in env.rq:
        if int(sj.job_instance_id) in keys and (not sj.is_hrt):
            droppable[int(sj.job_instance_id)] = int(sj.job_id)
    if not droppable:
        return 0
    for (_, job_id) in droppable.items():
        env.deadline_misses_be += 1
        abandon_counts[job_id] = abandon_counts.get(job_id, 0) + 1
    env.rq = [sj for sj in env.rq if int(sj.job_instance_id) not in droppable]
    _rebuild_dependencies_and_imminents(env)
    return len(droppable)

def _edf_task_order(tasks: Dict[int, CDMSATask], now: float) -> List[int]:
    return sorted(tasks.keys(), key=lambda k: (tasks[k].absolute_deadline, tasks[k].arrival_time, tasks[k].job_id, tasks[k].key))

def _choose_exclusion_candidate(tasks: Dict[int, CDMSATask], prefix: Sequence[int], miss_key: int) -> Optional[int]:
    """
    Choose the task to exclude as described around Algorithm 2.

    The paper chooses between tau_miss and tau_large and uses per-model abandon
    counts for fairness in tie-breaking. In the RETNAS-HRT setting we never
    abandon HRT requests; if a prefix contains only HRT tasks, the function
    returns None and the remaining HRT feasibility is left to the common HRT
    deployment safety layer.
    """
    large_key = max(prefix, key=lambda k: (tasks[k].cost, -tasks[k].absolute_deadline))
    candidates = []
    for k in (miss_key, large_key):
        if not tasks[k].is_hrt and k not in candidates:
            candidates.append(k)
    if not candidates:
        return None
    return min(candidates, key=lambda k: (tasks[k].abandoned_count, -tasks[k].cost, -tasks[k].absolute_deadline, k))

def _cdmsa_schedulability_prune(tasks: Dict[int, CDMSATask], now: float) -> Tuple[List[int], List[int], int]:
    """
    Apply CD-MSA Eq. (3) and Algorithm-2-like exclusion.

    Returns:
        kept_order: EDF order after removing excluded BE tasks.
        excluded: BE task keys abandoned by the scheduler.
        hrt_prefix_violations: number of violations that could not be fixed
            because the offending prefix contained no droppable BE task.
    """
    order = _edf_task_order(tasks, now)
    excluded: List[int] = []
    hrt_prefix_violations = 0
    changed = True
    while changed and order:
        changed = False
        cumulative = 0.0
        for (pos, key) in enumerate(order):
            cumulative += float(tasks[key].cost)
            Dk = tasks[key].remaining_deadline(now)
            if cumulative > Dk + EPS:
                prefix = order[:pos + 1]
                victim = _choose_exclusion_candidate(tasks, prefix, miss_key=key)
                if victim is None:
                    hrt_prefix_violations += 1
                    continue
                order.remove(victim)
                excluded.append(victim)
                changed = True
                break
    return (order, excluded, hrt_prefix_violations)

def _prefix_laxities(tasks: Dict[int, CDMSATask], order: Sequence[int], now: float) -> Dict[int, float]:
    """Eq. (4): laxity_k = 1 - (sum_{i<=k} C_i) / D_k."""
    result: Dict[int, float] = {}
    cumulative = 0.0
    for key in order:
        cumulative += float(tasks[key].cost)
        Dk = max(tasks[key].remaining_deadline(now), EPS)
        result[key] = 1.0 - cumulative / Dk
    return result

def _cdmsa_laxity_order(tasks: Dict[int, CDMSATask], edf_order: Sequence[int], now: float, period: float) -> List[int]:
    """
    CD-MSA laxity-aware EDF sequence construction.

    Tasks whose deadline is within the current scheduling period are mandatory.
    Some later tasks can be promoted if Eq. (5) holds for their EDF-prefix
    section; promoted tasks are ordered by increasing laxity because smaller
    laxity means higher miss risk.
    """
    if not edf_order:
        return []
    P = max(float(period), EPS)
    order = list(edf_order)
    pos = next((i for (i, k) in enumerate(order) if tasks[k].remaining_deadline(now) > P + EPS), len(order))
    mandatory = order[:pos]
    candidates = order[pos:]
    if not candidates:
        return mandatory
    suffix_cum = 0.0
    promotable: List[int] = []
    not_promoted: List[int] = []
    for key in candidates:
        suffix_cum += float(tasks[key].cost)
        Dk = tasks[key].remaining_deadline(now)
        if suffix_cum + P <= Dk + EPS:
            promotable.append(key)
        else:
            not_promoted.append(key)
    lax = _prefix_laxities(tasks, order, now)
    promotable.sort(key=lambda k: (lax[k], tasks[k].absolute_deadline, tasks[k].arrival_time, k))
    return mandatory + promotable + not_promoted

def _build_action_for_current_rq(env, config, task_order: Sequence[int]) -> np.ndarray:
    """
    Convert the CD-MSA task order into RETNAS' priority+accelerator action.

    The action has one row per current ready-queue entry. Priorities are
    assigned first by DNN-task order, then by layer_id. This preserves DNN
    chain precedence while expressing CD-MSA's request-level scheduling choice.
    """
    n = len(env.rq)
    if n == 0:
        return _empty_action(config.num_sas)
    task_rank = {int(k): r for (r, k) in enumerate(task_order)}
    default_rank = len(task_rank) + 10000
    assignment_mode = getattr(config, 'cdmsa_assignment', 'fastest').lower()
    assignments: Dict[int, int] = {}
    if assignment_mode == 'fastest':
        for (i, sj) in enumerate(env.rq):
            assignments[i] = _fastest_sa(config, sj)
    elif assignment_mode == 'eft':
        ordered_indices = sorted(range(n), key=lambda i: (task_rank.get(int(env.rq[i].job_instance_id), default_rank), env.rq[i].layer_id, env.rq[i].absolute_deadline, i))
        sa_busy = list(env.sa_busy_until)
        planned_finish: Dict[int, float] = {}
        for i in ordered_indices:
            sj = env.rq[i]
            lat = config.all_jobs[sj.job_id].layer_latencies[sj.layer_id]
            dep_ready = 0.0
            for d in getattr(sj, 'dependencies', []) or []:
                dep_ready = max(dep_ready, planned_finish.get(d, 0.0))
            best_sa = int(np.argmin([max(sa_busy[a], dep_ready) + lat[a] for a in range(config.num_sas)]))
            assignments[i] = best_sa
            st = max(sa_busy[best_sa], dep_ready)
            planned_finish[i] = st + float(lat[best_sa])
            sa_busy[best_sa] = planned_finish[i]
    else:
        raise ValueError("cdmsa_assignment must be 'fastest' or 'eft'")
    order_keys = {}
    for (i, sj) in enumerate(env.rq):
        order_keys[i] = (task_rank.get(int(sj.job_instance_id), default_rank), sj.layer_id, sj.absolute_deadline, sj.arrival_time, i)
    ordered = sorted(range(n), key=lambda i: order_keys[i])
    rank = {idx: r for (r, idx) in enumerate(ordered)}
    action = np.zeros((n, 1 + config.num_sas), dtype=np.float32)
    for i in range(n):
        action[i, 0] = float(rank[i])
        action[i, 1:1 + config.num_sas] = np.asarray(_one_hot(assignments[i], config.num_sas), dtype=np.float32)
    return action

def cdmsa(config):
    print('[CD-MSA] Cooperative deadline/laxity-aware EDF scheduler-core adaptation')
    print('[CD-MSA] DTF/fusion hardware is not modeled unless provided by the simulator; using exclusive-mode costs.')
    env = Environment(config, seed=config.seed)
    env.prepend_sa_remaining_time = False
    env.use_sa_assignement_heuristic = False
    if config.hrt and config.hrt_step_mode == 'deployment':
        env.hrt_step_mode = 'deployment'
    else:
        env.hrt_step_mode = 'all'
    env.reset()
    abandon_counts: Dict[int, int] = {}
    total_excluded_be = 0
    total_hrt_prefix_violations = 0
    debug = bool(getattr(config, 'cdmsa_debug', False))
    drop_excluded = bool(getattr(config, 'cdmsa_drop_excluded', True))
    for step in trange(config.num_steps):
        excluded_step = 0
        hrt_viol_step = 0
        if len(env.rq) > 0:
            tasks = _group_ready_queue(env, config, abandon_counts)
            (kept_edf, excluded, hrt_viol) = _cdmsa_schedulability_prune(tasks, env.current_time)
            hrt_viol_step = hrt_viol
            total_hrt_prefix_violations += hrt_viol
            if drop_excluded:
                excluded_step = _abandon_be_tasks(env, excluded, abandon_counts)
                total_excluded_be += excluded_step
                tasks = _group_ready_queue(env, config, abandon_counts)
                kept_edf = [k for k in kept_edf if k in tasks]
            else:
                kept_edf = [k for k in kept_edf if k in tasks]
                kept_edf += [k for k in excluded if k in tasks]
            final_task_order = _cdmsa_laxity_order(tasks, kept_edf, env.current_time, config.scheduling_period)
            action = _build_action_for_current_rq(env, config, final_task_order)
        else:
            action = _empty_action(config.num_sas)
        ret = env.step(action)
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = _safe_ratio(total_deadline_hits, total_arrived_jobs)
        slast_be = _safe_ratio(env.deadline_hits_be, env.total_arrived_jobs_be)
        slast_hrt = _safe_ratio(env.deadline_hits_hrt, env.total_arrived_jobs_hrt)
        percentage_wcet_relies = _safe_ratio(env.hrt_wcet_win, env.hrt_wcet_win + env.hrt_rl_win)
        if env.deadline_misses_hrt and bool(getattr(config, 'cdmsa_abort_on_hrt_miss', False)):
            raise RuntimeError('CD-MSA baseline produced HRT misses. This baseline was not designed for hard-real-time guarantees; disable cdmsa_abort_on_hrt_miss to record them as baseline failures.')
        extra = ''
        if debug:
            extra = f'\tRQ: {len(env.rq)}\tExcluded BE step/cum: {excluded_step}/{total_excluded_be}\tHRT-prefix viol step/cum: {hrt_viol_step}/{total_hrt_prefix_violations}'
        else:
            extra = f'\tExcluded BE step/cum: {excluded_step}/{total_excluded_be}'
        tqdm.write('Step:\t{}\tSlast:\t{:0.2f}%\tSlast_BE:\t{:0.2f}%\tSlast_HRT:\t{:0.2f}%\tHRT/BE MISS: {},{}\tWCET won {:0.2f}% times{}'.format(step + 1, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, percentage_wcet_relies * 100, extra))
    total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
    total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
    slast = _safe_ratio(total_deadline_hits, total_arrived_jobs)
    slast_be = _safe_ratio(env.deadline_hits_be, env.total_arrived_jobs_be)
    slast_hrt = _safe_ratio(env.deadline_hits_hrt, env.total_arrived_jobs_hrt)
    print(f'SLA Satisfaction Rate: {slast * 100:.2f}%')
    print(f'SLA Satisfaction Rate BE: {slast_be * 100:.2f}% ({env.deadline_hits_be}/{env.total_arrived_jobs_be})')
    print(f'SLA Satisfaction Rate HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    print(f'CD-MSA excluded BE jobs: {total_excluded_be}')
    print(f'CD-MSA unresolved HRT-prefix violations: {total_hrt_prefix_violations}')
    print('Throughput:', env.executed_sjs * 1000000000.0 / env.current_time if env.current_time > 0 else 0.0)
    if env.num_tenants > 0:
        slasr_per_tenant = [env.deadline_hits_per_tenant[i] / max(1, env.arrived_jobs_per_tenant[i]) if env.arrived_jobs_per_tenant[i] > 0 else 0.0 for i in range(env.num_tenants)]
        print('SLASR per tenant:', slasr_per_tenant)
        print('SLASR per tenant std:', np.std(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant min:', np.min(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant max:', np.max(slasr_per_tenant) * 100, '%')
