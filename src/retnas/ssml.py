"""
SSML (Jeon et al., Electronics 2019) scheduler-core baseline adapted to RETNAS.

The original SSML algorithm targets a *preemptive uniprocessor* mixed system
with hard periodic tasks and soft aperiodic requests scheduled under EDF.  When
aperiodic work is pending, SSML computes the slack sigma using the modified
look-ahead EDF defer() function.  If sigma is positive, the current aperiodic
request is temporarily given the highest priority; while it executes, sigma is
consumed.  When sigma is exhausted, periodic work regains priority.

RETNAS differs from the original model: DNN requests are chains/DAGs of
non-preemptive layer sub-jobs, the platform contains heterogeneous
accelerators, and Environment.step() may pre-dispatch every ready layer far into
future time.  This implementation therefore avoids the old unsafe action-only
mapping.  Instead it performs a strict layer-level simulation:

  * SSML slack is computed from the currently pending HRT layer work;
  * BE layers are admitted only up to the slack budget available in the current
    epoch and only if the whole layer completes in the current epoch;
  * non-admitted BE layers stay in the ready queue and are not executed
    accidentally;
  * HRT layers are scheduled by EDF/EFT once slack is exhausted or unavailable;
  * completed layers are counted only when finish_time <= current_time.

This is a conservative scheduler-core adaptation, not a full continuous-time
preemptive implementation of SSML.

Suggested config knobs:
    ssml_capacity_factor: 1.0          # original paper is uniprocessor; set to num_sas for aggregate variant
    ssml_cost_mode: "best"             # best|mean|worst for slack computation
    ssml_debug: false
    ssml_abort_on_hrt_miss: false
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
from tqdm.auto import trange
from tqdm import tqdm
from retnas.environment import Environment
EPS = 1e-09

@dataclass
class Interval:
    rq_idx: int
    sa: int
    start: float
    finish: float
    is_hrt: bool

    @property
    def duration(self) -> float:
        return max(0.0, self.finish - self.start)

@dataclass
class SSMLPlan:
    hrt: Dict[int, Interval] = field(default_factory=dict)
    be: Dict[int, Interval] = field(default_factory=dict)
    sigma: float = 0.0
    be_budget: float = 0.0
    be_budget_used: float = 0.0
    be_selected_layers: int = 0
    be_deferred_jobs: Set[int] = field(default_factory=set)
    hrt_miss_candidates: Set[int] = field(default_factory=set)

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0

def _advance_time(env):
    start = float(env.current_time)
    if env.scheduling_period > 0:
        end = start + float(env.scheduling_period)
        timeskip = float(env.scheduling_period)
    else:
        timeskip = float(env.distribution(env.alpha) * env.scale)
        end = start + timeskip
    env.current_time = end
    return (start, end, timeskip)

def _next_state(env):
    if env.next_state_mode == 'standard':
        env.new_jobs_arrival()
        rb_next_state = env.encode_state()
        actor_next_state = rb_next_state
    elif env.next_state_mode == 'nonewjobs':
        rb_next_state = env.encode_state()
        env.new_jobs_arrival()
        actor_next_state = env.encode_state()
    elif env.next_state_mode == 'hrt_only':
        rb_next_state = env.encode_state()
        env.new_jobs_arrival(hrt_only=True)
        actor_next_state = env.encode_state()
    elif env.next_state_mode == 'hrt_only_all_periods':
        rb_next_state = env.encode_state()
        env.new_jobs_arrival(hrt_only=True, hrt_all_periods=True)
        actor_next_state = env.encode_state()
    else:
        raise RuntimeError(f'Unsupported next_state_mode={env.next_state_mode}')
    done = len(env.rq) == 0
    if env.imminents:
        assert max(env.imminents) < len(env.rq), f'Stale imminents: max={max(env.imminents)}, len={len(env.rq)}'
    env.sort_ready_queue()
    return (actor_next_state, rb_next_state, done)

def _is_committed(sj) -> bool:
    return sj.start_time is not None and sj.finish_time is not None and (sj.sa_id is not None)

def _latency(config, sj, sa: int) -> float:
    return float(config.all_jobs[sj.job_id].layer_latencies[sj.layer_id][int(sa)])

def _cost_for_slack(config, sj, mode: str) -> float:
    vals = np.asarray(config.all_jobs[sj.job_id].layer_latencies[sj.layer_id], dtype=float)
    mode = mode.lower()
    if mode == 'best':
        return float(np.min(vals))
    if mode == 'mean':
        return float(np.mean(vals))
    if mode == 'worst':
        return float(np.max(vals))
    raise ValueError("ssml_cost_mode must be 'best', 'mean', or 'worst'")

def _period_map(config) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for cust in getattr(config, 'hrt_customers', []) or []:
        if len(cust) >= 3:
            (_, job_id, period) = cust[:3]
            job_id = int(job_id)
            out[job_id] = min(float(period), out.get(job_id, float('inf')))
    return out

def _compute_U_p(config, cost_mode: str, capacity: float) -> float:
    """Periodic/HRT utilization estimate used by SSML defer()."""
    if not getattr(config, 'hrt_customers', None):
        return 0.0
    total = 0.0
    for cust in config.hrt_customers:
        if len(cust) < 4:
            continue
        (_, job_id, period, _) = cust[:4]
        job = config.all_jobs[int(job_id)]
        for l in range(job.num_layers):
            vals = np.asarray(job.layer_latencies[l], dtype=float)
            if cost_mode == 'best':
                c = float(np.min(vals))
            elif cost_mode == 'mean':
                c = float(np.mean(vals))
            elif cost_mode == 'worst':
                c = float(np.max(vals))
            else:
                raise ValueError("ssml_cost_mode must be 'best', 'mean', or 'worst'")
            total += c / max(float(period), EPS)
    return total / max(float(capacity), EPS)

def _current_hrt_items(env, config, period_start: float, cost_mode: str, job_period: Dict[int, float]):
    """
    Build the current periodic-work view for defer().  Each pending HRT layer is
    treated as one remaining periodic job fragment.  If it has already been
    dispatched, only its remaining execution time from period_start is counted.
    """
    items = []
    for (i, sj) in enumerate(env.rq):
        if not sj.is_hrt:
            continue
        if _is_committed(sj):
            if sj.finish_time <= period_start + EPS:
                continue
            c_left = max(0.0, float(sj.finish_time) - max(period_start, float(sj.start_time)))
        else:
            c_left = _cost_for_slack(config, sj, cost_mode)
        if c_left <= EPS:
            continue
        period = job_period.get(int(sj.job_id), max(float(sj.relative_deadline), EPS))
        items.append((i, float(sj.absolute_deadline), c_left, float(period)))
    return items

def _ssml_defer(env, config, period_start: float, capacity: float, U_p: float, job_period: Dict[int, float]) -> float:
    """
    Modified laEDF defer() from SSML Algorithm 2, adapted to pending RETNAS HRT
    layers.  The returned sigma is a time slack until the earliest HRT deadline.
    """
    cost_mode = getattr(config, 'ssml_cost_mode', 'best').lower()
    items = _current_hrt_items(env, config, period_start, cost_mode, job_period)
    if not items:
        P = float(getattr(env, 'scheduling_period', 0.0))
        return P if P > 0 else float('inf')
    d_n = min((d for (_, d, _, _) in items))
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    U = float(U_p)
    s = 0.0
    for (_, d_i, c_left, period) in sorted_items:
        ui = c_left / max(period, EPS) / max(capacity, EPS)
        U -= ui
        delta = d_i - d_n
        if delta <= EPS:
            x = c_left
        else:
            x = max(0.0, c_left - max(0.0, U_p - U) * delta * capacity)
            U += (c_left - x) / max(delta * capacity, EPS)
        s += x
    return d_n - (period_start + s / max(capacity, EPS))

def _dependency_release(env, rq_idx: int, planned_finish: Dict[int, float], period_start: float, period_end: float) -> Optional[float]:
    sj = env.rq[rq_idx]
    release = max(period_start, float(sj.arrival_time))
    for dep in getattr(sj, 'dependencies', []) or []:
        if dep in planned_finish:
            release = max(release, planned_finish[dep])
        elif 0 <= dep < len(env.rq) and _is_committed(env.rq[dep]):
            release = max(release, float(env.rq[dep].finish_time))
        else:
            return None
    if release >= period_end - EPS:
        return None
    return release

def _choose_eft_sa(config, sj, sa_available: List[float], release: float) -> Tuple[int, float, float]:
    best = None
    for a in range(config.num_sas):
        st = max(float(sa_available[a]), release)
        ft = st + _latency(config, sj, a)
        cand = (ft, st, a)
        if best is None or cand < best:
            best = cand
    (ft, st, a) = best
    return (int(a), float(st), float(ft))

def _build_plan(env, config, period_start: float, period_end: float, sigma: float, capacity: float) -> SSMLPlan:
    plan = SSMLPlan(sigma=float(sigma))
    plan.be_budget = max(0.0, min(float(sigma), period_end - period_start)) * max(capacity, EPS)
    sa_available = [max(float(t), period_start) for t in env.sa_busy_until]
    planned_finish: Dict[int, float] = {}
    planned: Set[int] = set()
    for (idx, sj) in enumerate(env.rq):
        if _is_committed(sj):
            sa_available[int(sj.sa_id)] = max(sa_available[int(sj.sa_id)], float(sj.finish_time))
            if sj.finish_time <= period_end + EPS:
                planned_finish[idx] = float(sj.finish_time)

    def schedule_one(indices: List[int], allow_be: bool) -> bool:
        nonlocal sa_available, planned_finish, planned, plan
        best = None
        for idx in indices:
            if idx in planned or _is_committed(env.rq[idx]):
                continue
            sj = env.rq[idx]
            release = _dependency_release(env, idx, planned_finish, period_start, period_end)
            if release is None:
                continue
            (a, st, ft) = _choose_eft_sa(config, sj, sa_available, release)
            if st >= period_end - EPS:
                continue
            if allow_be:
                lat = ft - st
                if ft > period_end + EPS:
                    continue
                if plan.be_budget_used + lat > plan.be_budget + EPS:
                    continue
            cand = (sj.absolute_deadline, st, ft, idx, a)
            if best is None or cand < best:
                best = cand
        if best is None:
            return False
        (_, st, ft, idx, a) = best
        sj = env.rq[idx]
        itv = Interval(idx, a, st, ft, sj.is_hrt)
        if sj.is_hrt:
            plan.hrt[idx] = itv
        else:
            plan.be[idx] = itv
            plan.be_selected_layers += 1
            plan.be_budget_used += ft - st
        sa_available[a] = ft
        planned_finish[idx] = ft
        planned.add(idx)
        return True
    if plan.be_budget > EPS:
        while True:
            be_indices = sorted([i for (i, sj) in enumerate(env.rq) if not sj.is_hrt], key=lambda i: (env.rq[i].absolute_deadline, env.rq[i].arrival_time, env.rq[i].layer_id, i))
            if not schedule_one(be_indices, allow_be=True):
                break
    while True:
        hrt_indices = sorted([i for (i, sj) in enumerate(env.rq) if sj.is_hrt], key=lambda i: (env.rq[i].absolute_deadline, env.rq[i].arrival_time, env.rq[i].layer_id, i))
        if not schedule_one(hrt_indices, allow_be=False):
            break
    selected_be_jobs = {env.rq[i].job_instance_id for i in plan.be}
    for (i, sj) in enumerate(env.rq):
        if not sj.is_hrt and (not _is_committed(sj)) and (sj.job_instance_id not in selected_be_jobs):
            plan.be_deferred_jobs.add(sj.job_instance_id)
    for (i, sj) in enumerate(env.rq):
        if sj.is_hrt and (not _is_committed(sj)) and (i not in plan.hrt) and (period_end >= sj.absolute_deadline - EPS):
            plan.hrt_miss_candidates.add(sj.job_instance_id)
    _validate_no_overlap(env, plan)
    _validate_dependencies(env, plan, period_start)
    return plan

def _validate_no_overlap(env, plan: SSMLPlan) -> None:
    by_sa: Dict[int, List[Interval]] = {a: [] for a in range(env.num_sas)}
    for (i, sj) in enumerate(env.rq):
        if _is_committed(sj):
            by_sa[int(sj.sa_id)].append(Interval(i, int(sj.sa_id), float(sj.start_time), float(sj.finish_time), sj.is_hrt))
    for itv in list(plan.hrt.values()) + list(plan.be.values()):
        by_sa[itv.sa].append(itv)
    for (sa, arr) in by_sa.items():
        arr.sort(key=lambda x: (x.start, x.finish, x.rq_idx))
        for (u, v) in zip(arr, arr[1:]):
            if u.finish > v.start + EPS:
                raise AssertionError(f'SSML plan overlaps on SA{sa}: {u} vs {v}')

def _validate_dependencies(env, plan: SSMLPlan, period_start: float) -> None:
    finish = {}
    for (i, sj) in enumerate(env.rq):
        if _is_committed(sj):
            finish[i] = float(sj.finish_time)
    for (i, itv) in plan.hrt.items():
        finish[i] = itv.finish
    for (i, itv) in plan.be.items():
        finish[i] = itv.finish
    starts = {i: itv.start for (i, itv) in plan.hrt.items()}
    starts.update({i: itv.start for (i, itv) in plan.be.items()})
    for (i, st) in starts.items():
        for dep in getattr(env.rq[i], 'dependencies', []) or []:
            if dep not in finish:
                raise AssertionError(f'SSML scheduled rq[{i}] before unresolved dependency rq[{dep}]')
            if finish[dep] > st + EPS:
                raise AssertionError(f'SSML dependency violation: dep rq[{dep}] finishes {finish[dep]}, child rq[{i}] starts {st}')

def _apply_plan(env, plan: SSMLPlan) -> None:
    for (i, sj) in enumerate(env.rq):
        if _is_committed(sj):
            continue
        sj.start_time = None
        sj.finish_time = None
        sj.sa_id = None
    for mapping in (plan.be, plan.hrt):
        for (idx, itv) in mapping.items():
            sj = env.rq[idx]
            if _is_committed(sj):
                continue
            sj.sa_id = int(itv.sa)
            sj.start_time = float(itv.start)
            sj.finish_time = float(itv.finish)

def _update_env(env, config, plan: SSMLPlan, period_start: float, period_end: float):
    reward = 0.0
    hrt_feasible = True
    env.old_rq = deepcopy(env.rq)
    env.rq.clear()
    conv: Dict[int, int] = {}
    for (old_idx, sj) in enumerate(env.old_rq):
        if _is_committed(sj):
            if sj.start_time <= env.current_time + EPS:
                env.sa_busy_until[int(sj.sa_id)] = max(env.sa_busy_until[int(sj.sa_id)], float(sj.finish_time))
            if sj.finish_time <= env.current_time + EPS:
                env.executed_sjs += 1
                if sj.finish_time >= env.sa_busy_until[int(sj.sa_id)] - EPS:
                    env.sa_last_sj_executed[int(sj.sa_id)] = (sj.job_instance_id, sj.job_id, sj.layer_id)
                if env.reward_slack_factor > 0:
                    factor = 5 if sj.is_hrt else 1
                    reward += factor * env.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / max(sj.relative_deadline, EPS)
                if sj.is_last:
                    tmp = env.calculate_lastsj_reward(sj)
                    reward += tmp
                    if sj.is_hrt and sj.finish_time > sj.absolute_deadline + EPS:
                        hrt_feasible = False
            else:
                env.rq.append(sj)
                conv[old_idx] = len(env.rq) - 1
        else:
            if env.current_time >= sj.absolute_deadline - EPS:
                if sj.is_last:
                    if sj.is_hrt:
                        env.deadline_misses_hrt += 1
                        hrt_feasible = False
                    else:
                        env.deadline_misses_be += 1
                continue
            env.rq.append(sj)
            conv[old_idx] = len(env.rq) - 1
    env.imminents.clear()
    for (new_idx, sj) in enumerate(env.rq):
        sj.dependencies = [conv[d] for d in sj.dependencies if d in conv]
        sj.children = [conv[d] for d in sj.children if d in conv]
        if len(sj.dependencies) == 0 and (not _is_committed(sj)):
            env.imminents.add(new_idx)
    for a in range(env.num_sas):
        env.sa_busy_until[a] = max(env.sa_busy_until[a], env.current_time)
    return (reward, hrt_feasible)

def _ssml_strict_step(env, config, capacity: float, U_p: float, job_period: Dict[int, float]):
    (period_start, period_end, _) = _advance_time(env)
    if len(env.rq) == 0:
        (actor_next_state, rb_next_state, done) = _next_state(env)
        return (actor_next_state, rb_next_state, 0.0, done, True, 0.0, SSMLPlan())
    env.current_time = period_start
    sigma = _ssml_defer(env, config, period_start, capacity, U_p, job_period)
    plan = _build_plan(env, config, period_start, period_end, sigma, capacity)
    env.current_time = period_end
    _apply_plan(env, plan)
    (reward, hrt_feasible) = _update_env(env, config, plan, period_start, period_end)
    (actor_next_state, rb_next_state, done) = _next_state(env)
    return (actor_next_state, rb_next_state, reward, done, hrt_feasible, plan)

def ssml(config):
    print('[SSML STRICT] Slack-Stealing with Modified laEDF | direct layer-level execution')
    print('[SSML STRICT] BE layers are admitted only while sigma>0 and must complete in the current epoch.')
    print('[SSML STRICT] Non-admitted BE layers remain pending and are not passed to Environment.step().')
    capacity = float(getattr(config, 'ssml_capacity_factor', 1.0))
    cost_mode = getattr(config, 'ssml_cost_mode', 'best').lower()
    job_period = _period_map(config)
    U_p = _compute_U_p(config, cost_mode, capacity)
    print(f'[SSML STRICT] cost_mode={cost_mode} | capacity_factor={capacity:.3f} | U_p={U_p:.6f}')
    env = Environment(config, seed=config.seed)
    env.prepend_sa_remaining_time = False
    env.use_sa_assignement_heuristic = False
    env.reset()
    slast = slast_be = slast_hrt = 0.0
    total_selected_be_layers = 0
    total_deferred_be_jobs = 0
    total_hrt_infeasible_steps = 0
    debug = bool(getattr(config, 'ssml_debug', False))
    abort_on_hrt_miss = bool(getattr(config, 'ssml_abort_on_hrt_miss', False))
    for step in trange(config.num_steps):
        (_, _, _, _, hrt_feasible, plan) = _ssml_strict_step(env, config, capacity, U_p, job_period)
        total_selected_be_layers += plan.be_selected_layers
        total_deferred_be_jobs += len(plan.be_deferred_jobs)
        if not hrt_feasible:
            total_hrt_infeasible_steps += 1
            if abort_on_hrt_miss:
                raise RuntimeError('SSML produced an HRT miss/infeasible step.')
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = _safe_ratio(total_deadline_hits, total_arrived_jobs)
        slast_be = _safe_ratio(env.deadline_hits_be, env.total_arrived_jobs_be)
        slast_hrt = _safe_ratio(env.deadline_hits_hrt, env.total_arrived_jobs_hrt)
        if slast > 1.0 + 1e-09 or slast_be > 1.0 + 1e-09 or slast_hrt > 1.0 + 1e-09:
            raise RuntimeError(f'Invalid accounting: SLAST > 100% (total={slast}, BE={slast_be}, HRT={slast_hrt}).')
        if debug or step + 1 == config.num_steps:
            tqdm.write('Step:\t{}\tSlast:\t{:0.2f}%\tSlast_BE:\t{:0.2f}%\tSlast_HRT:\t{:0.2f}%\tHRT/BE MISS: {},{}\tBE selected_layers step/cum: {}/{}\tBE deferred_jobs step/cum: {}/{}\tsigma={:.2f}\tBE_budget={:.2f}/{:.2f}\tRQ={}\tHRT infeasible steps={}'.format(step + 1, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, plan.be_selected_layers, total_selected_be_layers, len(plan.be_deferred_jobs), total_deferred_be_jobs, plan.sigma, plan.be_budget_used, plan.be_budget, len(env.rq), total_hrt_infeasible_steps))
    print(f'SLA Satisfaction Rate: {slast * 100:.2f}%')
    print(f'SLA Satisfaction Rate BE: {slast_be * 100:.2f}% ({env.deadline_hits_be}/{env.total_arrived_jobs_be})')
    print(f'SLA Satisfaction Rate HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    print(f'SSML selected BE layers: {total_selected_be_layers}')
    print(f'SSML deferred BE jobs cumulative-count: {total_deferred_be_jobs}')
    print(f'SSML HRT infeasible steps: {total_hrt_infeasible_steps}')
    print('Throughput:', env.executed_sjs * 1000000000.0 / env.current_time if env.current_time > 0 else 0.0)
    if env.num_tenants > 0:
        slasr_per_tenant = [env.deadline_hits_per_tenant[i] / max(1, env.arrived_jobs_per_tenant[i]) if env.arrived_jobs_per_tenant[i] > 0 else 0.0 for i in range(env.num_tenants)]
        print('SLASR per tenant:', slasr_per_tenant)
        print('SLASR per tenant std:', np.std(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant min:', np.min(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant max:', np.max(slasr_per_tenant) * 100, '%')
