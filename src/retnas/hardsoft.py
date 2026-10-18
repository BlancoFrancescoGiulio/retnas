"""
Strict Tang et al. hard-periodic / soft-aperiodic baseline for RETNAS-HRT.

This version implements both phases of Tang et al.:
  (1) it builds an offline hard-periodic schedule internally using either
      H-EDF or FF;
  (2) it schedules soft aperiodic work online using RC, TC, or TCSD.

It does NOT use the generic RETNAS action interface to execute the
baseline, because the action interface only expresses priority and accelerator
assignment. Tang et al.'s RC/TC/TCSD algorithms, instead, make explicit
slack-window placement decisions. In particular:

  * soft jobs that do not fit in the computed slack table must not be executed
    anyway later in the same decision step;
  * a soft job accepted into a slack window must not start before the planned
    slack-window start;
  * TCSD may shift hard-periodic work earlier on the SAME accelerator, but it
    must not migrate hard work across accelerators;
  * the baseline must not fabricate an online fallback hard schedule if the
    offline reference schedule is missing.

The runner below therefore constructs a strict plan and applies it directly to
Environment.rq, while preserving the Environment counters, state encoding,
arrival generation, and reward/miss accounting conventions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
from copy import deepcopy
import math
import numpy as np
from tqdm.auto import trange
from tqdm import tqdm
from retnas.environment import Environment
EPS = 1e-09

@dataclass(frozen=True)
class SlotKey:
    job_id: int
    layer_id: int
    deadline: float
    ordinal: int

@dataclass
class RefSlot:
    """One hard-real-time segment in Tang et al.'s offline periodic schedule."""
    key: SlotKey
    sa: int
    job_id: int
    layer_id: int
    start: float
    finish: float
    deadline: float
    release: float = 0.0
    customer_id: int = -1
    instance_id: int = -1
    est: float = 0.0
    lst: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.finish - self.start)

@dataclass(frozen=True)
class HRTTask:
    """Layer-level hard-periodic task instance used to build Tang's offline schedule."""
    key: SlotKey
    customer_id: int
    instance_id: int
    job_id: int
    layer_id: int
    release: float
    deadline: float
    pred: Optional[SlotKey] = None

@dataclass
class Interval:
    sa: int
    start: float
    finish: float
    kind: str
    rq_idx: Optional[int] = None
    slot_key: Optional[SlotKey] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.finish - self.start)

@dataclass
class SlackWindow:
    sa: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return max(0.0, self.end - self.start)

@dataclass
class Plan:
    """Strict schedule proposal produced by RC/TC/TCSD."""
    hrt: Dict[int, Interval] = field(default_factory=dict)
    be: Dict[int, Interval] = field(default_factory=dict)
    rejected_be_jobs: Set[int] = field(default_factory=set)
    deferred_be_jobs: Set[int] = field(default_factory=set)
    failed_hrt_jobs: Set[int] = field(default_factory=set)
    mode: str = 'tc'

def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0

def _latency(config, sj, sa: int) -> float:
    return float(config.all_jobs[sj.job_id].layer_latencies[sj.layer_id][sa])

def _period_bounds(env) -> Tuple[float, float, float]:
    if env.scheduling_period <= 0:
        raise RuntimeError('HardSoft/Tang baseline requires a positive scheduling_period.')
    start = float(env.current_time)
    period = float(env.scheduling_period)
    return (start, start + period, period)

def _advance_time(env) -> Tuple[float, float, float]:
    (start, end, period) = _period_bounds(env)
    env.current_time = end
    return (start, end, period)

def _is_already_committed(sj, now: float) -> bool:
    """A sub-job was scheduled in a previous step and has not completed yet."""
    return sj.start_time is not None and sj.finish_time is not None and (sj.finish_time > now)

def _hardsoft_debug_enabled(config) -> bool:
    return bool(getattr(config, 'hardsoft_debug', False))

def _strict_reject(config) -> bool:
    return bool(getattr(config, 'hardsoft_strict_reject', True))

def _hard_scheduler_kind(config) -> str:
    """Select Tang's offline HRT scheduler: 'h_edf' or 'ff'."""
    kind = getattr(config, 'hardsoft_periodic_scheduler', getattr(config, 'hardsoft_hrt_scheduler', 'ff'))
    kind = str(kind).lower().replace('-', '_')
    if kind in ('hedf', 'h_edf', 'h_vedf'):
        return 'h_edf'
    if kind in ('ff', 'fastest_first'):
        return 'ff'
    raise ValueError("hardsoft_periodic_scheduler must be 'h_edf' or 'ff'.")

def _offline_horizon_end(env, config, period_end: float) -> float:
    """
    Horizon used to construct the offline hard-periodic schedule.  It must be
    longer than the current RETNAS scheduling period because LST extraction
    depends on future hard-periodic instances on the same resource.
    """
    periods = max(1, int(getattr(config, 'hardsoft_lookahead_periods', 4)))
    return period_end + periods * float(env.scheduling_period)

def _generate_hrt_tasks(config, horizon_end: float) -> List[HRTTask]:
    """Generate layer-level hard periodic instances from config.hrt_customers."""
    tasks: List[HRTTask] = []
    ordinal = 0
    for (hrt_customer_id, job_id, period, relative_deadline) in getattr(config, 'hrt_customers', []):
        period = float(period)
        relative_deadline = float(relative_deadline)
        if period <= 0:
            raise RuntimeError(f'Invalid HRT period for customer {hrt_customer_id}: {period}')
        job = config.all_jobs[job_id]
        k = 0
        release = 0.0
        while release < horizon_end - EPS:
            deadline = release + relative_deadline
            pred = None
            for layer_id in range(job.num_layers):
                key = SlotKey(job_id=int(job_id), layer_id=int(layer_id), deadline=float(deadline), ordinal=ordinal)
                tasks.append(HRTTask(key=key, customer_id=int(hrt_customer_id), instance_id=int(k), job_id=int(job_id), layer_id=int(layer_id), release=float(release), deadline=float(deadline), pred=pred))
                pred = key
                ordinal += 1
            k += 1
            release = k * period
    return tasks

def _task_latency(config, task: HRTTask, sa: int) -> float:
    return float(config.all_jobs[task.job_id].layer_latencies[task.layer_id][sa])

def _slots_from_schedule(config, tasks: List[HRTTask], schedule: Dict[SlotKey, Tuple[int, float, float]]) -> List[RefSlot]:
    by_key = {t.key: t for t in tasks}
    slots: List[RefSlot] = []
    for (key, (sa, st, ft)) in schedule.items():
        t = by_key[key]
        slots.append(RefSlot(key=key, sa=int(sa), job_id=t.job_id, layer_id=t.layer_id, start=float(st), finish=float(ft), deadline=float(t.deadline), release=float(t.release), customer_id=int(t.customer_id), instance_id=int(t.instance_id), est=max(float(t.release), float(st)), lst=max(float(t.release), float(st))))
    slots.sort(key=lambda s: (s.start, s.finish, s.sa, s.deadline, s.job_id, s.layer_id))
    return slots

def _simulate_partitioned_edf(config, tasks: List[HRTTask], assignment: Dict[SlotKey, int]) -> Optional[Dict[SlotKey, Tuple[int, float, float]]]:
    """
    Simulate local non-preemptive EDF on each assigned accelerator while
    respecting layer-chain dependencies.  Returns a full schedule if feasible.
    """
    task_by_key = {t.key: t for t in tasks if t.key in assignment}
    unscheduled: Set[SlotKey] = set(task_by_key)
    scheduled: Dict[SlotKey, Tuple[int, float, float]] = {}
    sa_available = [0.0] * int(config.num_sas)
    while unscheduled:
        progress = False
        for sa in sorted(range(config.num_sas), key=lambda a: (sa_available[a], a)):
            tnow = sa_available[sa]
            candidates = []
            for key in list(unscheduled):
                if assignment[key] != sa:
                    continue
                task = task_by_key[key]
                pred_finish = 0.0
                if task.pred is not None:
                    if task.pred not in scheduled:
                        continue
                    pred_finish = scheduled[task.pred][2]
                rel = max(task.release, pred_finish)
                if rel <= tnow + EPS:
                    candidates.append(task)
            if not candidates:
                continue
            task = min(candidates, key=lambda x: (x.deadline, x.release, x.layer_id, x.key.ordinal))
            st = max(tnow, task.release, scheduled[task.pred][2] if task.pred in scheduled else 0.0)
            ft = st + _task_latency(config, task, sa)
            if ft > task.deadline + EPS:
                return None
            scheduled[task.key] = (sa, st, ft)
            unscheduled.remove(task.key)
            sa_available[sa] = ft
            progress = True
        if progress:
            continue
        next_times = []
        for sa in range(config.num_sas):
            for key in unscheduled:
                if assignment[key] != sa:
                    continue
                task = task_by_key[key]
                pred_ready = 0.0
                if task.pred is not None:
                    if task.pred not in scheduled:
                        continue
                    pred_ready = scheduled[task.pred][2]
                next_times.append(max(sa_available[sa], task.release, pred_ready))
        if not next_times:
            return None
        nxt = min((x for x in next_times if x >= min(sa_available) - EPS))
        for sa in range(config.num_sas):
            if sa_available[sa] < nxt:
                sa_available[sa] = nxt
    return scheduled

def _offline_schedule_ff(env, config, horizon_end: float) -> List[RefSlot]:
    """
    Tang et al.'s Fastest First (FF) periodic scheduler, adapted to DNN layers.
    For each hard-periodic layer instance, resources are tested in ascending
    execution time; the fastest resource that preserves local non-preemptive EDF
    feasibility is selected.
    """
    tasks = _generate_hrt_tasks(config, horizon_end)
    assignment: Dict[SlotKey, int] = {}
    schedule: Dict[SlotKey, Tuple[int, float, float]] = {}
    for task in sorted(tasks, key=lambda t: (t.release, t.deadline, t.customer_id, t.instance_id, t.layer_id)):
        feasible_choice = None
        for sa in sorted(range(config.num_sas), key=lambda a: (_task_latency(config, task, a), a)):
            cand_assignment = dict(assignment)
            cand_assignment[task.key] = sa
            cand_schedule = _simulate_partitioned_edf(config, [t for t in tasks if t.key in cand_assignment], cand_assignment)
            if cand_schedule is not None:
                feasible_choice = (sa, cand_schedule)
                break
        if feasible_choice is None:
            raise RuntimeError(f'Tang-FF could not assign HRT task job={task.job_id}, layer={task.layer_id}, release={task.release}, deadline={task.deadline}.')
        (sa, schedule) = feasible_choice
        assignment[task.key] = sa
    return _slots_from_schedule(config, tasks, schedule)

def _offline_schedule_h_edf(env, config, horizon_end: float) -> List[RefSlot]:
    """
    Tang et al.'s centralized H-EDF periodic scheduler, adapted to layer-level
    HRT tasks with chain dependencies.  Ready instances are kept in a global EDF
    queue; when a resource becomes idle, the scheduler scans the queue and
    dispatches the first task that can finish on that resource before deadline.
    """
    tasks = _generate_hrt_tasks(config, horizon_end)
    task_by_key = {t.key: t for t in tasks}
    unscheduled: Set[SlotKey] = set(task_by_key)
    scheduled: Dict[SlotKey, Tuple[int, float, float]] = {}
    sa_available = [0.0] * int(config.num_sas)
    while unscheduled:
        tnow = min(sa_available)
        idle_sas = [a for a in range(config.num_sas) if sa_available[a] <= tnow + EPS]
        progress = False
        for sa in idle_sas:
            ready = []
            for key in unscheduled:
                task = task_by_key[key]
                pred_finish = 0.0
                if task.pred is not None:
                    if task.pred not in scheduled:
                        continue
                    pred_finish = scheduled[task.pred][2]
                if max(task.release, pred_finish) <= tnow + EPS:
                    ready.append(task)
            ready.sort(key=lambda t: (t.deadline, t.release, t.layer_id, t.key.ordinal))
            chosen = None
            for task in ready:
                st = max(tnow, task.release, scheduled[task.pred][2] if task.pred in scheduled else 0.0)
                ft = st + _task_latency(config, task, sa)
                if ft <= task.deadline + EPS:
                    chosen = (task, st, ft)
                    break
                fastest = min(range(config.num_sas), key=lambda a: _task_latency(config, task, a))
                earliest_fastest_start = max(tnow, task.release, scheduled[task.pred][2] if task.pred in scheduled else 0.0, sa_available[fastest])
                if earliest_fastest_start + _task_latency(config, task, fastest) > task.deadline + EPS:
                    raise RuntimeError(f'Tang-H-EDF rejects an HRT task: job={task.job_id}, layer={task.layer_id}, release={task.release}, deadline={task.deadline}.')
            if chosen is not None:
                (task, st, ft) = chosen
                scheduled[task.key] = (sa, st, ft)
                unscheduled.remove(task.key)
                sa_available[sa] = ft
                progress = True
        if progress:
            continue
        next_events = []
        for key in unscheduled:
            task = task_by_key[key]
            next_events.append(task.release)
            if task.pred is not None and task.pred in scheduled:
                next_events.append(scheduled[task.pred][2])
        busy_events = [x for x in sa_available if x > tnow + EPS]
        if busy_events:
            next_events.append(min(busy_events))
        future = [x for x in next_events if x > tnow + EPS]
        if not future:
            raise RuntimeError('Tang-H-EDF got stuck while building the offline HRT schedule.')
        nxt = min(future)
        for sa in range(config.num_sas):
            if sa_available[sa] <= tnow + EPS:
                sa_available[sa] = nxt
    return _slots_from_schedule(config, tasks, scheduled)

def _offline_reference_slots(env, config, horizon_end: float) -> List[RefSlot]:
    """Build Tang et al.'s offline hard-periodic schedule internally."""
    kind = _hard_scheduler_kind(config)
    if kind == 'ff':
        return _offline_schedule_ff(env, config, horizon_end)
    return _offline_schedule_h_edf(env, config, horizon_end)

def _reference_slots_for_period(env, config, period_start: float, period_end: float) -> Tuple[List[RefSlot], List[RefSlot]]:
    """
    Return (all_offline_slots, current_period_slots).  The schedule is built
    internally with H-EDF or FF, not read from feasible_scheduling.yaml.
    """
    horizon_end = _offline_horizon_end(env, config, period_end)
    all_slots = _offline_reference_slots(env, config, horizon_end)
    current_slots = [s for s in all_slots if s.start < period_end - EPS and s.finish > period_start + EPS]
    return (all_slots, current_slots)

def _lcm_int(values: List[int]) -> int:
    l = 1
    for v in values:
        v = int(round(v))
        if v <= 0:
            raise RuntimeError(f'Invalid period in HRT task set: {v}')
        l = abs(l * v) // math.gcd(l, v)
    return l

def _tang_hyperperiod(config) -> float:
    periods = [int(round(float(x[2]))) for x in getattr(config, 'hrt_customers', [])]
    if not periods:
        return 1.0
    return float(_lcm_int(periods))

def _max_hrt_relative_deadline(config) -> float:
    if not getattr(config, 'hrt_customers', []):
        return 0.0
    return float(max((float(x[3]) for x in config.hrt_customers)))

def _clone_slot_with_offset(s: RefSlot, offset: float, copy_idx: int) -> RefSlot:
    key = SlotKey(job_id=s.key.job_id, layer_id=s.key.layer_id, deadline=float(s.key.deadline + offset), ordinal=int(s.key.ordinal + copy_idx * 100000000))
    return RefSlot(key=key, sa=int(s.sa), job_id=int(s.job_id), layer_id=int(s.layer_id), start=float(s.start + offset), finish=float(s.finish + offset), deadline=float(s.deadline + offset), release=float(s.release + offset), customer_id=int(s.customer_id), instance_id=int(s.instance_id + copy_idx * 100000000), est=float(max(s.release + offset, s.start + offset)), lst=float(max(s.release + offset, s.start + offset)))

def _get_cached_base_offline_slots(env, config) -> Tuple[List[RefSlot], float, float]:
    """
    Build Tang's offline hard schedule once over one hyperperiod and reuse it.

    The previous version rebuilt the FF/H-EDF schedule from time 0 up to the
    current look-ahead horizon at every simulation step.  With FF this is
    catastrophic, because each newly generated HRT layer triggers a feasibility
    re-simulation of all previously assigned layers.  Tang's hard workload is
    periodic, so the correct implementation is to build the hard-periodic
    schedule over one hyperperiod and then repeat it.
    """
    kind = _hard_scheduler_kind(config)
    H = _tang_hyperperiod(config)
    maxD = _max_hrt_relative_deadline(config)
    key = (kind, H, tuple(((int(c), int(j), float(p), float(d)) for (c, j, p, d) in getattr(config, 'hrt_customers', []))))
    cache = getattr(env, '_tang_base_schedule_cache', None)
    if cache is not None and cache.get('key') == key:
        return (cache['slots'], cache['hyperperiod'], cache['max_deadline'])
    slots = _offline_reference_slots(env, config, H)
    if not slots and getattr(config, 'hrt_customers', []):
        raise RuntimeError('Tang offline scheduler generated an empty HRT schedule over the hyperperiod.')
    env._tang_base_schedule_cache = {'key': key, 'slots': slots, 'hyperperiod': H, 'max_deadline': maxD}
    return (slots, H, maxD)

def _repeat_base_slots_for_window(base_slots: List[RefSlot], hyperperiod: float, max_deadline: float, period_start: float, horizon_end: float) -> List[RefSlot]:
    if hyperperiod <= 0:
        raise RuntimeError('Invalid Tang hyperperiod.')
    if not base_slots:
        return []
    first_copy = math.floor((period_start - max_deadline - hyperperiod) / hyperperiod) - 1
    last_copy = math.ceil((horizon_end + max_deadline + hyperperiod) / hyperperiod) + 1
    all_slots: List[RefSlot] = []
    for k in range(int(first_copy), int(last_copy) + 1):
        offset = k * hyperperiod
        for s in base_slots:
            c = _clone_slot_with_offset(s, offset, k)
            if c.finish <= period_start + EPS:
                continue
            if c.release >= horizon_end + max_deadline + hyperperiod + EPS:
                continue
            all_slots.append(c)
    all_slots.sort(key=lambda x: (x.start, x.finish, x.sa, x.deadline, x.job_id, x.layer_id, x.key.ordinal))
    return all_slots

def _reference_slots_for_period(env, config, period_start: float, period_end: float) -> Tuple[List[RefSlot], List[RefSlot]]:
    """Return repeated Tang offline slots for the current planning window."""
    horizon_end = _offline_horizon_end(env, config, period_end)
    (base_slots, H, maxD) = _get_cached_base_offline_slots(env, config)
    all_slots = _repeat_base_slots_for_window(base_slots, H, maxD, period_start, horizon_end)
    current_slots = [s for s in all_slots if s.start < period_end - EPS and s.finish > period_start + EPS]
    return (all_slots, current_slots)

def _compute_lst_with_machine_and_chain_constraints(slots: List[RefSlot], env, config, horizon_end: float) -> None:
    """
    Compute latest-start times by preserving:
      1) reference order on each accelerator;
      2) layer-chain precedence for HRT jobs with the same job_id/deadline.

    This is stricter than a machine-only backward pass and avoids fabricating
    slack that would violate DNN layer dependencies.
    """
    succ: Dict[SlotKey, List[SlotKey]] = {s.key: [] for s in slots}
    by_key = {s.key: s for s in slots}
    by_sa: Dict[int, List[RefSlot]] = {m: [] for m in range(config.num_sas)}
    for s in slots:
        by_sa.setdefault(s.sa, []).append(s)
    for arr in by_sa.values():
        arr.sort(key=lambda x: (x.start, x.finish, x.job_id, x.layer_id, x.key.ordinal))
        for (a, b) in zip(arr, arr[1:]):
            succ[a.key].append(b.key)
    by_job_deadline: Dict[Tuple[int, float], List[RefSlot]] = {}
    for s in slots:
        by_job_deadline.setdefault((s.job_id, s.deadline), []).append(s)
    for arr in by_job_deadline.values():
        arr.sort(key=lambda x: (x.layer_id, x.start, x.finish, x.key.ordinal))
        for (a, b) in zip(arr, arr[1:]):
            if b.layer_id == a.layer_id + 1:
                succ[a.key].append(b.key)
    for s in slots:
        s.lst = max(s.est, min(s.deadline, horizon_end) - s.duration)
    changed = True
    while changed:
        changed = False
        for s in sorted(slots, key=lambda x: (x.deadline, x.start, x.layer_id), reverse=True):
            latest_finish = min(s.deadline, horizon_end)
            if succ[s.key]:
                latest_finish = min(latest_finish, min((by_key[k].lst for k in succ[s.key])))
            new_lst = max(s.est, latest_finish - s.duration)
            if abs(new_lst - s.lst) > EPS:
                s.lst = new_lst
                changed = True

def _match_hrt_slots_to_rq(env, slots: List[RefSlot]) -> Dict[int, RefSlot]:
    """
    Match current ready-queue HRT sub-jobs to reference slots.  Matching uses
    (job_id, layer_id) and then the closest absolute deadline.  This prevents
    silently consuming an arbitrary slot when multiple instances of the same
    layer appear in the hyperperiod.
    """
    remaining = list(slots)
    match: Dict[int, RefSlot] = {}
    env._hardsoft_failed_hrt_jobs = set()
    for (i, sj) in sorted([(i, sj) for (i, sj) in enumerate(env.rq) if sj.is_hrt and (not _is_already_committed(sj, env.current_time))], key=lambda x: (x[1].absolute_deadline, x[1].job_id, x[1].layer_id)):
        if sj.absolute_deadline <= env.current_time + EPS:
            continue
        candidates = [s for s in remaining if s.job_id == sj.job_id and s.layer_id == sj.layer_id]
        if not candidates:
            if sj.absolute_deadline <= env.current_time + env.scheduling_period + EPS:
                env._hardsoft_failed_hrt_jobs.add(sj.job_instance_id)
            continue
        best = min(candidates, key=lambda s: (abs(float(s.deadline) - float(sj.absolute_deadline)), s.start))
        match[i] = best
        remaining.remove(best)
    return match

def _initial_fixed_intervals(env, period_start: float) -> List[Interval]:
    """Intervals already committed before this decision step and still active."""
    fixed: List[Interval] = []
    for (i, sj) in enumerate(env.rq):
        if _is_already_committed(sj, period_start):
            fixed.append(Interval(sa=int(sj.sa_id), start=float(sj.start_time), finish=float(sj.finish_time), kind='BUSY' if not sj.is_hrt else 'HRT', rq_idx=i))
    return fixed

def _intervals_to_windows(intervals: List[Interval], env, period_start: float, period_end: float, num_sas: int) -> List[SlackWindow]:
    windows: List[SlackWindow] = []
    by_sa: Dict[int, List[Interval]] = {m: [] for m in range(num_sas)}
    for itv in intervals:
        by_sa.setdefault(itv.sa, []).append(itv)
    for sa in range(num_sas):
        cursor = max(period_start, float(env.sa_busy_until[sa]))
        arr = sorted(by_sa.get(sa, []), key=lambda x: (x.start, x.finish))
        for itv in arr:
            if itv.finish <= cursor + EPS:
                cursor = max(cursor, itv.finish)
                continue
            if cursor < itv.start - EPS:
                windows.append(SlackWindow(sa=sa, start=cursor, end=itv.start))
            cursor = max(cursor, itv.finish)
        if cursor < period_end - EPS:
            windows.append(SlackWindow(sa=sa, start=cursor, end=period_end))
    windows.sort(key=lambda w: (w.start, w.end, w.sa))
    return windows

def _build_hrt_intervals(env, config, hrt_match: Dict[int, RefSlot], placement: Dict[SlotKey, float]) -> Dict[int, Interval]:
    hrt_intervals: Dict[int, Interval] = {}
    for (rq_idx, slot) in hrt_match.items():
        st = float(placement[slot.key])
        ft = st + slot.duration
        hrt_intervals[rq_idx] = Interval(sa=slot.sa, start=st, finish=ft, kind='HRT', rq_idx=rq_idx, slot_key=slot.key)
    return hrt_intervals

def _dependency_release_or_none(env, rq_idx: int, planned: Dict[int, Interval], period_start: float) -> Optional[float]:
    """
    Return the earliest start imposed by dependencies.  If a dependency is not
    completed and is not planned in the current strict plan, return None so the
    sub-job cannot be scheduled speculatively.
    """
    sj = env.rq[rq_idx]
    release = max(period_start, float(sj.arrival_time))
    for dep in getattr(sj, 'dependencies', []) or []:
        if dep in planned:
            release = max(release, planned[dep].finish)
        elif 0 <= dep < len(env.rq):
            dep_sj = env.rq[dep]
            if dep_sj.finish_time is not None and dep_sj.finish_time <= period_start + EPS:
                release = max(release, float(dep_sj.finish_time))
            else:
                return None
        else:
            return None
    return release

def _try_place_subjob_in_windows(config, sj, release: float, windows: List[SlackWindow], mode: str) -> Optional[Tuple[int, float, float, int]]:
    """
    Return (window_index, start, finish, sa) for one sub-job.
      RC: first feasible resource window in chronological order.
      TC/TCSD: feasible placement with earliest finish.
    """
    best = None
    for (wi, w) in enumerate(windows):
        lat = _latency(config, sj, w.sa)
        st = max(w.start, release)
        ft = st + lat
        if ft <= w.end + EPS:
            cand = (wi, st, ft, w.sa)
            if mode == 'rc':
                return cand
            if best is None or (ft, st, w.sa) < (best[2], best[1], best[3]):
                best = cand
    return best

def _commit_window_placement(windows: List[SlackWindow], wi: int, ft: float) -> None:
    windows[wi].start = ft

def _compact_hrt_earlier_on_sa(env, slots: List[RefSlot], placement: Dict[SlotKey, float], fixed_intervals: List[Interval], accepted_be: Dict[int, Interval], sa: int, period_start: float, period_end: float) -> Optional[Dict[SlotKey, float]]:
    """
    TCSD local defragmentation candidate: left-compact HRT segments only on
    the selected accelerator, without migrating them and without overlapping
    already accepted BE intervals. This is triggered lazily only when normal TC
    cannot place the current soft request.
    """
    new_placement = dict(placement)
    hrt_on_sa = sorted([s for s in slots if s.sa == sa], key=lambda s: (s.start, s.finish, s.layer_id, s.key.ordinal))
    fixed_on_sa = sorted([x for x in fixed_intervals if x.sa == sa] + [x for x in accepted_be.values() if x.sa == sa], key=lambda x: (x.start, x.finish))
    cursor = max(period_start, float(env.sa_busy_until[sa]))

    def advance_past_fixed(st: float, dur: float) -> float:
        changed = True
        while changed:
            changed = False
            for x in fixed_on_sa:
                if st + dur <= x.start + EPS or st >= x.finish - EPS:
                    continue
                st = x.finish
                changed = True
        return st
    for s in hrt_on_sa:
        st = max(cursor, s.est)
        st = advance_past_fixed(st, s.duration)
        ft = st + s.duration
        if st < s.est - EPS or st > s.lst + EPS or ft > s.deadline + EPS or (ft > period_end + max(0.0, s.duration) + EPS):
            return None
        new_placement[s.key] = st
        cursor = ft
    return new_placement

def _validate_no_overlap(intervals: List[Interval]) -> None:
    by_sa: Dict[int, List[Interval]] = {}
    for itv in intervals:
        by_sa.setdefault(itv.sa, []).append(itv)
    for (sa, arr) in by_sa.items():
        arr.sort(key=lambda x: (x.start, x.finish))
        for (a, b) in zip(arr, arr[1:]):
            if a.finish > b.start + EPS:
                raise AssertionError(f'HardSoft strict schedule overlap on SA {sa}: {a.kind}@{a.rq_idx}[{a.start},{a.finish}] vs {b.kind}@{b.rq_idx}[{b.start},{b.finish}]')

def _validate_dependencies(env, plan: Plan, period_start: float) -> None:
    planned: Dict[int, Interval] = {}
    planned.update(plan.hrt)
    planned.update(plan.be)
    for (rq_idx, itv) in planned.items():
        sj = env.rq[rq_idx]
        for dep in getattr(sj, 'dependencies', []) or []:
            if dep in planned:
                if planned[dep].finish > itv.start + EPS:
                    raise AssertionError(f'Dependency violation: dep {dep} finishes at {planned[dep].finish}, but {rq_idx} starts at {itv.start}.')
            elif 0 <= dep < len(env.rq):
                dep_sj = env.rq[dep]
                if dep_sj.finish_time is None or dep_sj.finish_time > itv.start + EPS:
                    raise AssertionError(f'Unplanned unfinished dependency {dep} for planned sub-job {rq_idx}.')

def _be_groups(env) -> List[Tuple[int, List[int]]]:
    """Return remaining BE job instances in FCFS order, with layers in order."""
    groups: Dict[int, List[int]] = {}
    first_arrival: Dict[int, float] = {}
    first_deadline: Dict[int, float] = {}
    for (i, sj) in enumerate(env.rq):
        if sj.is_hrt or _is_already_committed(sj, env.current_time):
            continue
        groups.setdefault(sj.job_instance_id, []).append(i)
        first_arrival.setdefault(sj.job_instance_id, float(sj.arrival_time))
        first_deadline.setdefault(sj.job_instance_id, float(sj.absolute_deadline))
    ordered = sorted(groups.keys(), key=lambda jid: (first_arrival[jid], first_deadline[jid], jid))
    return [(jid, sorted(groups[jid], key=lambda i: (env.rq[i].layer_id, i))) for jid in ordered]

def _try_schedule_be_group(env, config, group_indices: List[int], windows: List[SlackWindow], planned: Dict[int, Interval], mode: str, period_start: float) -> Optional[Tuple[Dict[int, Interval], List[SlackWindow]]]:
    """Try to place all remaining layers of one BE request. Commit atomically."""
    local_windows = deepcopy(windows)
    local_plan: Dict[int, Interval] = {}
    local_planned = dict(planned)
    for idx in group_indices:
        sj = env.rq[idx]
        release = _dependency_release_or_none(env, idx, local_planned, period_start)
        if release is None:
            return None
        found = _try_place_subjob_in_windows(config, sj, release, local_windows, mode)
        if found is None:
            return None
        (wi, st, ft, sa) = found
        itv = Interval(sa=sa, start=st, finish=ft, kind='BE', rq_idx=idx)
        local_plan[idx] = itv
        local_planned[idx] = itv
        _commit_window_placement(local_windows, wi, ft)
    return (local_plan, local_windows)

def _strict_plan(env, config, mode: str) -> Plan:
    mode = mode.lower()
    if mode not in ('rc', 'tc', 'tcsd'):
        raise ValueError(f'Unsupported hardsoft_mode={mode}')
    (period_start, period_end, _) = _period_bounds(env)
    (all_slots, _) = _reference_slots_for_period(env, config, period_start, period_end)
    lst_horizon = max([period_end] + [s.deadline for s in all_slots] + [s.finish for s in all_slots])
    _compute_lst_with_machine_and_chain_constraints(all_slots, env, config, lst_horizon)
    slots = [s for s in all_slots if s.release <= period_start + EPS and s.deadline > period_start + EPS and (s.lst < period_end - EPS)]
    hrt_match = _match_hrt_slots_to_rq(env, slots)
    placement: Dict[SlotKey, float] = {s.key: s.lst for s in slots}
    fixed_intervals = _initial_fixed_intervals(env, period_start)
    hrt_intervals = _build_hrt_intervals(env, config, hrt_match, placement)
    plan = Plan(hrt=dict(hrt_intervals), mode=mode)
    plan.failed_hrt_jobs.update(getattr(env, '_hardsoft_failed_hrt_jobs', set()))
    planned: Dict[int, Interval] = {}
    planned.update(plan.hrt)

    def rebuild_windows() -> List[SlackWindow]:
        all_intervals = fixed_intervals + list(plan.hrt.values()) + list(plan.be.values())
        return _intervals_to_windows(all_intervals, env, period_start, period_end, config.num_sas)
    windows = rebuild_windows()
    for (job_instance_id, indices) in _be_groups(env):
        attempt = _try_schedule_be_group(env, config, indices, windows, planned, 'rc' if mode == 'rc' else 'tc', period_start)
        if attempt is not None:
            (local_plan, windows) = attempt
            plan.be.update(local_plan)
            planned.update(local_plan)
            continue
        if mode == 'tcsd':
            best_candidate = None
            for sa in range(config.num_sas):
                new_placement = _compact_hrt_earlier_on_sa(env, slots, placement, fixed_intervals, plan.be, sa, period_start, period_end)
                if new_placement is None:
                    continue
                new_hrt = _build_hrt_intervals(env, config, hrt_match, new_placement)
                candidate_intervals = fixed_intervals + list(new_hrt.values()) + list(plan.be.values())
                candidate_windows = _intervals_to_windows(candidate_intervals, env, period_start, period_end, config.num_sas)
                candidate_planned = dict(new_hrt)
                candidate_planned.update(plan.be)
                attempt2 = _try_schedule_be_group(env, config, indices, candidate_windows, candidate_planned, 'tc', period_start)
                if attempt2 is None:
                    continue
                (local_plan, updated_windows) = attempt2
                completion = max((x.finish for x in local_plan.values()))
                cand = (completion, sa, new_placement, new_hrt, local_plan, updated_windows)
                if best_candidate is None or cand[:2] < best_candidate[:2]:
                    best_candidate = cand
            if best_candidate is not None:
                (_, _, placement, new_hrt, local_plan, windows) = best_candidate
                plan.hrt = new_hrt
                plan.be.update(local_plan)
                planned = dict(plan.hrt)
                planned.update(plan.be)
                continue
        if _strict_reject(config):
            plan.rejected_be_jobs.add(job_instance_id)
        else:
            plan.deferred_be_jobs.add(job_instance_id)
    all_planned_intervals = fixed_intervals + list(plan.hrt.values()) + list(plan.be.values())
    _validate_no_overlap(all_planned_intervals)
    _validate_dependencies(env, plan, period_start)
    for (rq_idx, itv) in plan.hrt.items():
        slot = hrt_match[rq_idx]
        if itv.sa != slot.sa:
            raise AssertionError('TCSD migrated an HRT segment, which is forbidden.')
        if itv.start < slot.est - EPS or itv.start > slot.lst + EPS:
            raise AssertionError('HRT segment moved outside its [EST,LST] window.')
    return plan

def _mark_rejected_jobs(env, rejected_job_ids: Set[int]) -> None:
    """Mark rejected BE requests so they are removed and counted as misses."""
    if not rejected_job_ids:
        return
    for sj in env.rq:
        if not sj.is_hrt and sj.job_instance_id in rejected_job_ids:
            pass

def _apply_plan_to_rq(env, plan: Plan) -> None:
    for (i, sj) in enumerate(env.rq):
        if sj.start_time is not None and sj.finish_time is not None:
            continue
        sj.start_time = None
        sj.finish_time = None
        sj.sa_id = None
    for mapping in (plan.hrt, plan.be):
        for (rq_idx, itv) in mapping.items():
            sj = env.rq[rq_idx]
            if sj.start_time is not None and sj.finish_time is not None:
                continue
            sj.sa_id = int(itv.sa)
            sj.start_time = float(itv.start)
            sj.finish_time = float(itv.finish)

def _update_env_after_strict_plan(env, config, plan: Plan, period_start: float, period_end: float):
    """
    Complete/retain/drop sub-jobs after applying a strict plan.  A sub-job is
    counted as executed only if it actually completes by the new current time.
    Scheduled-but-not-yet-completed work remains in the ready queue and keeps
    the corresponding accelerator busy via sa_busy_until.
    """
    reward = 0.0
    hrt_feasible = True
    env.old_rq = deepcopy(env.rq)
    env.rq.clear()
    dependencies_conversion_table: Dict[int, int] = {}
    rejected = set(plan.rejected_be_jobs)
    failed_hrt = set(plan.failed_hrt_jobs)
    for (old_idx, sj) in enumerate(env.old_rq):
        rejected_job = not sj.is_hrt and sj.job_instance_id in rejected
        failed_hrt_job = sj.is_hrt and sj.job_instance_id in failed_hrt
        if rejected_job:
            if sj.is_last:
                env.deadline_misses_be += 1
            continue
        if failed_hrt_job:
            if sj.is_last:
                env.deadline_misses_hrt += 1
                hrt_feasible = False
            continue
        if sj.start_time is not None and sj.finish_time is not None:
            if sj.start_time <= env.current_time + EPS:
                env.sa_busy_until[sj.sa_id] = max(env.sa_busy_until[sj.sa_id], sj.finish_time)
            if sj.finish_time <= env.current_time + EPS:
                env.executed_sjs += 1
                if sj.finish_time >= env.sa_busy_until[sj.sa_id] - EPS:
                    env.sa_last_sj_executed[sj.sa_id] = (sj.job_instance_id, sj.job_id, sj.layer_id)
                if env.reward_slack_factor > 0:
                    factor = 5 if sj.is_hrt else 1
                    reward += factor * env.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                if sj.is_last:
                    tmp = env.calculate_lastsj_reward(sj)
                    reward += tmp
                    if sj.is_hrt and sj.finish_time > sj.absolute_deadline + EPS:
                        hrt_feasible = False
            else:
                env.rq.append(sj)
                dependencies_conversion_table[old_idx] = len(env.rq) - 1
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
            dependencies_conversion_table[old_idx] = len(env.rq) - 1
    env.imminents.clear()
    for (new_idx, sj) in enumerate(env.rq):
        sj.dependencies = [dependencies_conversion_table[d] for d in sj.dependencies if d in dependencies_conversion_table]
        sj.children = [dependencies_conversion_table[d] for d in sj.children if d in dependencies_conversion_table]
        if len(sj.dependencies) == 0 and (not _is_already_committed(sj, env.current_time)):
            env.imminents.add(new_idx)
    for a in range(env.num_sas):
        env.sa_busy_until[a] = max(env.sa_busy_until[a], env.current_time)
    return (reward, hrt_feasible)

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

def _hardsoft_strict_step(env, config, mode: str):
    (period_start, period_end, _) = _advance_time(env)
    if len(env.rq) == 0:
        (actor_next_state, rb_next_state, done) = _next_state(env)
        return (actor_next_state, rb_next_state, 0.0, done, True, 0.0, Plan(mode=mode))
    env.current_time = period_start
    plan = _strict_plan(env, config, mode)
    env.current_time = period_end
    _apply_plan_to_rq(env, plan)
    (reward, hrt_feasible) = _update_env_after_strict_plan(env, config, plan, period_start, period_end)
    (actor_next_state, rb_next_state, done) = _next_state(env)
    env.hardsoft_last_plan = plan
    env.hardsoft_accepted_be_sj = len(plan.be)
    env.hardsoft_rejected_be_jobs = len(plan.rejected_be_jobs)
    env.hardsoft_deferred_be_jobs = len(plan.deferred_be_jobs)
    env.hardsoft_failed_hrt_jobs = len(plan.failed_hrt_jobs)
    return (actor_next_state, rb_next_state, reward, done, hrt_feasible, plan)

def _empty_action(num_sas: int) -> np.ndarray:
    return np.zeros((0, 1 + num_sas), dtype=np.float32)

def _action_rc(env, config):
    raise RuntimeError('Use hardsoft(config) strict runner; RC cannot be faithfully expressed as a plain action matrix.')

def _action_tc(env, config):
    raise RuntimeError('Use hardsoft(config) strict runner; TC cannot be faithfully expressed as a plain action matrix.')

def _action_tcsd(env, config):
    raise RuntimeError('Use hardsoft(config) strict runner; TCSD cannot be faithfully expressed as a plain action matrix.')

def hardsoft(config):
    mode = getattr(config, 'hardsoft_mode', 'tc').lower()
    assert mode in ('rc', 'tc', 'tcsd'), f'hardsoft_mode must be rc/tc/tcsd, got {mode}'
    periodic_scheduler = _hard_scheduler_kind(config).upper()
    print(f'[HardSoft/Tang2011 STRICT] HRT={periodic_scheduler} | BE={mode.upper()} | direct slack-window execution')
    print('[HardSoft/Tang2011 STRICT] HRT offline schedule is generated internally; feasible_scheduling.yaml is not used.')
    print('[HardSoft/Tang2011 STRICT] rejected/deferred BE sub-jobs are NOT passed to Environment.step().')
    env = Environment(config, seed=config.seed)
    env.prepend_sa_remaining_time = False
    env.use_sa_assignement_heuristic = False
    env.hrt_step_mode = 'all'
    env.reset()
    total_accepted_be = 0
    total_rejected_be = 0
    total_deferred_be = 0
    for step_idx in trange(config.num_steps):
        (_, _, _, _, hrt_feasible, plan) = _hardsoft_strict_step(env, config, mode)
        total_accepted_be += len(plan.be)
        total_rejected_be += len(plan.rejected_be_jobs)
        total_deferred_be += len(plan.deferred_be_jobs)
        total_deadline_hits = env.deadline_hits_be + env.deadline_hits_hrt
        total_arrived_jobs = env.total_arrived_jobs_be + env.total_arrived_jobs_hrt
        slast = _safe_ratio(total_deadline_hits, total_arrived_jobs)
        slast_be = _safe_ratio(env.deadline_hits_be, env.total_arrived_jobs_be)
        slast_hrt = _safe_ratio(env.deadline_hits_hrt, env.total_arrived_jobs_hrt)
        if env.deadline_misses_hrt and bool(getattr(config, 'hardsoft_abort_on_hrt_miss', False)):
            print('[PROBLEM] HRT deadline miss in strict HardSoft baseline ->', env.deadline_misses_hrt)
            raise RuntimeError('HardSoft strict baseline violated HRT correctness.')
        if _hardsoft_debug_enabled(config):
            tqdm.write(f'Step {step_idx + 1}: BE accepted_sj={len(plan.be)}, rejected_jobs={len(plan.rejected_be_jobs)}, deferred_jobs={len(plan.deferred_be_jobs)}, failed_hrt_jobs={len(plan.failed_hrt_jobs)}')
        tqdm.write('Step:\t{}\tSlast:\t{:0.2f}%\tSlast_BE:\t{:0.2f}%\tSlast_HRT:\t{:0.2f}%\tHRT/BE MISS: {},{}\tBE accepted_sj step/cum: {}/{}\tBE rejected_jobs step/cum: {}/{}'.format(step_idx + 1, slast * 100, slast_be * 100, slast_hrt * 100, env.deadline_misses_hrt, env.deadline_misses_be, len(plan.be), total_accepted_be, len(plan.rejected_be_jobs), total_rejected_be))
    print(f'SLA Satisfaction Rate: {slast * 100:.2f}%')
    print(f'SLA Satisfaction Rate BE: {slast_be * 100:.2f}% ({env.deadline_hits_be}/{env.total_arrived_jobs_be})')
    print(f'SLA Satisfaction Rate HRT: {slast_hrt * 100:.2f}% ({env.deadline_hits_hrt}/{env.total_arrived_jobs_hrt})')
    print('Throughput:', env.executed_sjs * 1000000000.0 / env.current_time if env.current_time > 0 else 0.0)
    print(f'HardSoft accepted BE sub-jobs: {total_accepted_be}')
    print(f'HardSoft rejected BE jobs: {total_rejected_be}')
    print(f'HardSoft deferred BE jobs: {total_deferred_be}')
    if env.num_tenants > 0:
        slasr_per_tenant = [env.deadline_hits_per_tenant[i] / max(1, env.arrived_jobs_per_tenant[i]) if env.arrived_jobs_per_tenant[i] > 0 else 0.0 for i in range(env.num_tenants)]
        print('SLASR per tenant:', slasr_per_tenant)
        print('SLASR per tenant std:', np.std(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant min:', np.min(slasr_per_tenant) * 100, '%')
        print('SLASR per tenant max:', np.max(slasr_per_tenant) * 100, '%')

def _active_hrt_tasks_from_env(env) -> List[HRTTask]:
    """Create layer-level HRT tasks from the current ready queue.

    Only not-yet-committed, not-expired HRT sub-jobs are included.  Already
    committed HRT sub-jobs are represented separately as fixed intervals by
    _initial_fixed_intervals().  The runtime job_instance_id is stored as the
    HRTTask.instance_id so matching is exact and cannot swap two instances with
    the same (job_id, layer_id, deadline).
    """
    key_by_rq_idx: Dict[int, SlotKey] = {}
    tasks_by_rq_idx: Dict[int, HRTTask] = {}
    for (i, sj) in enumerate(env.rq):
        if not sj.is_hrt:
            continue
        if _is_already_committed(sj, env.current_time):
            continue
        if sj.absolute_deadline <= env.current_time + EPS:
            continue
        key = SlotKey(job_id=int(sj.job_id), layer_id=int(sj.layer_id), deadline=float(sj.absolute_deadline), ordinal=int(sj.job_instance_id) * 1000000 + int(sj.layer_id))
        key_by_rq_idx[i] = key
    for (i, sj) in enumerate(env.rq):
        if i not in key_by_rq_idx:
            continue
        pred_key = None
        for dep in getattr(sj, 'dependencies', []) or []:
            if dep in key_by_rq_idx:
                pred_key = key_by_rq_idx[dep]
                break
        tasks_by_rq_idx[i] = HRTTask(key=key_by_rq_idx[i], customer_id=-1, instance_id=int(sj.job_instance_id), job_id=int(sj.job_id), layer_id=int(sj.layer_id), release=float(sj.arrival_time), deadline=float(sj.absolute_deadline), pred=pred_key)
    return [tasks_by_rq_idx[i] for i in sorted(tasks_by_rq_idx)]

def _offline_schedule_ff_from_tasks(env, config, tasks: List[HRTTask]) -> List[RefSlot]:
    if not tasks:
        return []
    assignment: Dict[SlotKey, int] = {}
    schedule: Dict[SlotKey, Tuple[int, float, float]] = {}
    for task in sorted(tasks, key=lambda t: (t.release, t.deadline, t.instance_id, t.layer_id)):
        feasible_choice = None
        for sa in sorted(range(config.num_sas), key=lambda a: (_task_latency(config, task, a), a)):
            cand_assignment = dict(assignment)
            cand_assignment[task.key] = sa
            active_tasks = [t for t in tasks if t.key in cand_assignment]
            cand_schedule = _simulate_partitioned_edf(config, active_tasks, cand_assignment)
            if cand_schedule is not None:
                feasible_choice = (sa, cand_schedule)
                break
        if feasible_choice is None:
            continue
        (sa, schedule) = feasible_choice
        assignment[task.key] = sa
    return _slots_from_schedule(config, [t for t in tasks if t.key in schedule], schedule)

def _offline_schedule_h_edf_from_tasks(env, config, tasks: List[HRTTask]) -> List[RefSlot]:
    if not tasks:
        return []
    task_by_key = {t.key: t for t in tasks}
    unscheduled: Set[SlotKey] = set(task_by_key)
    scheduled: Dict[SlotKey, Tuple[int, float, float]] = {}
    sa_available = [float(env.sa_busy_until[a]) for a in range(config.num_sas)]
    while unscheduled:
        tnow = min(sa_available)
        idle_sas = [a for a in range(config.num_sas) if sa_available[a] <= tnow + EPS]
        progress = False
        for sa in idle_sas:
            ready = []
            for key in list(unscheduled):
                task = task_by_key[key]
                pred_finish = 0.0
                if task.pred is not None:
                    if task.pred not in scheduled:
                        continue
                    pred_finish = scheduled[task.pred][2]
                if max(task.release, pred_finish) <= tnow + EPS:
                    ready.append(task)
            ready.sort(key=lambda t: (t.deadline, t.release, t.layer_id, t.key.ordinal))
            chosen = None
            for task in ready:
                pred_finish = scheduled[task.pred][2] if task.pred in scheduled else 0.0
                st = max(tnow, task.release, pred_finish)
                ft = st + _task_latency(config, task, sa)
                if ft <= task.deadline + EPS:
                    chosen = (task, st, ft)
                    break
            if chosen is not None:
                (task, st, ft) = chosen
                scheduled[task.key] = (sa, st, ft)
                unscheduled.remove(task.key)
                sa_available[sa] = ft
                progress = True
        if progress:
            continue
        next_events = []
        for key in unscheduled:
            task = task_by_key[key]
            next_events.append(task.release)
            if task.pred is not None and task.pred in scheduled:
                next_events.append(scheduled[task.pred][2])
        busy_events = [x for x in sa_available if x > tnow + EPS]
        if busy_events:
            next_events.append(min(busy_events))
        future = [x for x in next_events if x > tnow + EPS]
        if not future:
            break
        nxt = min(future)
        for sa in range(config.num_sas):
            if sa_available[sa] <= tnow + EPS:
                sa_available[sa] = nxt
    return _slots_from_schedule(config, [t for t in tasks if t.key in scheduled], scheduled)

def _reference_slots_for_period(env, config, period_start: float, period_end: float) -> Tuple[List[RefSlot], List[RefSlot]]:
    """V6 override: build Tang HRT slots from actual active HRT instances.

    This avoids mismatches between a purely periodic config-generated schedule
    and RETNAS' runtime HRT arrivals.  It also means that matching uses the
    actual job_instance_id carried by Environment.
    """
    tasks = _active_hrt_tasks_from_env(env)
    if _hard_scheduler_kind(config) == 'ff':
        all_slots = _offline_schedule_ff_from_tasks(env, config, tasks)
    else:
        all_slots = _offline_schedule_h_edf_from_tasks(env, config, tasks)
    current_slots = [s for s in all_slots if s.start < period_end - EPS and s.finish > period_start + EPS]
    return (all_slots, current_slots)

def _match_hrt_slots_to_rq(env, slots: List[RefSlot]) -> Dict[int, RefSlot]:
    """V6 override: exact match by runtime job_instance_id when available."""
    remaining = list(slots)
    match: Dict[int, RefSlot] = {}
    env._hardsoft_failed_hrt_jobs = set()
    for (i, sj) in sorted([(i, sj) for (i, sj) in enumerate(env.rq) if sj.is_hrt and (not _is_already_committed(sj, env.current_time))], key=lambda x: (x[1].absolute_deadline, x[1].job_instance_id, x[1].layer_id)):
        if sj.absolute_deadline <= env.current_time + EPS:
            continue
        candidates = [s for s in remaining if s.instance_id == sj.job_instance_id and s.job_id == sj.job_id and (s.layer_id == sj.layer_id)]
        if not candidates:
            if sj.absolute_deadline <= env.current_time + env.scheduling_period + EPS:
                env._hardsoft_failed_hrt_jobs.add(sj.job_instance_id)
            continue
        best = min(candidates, key=lambda s: (s.start, s.finish, s.sa))
        match[i] = best
        remaining.remove(best)
    return match
