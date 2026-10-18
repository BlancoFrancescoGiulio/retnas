from copy import deepcopy
from dataclasses import replace
import numpy as np
from retnas.job import *
from retnas.safe_prefix import Operation, PeriodicReference, ReferenceScheduleError, check_safe_prefix, check_safe_continuation, validate_operations
MAX_RQ_SIZE = 192

class Environment:

    def __init__(self, config, seed=None, use_sa_assignement_heuristic=False):
        self.next_job_instance_id = 0
        self.num_sas = config.num_sas
        self.jobs = config.all_jobs
        self.possible_jobs = config.jobs
        self.possible_jobs_ids = config.possible_jobs
        self.qos_factor = config.qos_factor
        self.jobs_per_arrival = config.jobs_per_arrival
        self.alpha = config.alpha
        self.scale = config.dscale
        self.wcet_schedule = None
        self._hrt_reference = None
        self._committed_operations = {}
        self._pending_hrt_completions = {}
        self._settled_hrt_instances = set()
        self._pending_be_completions = {}
        self._settled_be_instances = set()
        self.hrt_consistency_reason = None
        self.hrt_guardian_rejections = {}
        self.wcet_schedule_by_period = None
        self.rng = np.random.default_rng(seed=seed)
        self.distribution = self.rng.pareto
        self.max_layer_id = max((len(job.layer_latencies) for job in self.possible_jobs))
        self.max_deadline = max((job.min_latency * self.qos_factor for job in self.possible_jobs))
        self.max_layer_latency = max((max((max(layer) for layer in job.layer_latencies)) for job in self.possible_jobs))
        self.job_id_mean = np.mean(np.arange(len(self.possible_jobs)))
        self.job_id_std = np.std(np.arange(len(self.possible_jobs)))
        self.layer_id_mean = np.mean(np.arange(self.max_layer_id))
        self.layer_id_std = np.std(np.arange(self.max_layer_id))
        self.deadline_mean = np.mean([job.min_latency * self.qos_factor for job in self.possible_jobs])
        self.deadline_std = np.std([job.min_latency * self.qos_factor for job in self.possible_jobs])
        self.latency_mean = np.mean(np.concatenate([[layer for layer in job.layer_latencies] for job in self.possible_jobs]), axis=0)
        self.latency_std = np.std(np.concatenate([[layer for layer in job.layer_latencies] for job in self.possible_jobs]), axis=0)
        self.hit_reward = 0.1
        self.miss_reward = -0.11
        self.use_sa_assignement_heuristic = use_sa_assignement_heuristic
        self.next_state_mode = config.next_state_mode
        self.prepend_sa_remaining_time = config.prepend_sa_remaining_time
        self.normalize_reward = config.normalize_reward
        self.full_sim_reward_factor = config.full_sim_reward_factor
        self.reward_slack_factor = config.reward_slack_factor
        self.scheduling_period = config.scheduling_period
        self.min_deadline_period_factor = config.min_deadline_period_factor
        self.sort_attr = 'absolute_deadline'
        self.dynamic_qos = getattr(config, 'dynamic_qos', None)
        self.uniform_arrivals = getattr(config, 'uniform_arrivals', False)
        self.num_tenants = getattr(config, 'num_tenants', 0)
        self.arrived_jobs_per_tenant = [0] * self.num_tenants
        self.deadline_hits_per_tenant = [0] * self.num_tenants
        self.enable_balancing_reward = getattr(config, 'enable_balancing_reward', False)
        self.balancing_factor_degree = getattr(config, 'balancing_factor_degree', 1)
        self.tenant_jobs = []
        if self.num_tenants > 0:
            tot = sum((w[0] for w in config.tenant_workloads))
            for (i, w) in enumerate(config.tenant_workloads):
                num = np.floor(w[0] / tot * self.num_tenants).astype(int) if i != len(config.tenant_workloads) - 1 else self.num_tenants - len(self.tenant_jobs)
                self.tenant_jobs.extend(num * [w[1]])
        self.tenant_targets = []
        if (tenant_targets := getattr(config, 'tenant_targets', None)):
            tot = sum((t[0] for t in tenant_targets))
            for (i, t) in enumerate(tenant_targets):
                num = np.floor(t[0] / tot * self.num_tenants).astype(int) if i != len(config.tenant_targets) - 1 else self.num_tenants - len(self.tenant_targets)
                self.tenant_targets.extend(num * [t[1]])
        self.tenant_qos = []
        if (tenant_qos := getattr(config, 'tenant_qos', None)):
            tot = sum((q[0] for q in tenant_qos))
            for (i, q) in enumerate(tenant_qos):
                num = np.floor(q[0] / tot * self.num_tenants).astype(int) if i != len(config.tenant_qos) - 1 else self.num_tenants - len(self.tenant_qos)
                self.tenant_qos.extend(num * [q[1]])
            self.rng.shuffle(self.tenant_qos)
        self.hrt = getattr(config, 'hrt', False)
        self.hrt_flag_provided_to_the_network = getattr(config, 'hrt_flag_provided_to_the_network', True)
        self.hrt_customers = getattr(config, 'hrt_customers', [])
        self.be_customers = getattr(config, 'be_customers', self.possible_jobs_ids)
        self.hrt_miss_penalty = getattr(config, 'hrt_miss_penalty', -10.0)
        self.hrt_wcet_win = 0
        self.hrt_rl_win = 0
        self.hrt_step_mode = getattr(config, 'hrt_step_mode', 'all')
        self.hrt_geom_p = getattr(config, 'hrt_geom_p', 0.6)
        self.hrt_geom_kmax = getattr(config, 'hrt_geom_kmax', 5)
        self.hrt_next_arrival = {}
        self._hrt_arrival_horizon = None
        for hrt_cust in self.hrt_customers:
            (hrt_customer_id, job_id, period, _) = hrt_cust
            self.hrt_next_arrival[hrt_customer_id] = 0.0
        if self.hrt and self.next_state_mode != 'hrt_only_all_periods':
            schedule_path = getattr(config, 'wcet_schedule_path', 'feasible_scheduling.yaml')
            self.load_wcet_schedule(schedule_path)

    def load_wcet_schedule(self, schedule_path='feasible_scheduling.yaml'):
        """
        Load WCET schedule from YAML file and process it for efficient lookup.
        The schedule is organized by the relative time within the LCM period.
        """
        import yaml
        try:
            with open(schedule_path, 'r') as f:
                data = yaml.safe_load(f)
                self.wcet_schedule = data.get('feasible_schedule', [])
                self._hrt_reference = None
            if not self.wcet_schedule:
                print(f'[WARN] Empty WCET schedule loaded from {schedule_path}')
                return False
            print(f'[INFO] Loaded WCET schedule with {len(self.wcet_schedule)} subjobs from {schedule_path}')
            self.wcet_schedule_map = {}
            for item in self.wcet_schedule:
                key = (item['job_id'], item['layer_id'])
                if key not in self.wcet_schedule_map:
                    self.wcet_schedule_map[key] = []
                self.wcet_schedule_map[key].append(item)
            return True
        except FileNotFoundError:
            print(f'[WARN] WCET schedule file {schedule_path} not found')
            return False
        except Exception as e:
            print(f'[ERROR] Failed to load WCET schedule: {e}')
            return False

    def _ensure_hrt_reference(self):
        if self._hrt_reference is None:
            if not self.wcet_schedule:
                raise ReferenceScheduleError('deployment requires a nonempty HRT reference')
            self._hrt_reference = PeriodicReference(self.wcet_schedule, self.hrt_customers, self.jobs, self.num_sas)
        return self._hrt_reference

    def _prune_committed_operations(self, now):
        active = {(sj.is_hrt, sj.job_instance_id) for sj in self.rq}
        retained = set(active)
        for op in self._committed_operations.values():
            if op.finish > now:
                retained.add((op.hrt, op.instance))
        self._committed_operations = {key: op for (key, op) in self._committed_operations.items() if (op.hrt, op.instance) in retained}
        if self._hrt_reference is not None:
            self._hrt_reference.bindings = {instance: ops for (instance, ops) in self._hrt_reference.bindings.items() if (True, instance) in retained}

    def _validate_hrt_dispatch_period(self):
        if self.scheduling_period <= 0:
            raise ReferenceScheduleError('safe-prefix deployment requires a positive dispatch period')
        if any((customer[2] % self.scheduling_period != 0 for customer in self.hrt_customers)):
            raise ReferenceScheduleError('each HRT task period must be a multiple of scheduling_period')

    def _prepare_hrt_reference(self, period_start):
        if not self.hrt_customers:
            self._prune_committed_operations(period_start)
            return []
        self._validate_hrt_dispatch_period()
        reference = self._ensure_hrt_reference()
        self._prune_committed_operations(period_start)
        for sj in self.rq:
            if sj.is_hrt:
                reference.bind(sj, period_start)
        return [op for ops in reference.bindings.values() for op in ops]

    def get_wcet_schedule_for_subjob(self, job_id, layer_id, relative_time_in_lcm, job_instance_id=None):
        """Lookup a bound runtime instance; never choose an arbitrary first row."""
        reference = self._ensure_hrt_reference()
        matches = [op for (instance, ops) in reference.bindings.items() for op in ops if op.job == job_id and op.layer == layer_id and (job_instance_id is None or instance == job_instance_id)]
        if len(matches) != 1:
            raise ReferenceScheduleError('WCET lookup requires a uniquely bound instance')
        op = matches[0]
        return dict(sa_id=op.sa, start_time=op.start, finish_time=op.finish, absolute_deadline=op.deadline, job_instance_id=op.instance)

    def get_wcet_scheduled_subjobs_in_period(self, period_start):
        """Bound instances whose reference starts in the half-open tick."""
        return [dict(job_id=op.job, layer_id=op.layer, accelerator=op.sa, start_time=op.start, finish_time=op.finish, absolute_deadline=op.deadline, job_instance_id=op.instance) for op in self._prepare_hrt_reference(period_start) if period_start <= op.start < period_start + self.scheduling_period]

    def _committed_predecessor_finish(self, sj):
        predecessor = self._committed_operations.get((sj.is_hrt, sj.job_instance_id, sj.layer_id - 1))
        return predecessor.finish if predecessor is not None else sj.arrival_time

    def _record_committed_subjobs(self, cut):
        for sj in self.rq:
            if sj.start_time is not None and sj.start_time < cut:
                op = Operation.from_subjob(sj)
                previous = self._committed_operations.get(op.key)
                if previous is not None and previous != op:
                    raise ReferenceScheduleError('attempt to rewrite committed execution')
                self._committed_operations[op.key] = op
                if sj.is_hrt and sj.is_last:
                    self._pending_hrt_completions[sj.job_instance_id] = deepcopy(sj)
                elif sj.is_last:
                    self._pending_be_completions[sj.job_instance_id] = deepcopy(sj)

    def _settle_hrt_completions(self):
        """Settle dispatched last layers, including BE tails, exactly once."""
        reward = 0
        pending = list(self._pending_hrt_completions.values()) + list(self._pending_be_completions.values())
        for sj in pending:
            if sj.finish_time <= self.current_time or self.current_time > sj.absolute_deadline:
                reward += self.calculate_lastsj_reward(sj)
        return reward

    def check_hrt_feasibility(self):
        """
        Verifica se la schedulazione corrente rispetta tutti i deadline HRT.
        Ritorna True se tutti i subjob HRT hanno finish_time <= absolute_deadline, False altrimenti.
        """
        for sj in self.rq:
            if sj.is_hrt and sj.finish_time is not None:
                if sj.finish_time > sj.absolute_deadline:
                    return False
        return True

    def apply_wcet_schedule(self, schedule):
        """
        Applica una schedulazione pre-calcolata (WCET) alla ready queue.
        Questa funzione viene usata quando Retnas non trova una schedulazione
        feasible entro MAX_TIME.

        Args:
            schedule: Lista di dict con schedulazione [{job_instance_id, job_id, layer_id, sa_id, start_time, finish_time}, ...]
        """
        for sj in self.rq:
            if sj.is_hrt:
                for item in schedule:
                    if item['job_id'] == sj.job_id and item['layer_id'] == sj.layer_id:
                        sj.sa_id = item['sa_id']
                        sj.start_time = self.current_time + item['start_time']
                        sj.finish_time = self.current_time + item['finish_time']
                        break

    def schedule_hrt_wcet(self, period_start=None):
        """Restore only the uncommitted reference suffix, without time travel."""
        if period_start is None:
            period_start = self.current_time
        reference = self._prepare_hrt_reference(period_start)
        remaining = {op.key: op for op in reference if op.key not in self._committed_operations}
        if any((op.start < period_start for op in remaining.values())):
            raise ReferenceScheduleError('reference suffix would start in the past')
        hybrid = list(self._committed_operations.values()) + list(remaining.values())
        reason = validate_operations(hybrid, self.jobs, self.num_sas, complete=True)
        if reason:
            raise ReferenceScheduleError('reference suffix cannot be resumed: ' + reason)
        scheduled = []
        for sj in self.rq:
            if sj.is_hrt:
                op = remaining.get((True, sj.job_instance_id, sj.layer_id))
                if op is None:
                    raise ReferenceScheduleError('missing uncommitted reference subjob')
                (sj.sa_id, sj.start_time, sj.finish_time) = (op.sa, op.start, op.finish)
                scheduled.append(sj)
        return scheduled

    def _select_hrt_schedule(self, period_start):
        (consistent, violation) = self.check_hrt_consistency_with_wcet(period_start)
        if consistent:
            self.hrt_rl_win += 1
            return True
        self.hrt_wcet_win += 1
        reason = self.hrt_consistency_reason
        self.hrt_guardian_rejections[reason] = self.hrt_guardian_rejections.get(reason, 0) + 1
        print('[HRT DEPLOYMENT] FALLBACK to reference: ' + self.hrt_consistency_reason)
        self.schedule_hrt_wcet(period_start)
        for sj in self.rq:
            if not sj.is_hrt:
                sj.start_time = sj.finish_time = sj.sa_id = None
                if sj.is_last and sj.job_instance_id not in self._settled_be_instances:
                    self.deadline_misses_be += 1
                    self._settled_be_instances.add(sj.job_instance_id)
        return False

    def step(self, action):
        """Dispatch with certification in deployment; use unconstrained actions in training."""
        if self.hrt_step_mode == 'all':
            return self.step_with_hrt_all(action)
        if self.hrt_step_mode == 'deployment':
            return self.step_with_hrt_deployment(action)
        raise ValueError('hrt_step_mode must be all or deployment')

    def step_with_hrt_all(self, action):
        hrt_feasible = True
        if self.prepend_sa_remaining_time:
            action = action[1:]
        self._prune_committed_operations(self.current_time)
        if self.scheduling_period > 0:
            self.current_time += self.scheduling_period
        else:
            timeskip = self.distribution(self.alpha) * self.scale
            self.current_time += timeskip
        reward = 0
        num_subjobs = action.shape[0]
        if len(self.rq) > 0:
            sa_busy_until_temp = self.sa_busy_until.copy()
            action_with_index = np.column_stack((action, np.arange(action.shape[0])))
            mapped = np.zeros(action.shape[0], dtype=bool)
            keytimes = []
            di = 0
            while self.imminents:
                self.imminents_lengths.append(len(self.imminents))
                imminents_actions = action_with_index[list(self.imminents)]
                decision = imminents_actions[imminents_actions[:, 0].argmin()]
                rq_sj_id = decision[-1].astype(int)
                assigned_sa = decision[1:1 + self.num_sas].argmax()
                sj = self.rq[rq_sj_id]
                job = self.jobs[sj.job_id]
                if self.use_sa_assignement_heuristic:
                    assigned_sa = np.argmin([sa_busy_until_temp[a] + job.layer_latencies[sj.layer_id][a] for a in range(self.num_sas)])
                latency = job.layer_latencies[sj.layer_id][assigned_sa]
                if len(sj.dependencies) > 0:
                    self.rq[rq_sj_id].start_time = max(sa_busy_until_temp[assigned_sa], *[self.rq[d].finish_time for d in sj.dependencies])
                else:
                    self.rq[rq_sj_id].start_time = sa_busy_until_temp[assigned_sa]
                self.rq[rq_sj_id].start_time = max(self.rq[rq_sj_id].start_time, sj.arrival_time, self._committed_predecessor_finish(sj))
                self.rq[rq_sj_id].finish_time = self.rq[rq_sj_id].start_time + latency
                self.rq[rq_sj_id].sa_id = assigned_sa
                sa_busy_until_temp[assigned_sa] = sj.finish_time
                keytimes.extend([self.rq[rq_sj_id].start_time, self.rq[rq_sj_id].finish_time])
                mapped[rq_sj_id] = True
                self.imminents.remove(rq_sj_id)
                self.imminents.update([c for c in sj.children if mapped[self.rq[c].dependencies].all()])
                di += 1
            assert di == len(self.rq), f'{di}/{len(self.rq)}'
            self.old_rq = deepcopy(self.rq)
            self._record_committed_subjobs(self.current_time)
            self.rq.clear()
            new_keytimes = []
            dependencies_conversion_table = {}
            for (rq_sj_id, sj) in enumerate(self.old_rq):
                if sj.start_time < self.current_time:
                    self.sa_busy_until[sj.sa_id] = max(self.sa_busy_until[sj.sa_id], sj.finish_time)
                    if sj.finish_time >= self.sa_busy_until[sj.sa_id]:
                        self.sa_last_sj_executed[sj.sa_id] = (sj.job_instance_id, sj.job_id, sj.layer_id)
                    self.executed_sjs += 1
                elif self.current_time < sj.absolute_deadline:
                    self.rq.append(sj)
                    dependencies_conversion_table[rq_sj_id] = len(self.rq) - 1
                somma = 0
                factor = 1
                if self.reward_slack_factor > 0:
                    if sj.finish_time <= self.current_time:
                        somma = self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                    elif self.full_sim_reward_factor > 0:
                        somma = self.full_sim_reward_factor * self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                if sj.is_hrt:
                    factor = 5
                reward += somma * factor
                if sj.is_last:
                    tmp = self.calculate_lastsj_reward(sj)
                    reward += tmp
                    if sj.is_hrt and tmp <= self.hrt_miss_penalty:
                        hrt_feasible = False
                    if sj.is_hrt and sj.finish_time > sj.absolute_deadline:
                        hrt_feasible = False
            self.imminents.clear()
            for rq_sj_id in range(len(self.rq)):
                self.rq[rq_sj_id].dependencies = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].dependencies if d in dependencies_conversion_table]
                if len(self.rq[rq_sj_id].dependencies) == 0:
                    self.imminents.add(rq_sj_id)
                self.rq[rq_sj_id].children = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].children if d in dependencies_conversion_table]
        reward += self._settle_hrt_completions()
        if self.deadline_misses_hrt:
            hrt_feasible = False
        for a in range(self.num_sas):
            self.sa_busy_until[a] = max(self.sa_busy_until[a], self.current_time)
        if self.next_state_mode == 'noarrivals':
            rb_next_state = actor_next_state = self.encode_state()
        elif self.next_state_mode == 'standard':
            self.new_jobs_arrival()
            rb_next_state = self.encode_state()
            actor_next_state = rb_next_state
        elif self.next_state_mode == 'nonewjobs':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival()
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True)
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only_all_periods':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True, hrt_all_periods=True)
            actor_next_state = self.encode_state()
        else:
            print('Next state mode not implemented:', self.next_state_mode)
            quit(1)
        if self.normalize_reward:
            reward = reward * (self.scale / (self.alpha - 1)) / timeskip
        done = len(self.rq) == 0
        if self.imminents:
            assert max(self.imminents) < len(self.rq), f'Stale imminents index: max={max(self.imminents)}, len(rq)={len(self.rq)}'
        self.sort_ready_queue()
        return (actor_next_state, rb_next_state, reward, done, hrt_feasible)

    def check_hrt_consistency_with_wcet(self, current_period_start):
        """Certify committed actor work followed by the reference continuation."""
        reference = self._prepare_hrt_reference(current_period_start)
        if not self.hrt_customers:
            self.hrt_consistency_reason = 'no HRT customers'
            return (True, 0)
        candidate = dict(self._committed_operations)
        for sj in self.rq:
            op = Operation.from_subjob(sj)
            if op.key in candidate:
                self.hrt_consistency_reason = 'candidate rewrites a committed subjob'
                return (False, float('inf'))
            if op.start is not None and op.start < current_period_start:
                self.hrt_consistency_reason = 'candidate starts new work in the past'
                return (False, float('inf'))
            candidate[op.key] = op
        cut = current_period_start + self.scheduling_period
        tail_end = max((op.finish for op in candidate.values() if op.start is not None and op.start < cut and (op.finish is not None)), default=cut)
        future = self._hrt_reference.future_operations(cut, tail_end, self.hrt_next_arrival)
        (consistent, reason) = check_safe_continuation(reference, list(candidate.values()), cut, self.jobs, self.num_sas, future_reference=future)
        self.hrt_consistency_reason = reason
        return (consistent, 0 if consistent else float('inf'))

    def step_with_hrt_deployment(self, action):
        hrt_feasible = True
        if self.prepend_sa_remaining_time:
            action = action[1:]
        period_start_time = self.current_time
        self._prepare_hrt_reference(period_start_time)
        if self.scheduling_period > 0:
            self.current_time += self.scheduling_period
        else:
            timeskip = self.distribution(self.alpha) * self.scale
            self.current_time += timeskip
        reward = 0
        num_subjobs = action.shape[0]
        if len(self.rq) > 0:
            sa_busy_until_temp = self.sa_busy_until.copy()
            action_with_index = np.column_stack((action, np.arange(action.shape[0])))
            mapped = np.zeros(action.shape[0], dtype=bool)
            keytimes = []
            di = 0
            while self.imminents:
                self.imminents_lengths.append(len(self.imminents))
                imminents_actions = action_with_index[list(self.imminents)]
                decision = imminents_actions[imminents_actions[:, 0].argmin()]
                rq_sj_id = decision[-1].astype(int)
                assigned_sa = decision[1:1 + self.num_sas].argmax()
                sj = self.rq[rq_sj_id]
                job = self.jobs[sj.job_id]
                if self.use_sa_assignement_heuristic:
                    assigned_sa = np.argmin([sa_busy_until_temp[a] + job.layer_latencies[sj.layer_id][a] for a in range(self.num_sas)])
                latency = job.layer_latencies[sj.layer_id][assigned_sa]
                if len(sj.dependencies) > 0:
                    self.rq[rq_sj_id].start_time = max(sa_busy_until_temp[assigned_sa], *[self.rq[d].finish_time for d in sj.dependencies])
                else:
                    self.rq[rq_sj_id].start_time = sa_busy_until_temp[assigned_sa]
                self.rq[rq_sj_id].start_time = max(self.rq[rq_sj_id].start_time, sj.arrival_time, self._committed_predecessor_finish(sj))
                self.rq[rq_sj_id].finish_time = self.rq[rq_sj_id].start_time + latency
                self.rq[rq_sj_id].sa_id = assigned_sa
                sa_busy_until_temp[assigned_sa] = sj.finish_time
                keytimes.extend([self.rq[rq_sj_id].start_time, self.rq[rq_sj_id].finish_time])
                mapped[rq_sj_id] = True
                self.imminents.remove(rq_sj_id)
                self.imminents.update([c for c in sj.children if mapped[self.rq[c].dependencies].all()])
                di += 1
            assert di == len(self.rq), f'{di}/{len(self.rq)}'
            is_consistent = self._select_hrt_schedule(period_start_time)
            if not is_consistent and any((not sj.is_hrt for sj in self.rq)):
                reward += self.hrt_miss_penalty
            self._record_committed_subjobs(self.current_time)
            self.old_rq = deepcopy(self.rq)
            self.rq.clear()
            dependencies_conversion_table = {}
            for (rq_sj_id, sj) in enumerate(self.old_rq):
                if sj.start_time is None:
                    continue
                if sj.start_time < self.current_time:
                    self.sa_busy_until[sj.sa_id] = max(self.sa_busy_until[sj.sa_id], sj.finish_time)
                    if sj.finish_time >= self.sa_busy_until[sj.sa_id]:
                        self.sa_last_sj_executed[sj.sa_id] = (sj.job_instance_id, sj.job_id, sj.layer_id)
                    self.executed_sjs += 1
                elif self.current_time < sj.absolute_deadline:
                    self.rq.append(sj)
                    dependencies_conversion_table[rq_sj_id] = len(self.rq) - 1
                somma = 0
                factor = 1
                if self.reward_slack_factor > 0:
                    if sj.finish_time <= self.current_time:
                        somma = self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                    elif self.full_sim_reward_factor > 0:
                        somma = self.full_sim_reward_factor * self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                if sj.is_hrt:
                    factor = 5
                reward += somma * factor
                if sj.is_last:
                    tmp = self.calculate_lastsj_reward(sj)
                    reward += tmp
                    if sj.is_hrt and tmp <= self.hrt_miss_penalty:
                        hrt_feasible = False
                    if sj.is_hrt and sj.finish_time > sj.absolute_deadline:
                        hrt_feasible = False
            self.imminents.clear()
            for rq_sj_id in range(len(self.rq)):
                self.rq[rq_sj_id].dependencies = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].dependencies if d in dependencies_conversion_table]
                if len(self.rq[rq_sj_id].dependencies) == 0:
                    self.imminents.add(rq_sj_id)
                self.rq[rq_sj_id].children = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].children if d in dependencies_conversion_table]
        reward += self._settle_hrt_completions()
        if self.deadline_misses_hrt:
            hrt_feasible = False
        for a in range(self.num_sas):
            self.sa_busy_until[a] = max(self.sa_busy_until[a], self.current_time)
        if self.next_state_mode == 'noarrivals':
            rb_next_state = actor_next_state = self.encode_state()
        elif self.next_state_mode == 'standard':
            self.new_jobs_arrival()
            rb_next_state = self.encode_state()
            actor_next_state = rb_next_state
        elif self.next_state_mode == 'nonewjobs':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival()
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True)
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only_all_periods':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True, hrt_all_periods=True)
            actor_next_state = self.encode_state()
        else:
            print('Next state mode not implemented:', self.next_state_mode)
            quit(1)
        if self.normalize_reward:
            reward = reward * (self.scale / (self.alpha - 1)) / timeskip
        done = len(self.rq) == 0
        if self.imminents:
            assert max(self.imminents) < len(self.rq), f'Stale imminents index: max={max(self.imminents)}, len(rq)={len(self.rq)}'
        self.sort_ready_queue()
        return (actor_next_state, rb_next_state, reward, done, hrt_feasible)

    def calculate_lastsj_reward(self, sj):
        reward = 0
        if sj.is_hrt:
            if sj.job_instance_id in self._settled_hrt_instances:
                return 0
            if sj.finish_time <= self.current_time or self.current_time > sj.absolute_deadline:
                self._settled_hrt_instances.add(sj.job_instance_id)
                self._pending_hrt_completions.pop(sj.job_instance_id, None)
            if sj.finish_time <= self.current_time:
                if sj.finish_time <= sj.absolute_deadline:
                    self.deadline_hits_hrt += 1
                    if sj.owner is not None:
                        self.deadline_hits_per_tenant[sj.owner] += 1
                else:
                    self.deadline_misses_hrt += 1
                    reward += self.hrt_miss_penalty
            elif self.current_time > sj.absolute_deadline:
                reward += self.hrt_miss_penalty * 2
                self.deadline_misses_hrt += 1
            elif self.full_sim_reward_factor > 0:
                if sj.finish_time <= sj.absolute_deadline:
                    pass
                else:
                    pass
            return reward
        if not sj.is_hrt:
            if sj.job_instance_id in self._settled_be_instances:
                return 0
            if sj.finish_time <= self.current_time or self.current_time > sj.absolute_deadline:
                self._settled_be_instances.add(sj.job_instance_id)
                self._pending_be_completions.pop(sj.job_instance_id, None)
            if sj.finish_time <= self.current_time:
                if sj.finish_time <= sj.absolute_deadline:
                    self.deadline_hits_be += 1
                    if sj.owner is not None:
                        self.deadline_hits_per_tenant[sj.owner] += 1
                else:
                    self.deadline_misses_be += 1
                    reward += self.miss_reward
                    reward += 0.1 * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
            elif self.current_time > sj.absolute_deadline:
                reward += self.miss_reward * 2
                self.deadline_misses_be += 1
            elif self.full_sim_reward_factor > 0:
                if sj.finish_time <= sj.absolute_deadline:
                    pass
                else:
                    reward += self.full_sim_reward_factor * self.miss_reward
                    reward += self.full_sim_reward_factor * 0.1 * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
            if sj.owner is not None and self.enable_balancing_reward:
                mean_slasr = sum(self.deadline_hits_per_tenant) / (sum(self.arrived_jobs_per_tenant) + 1e-08)
                target_slasr = mean_slasr
                target_slasr = sj.target_slasr
                actual_slasr = self.deadline_hits_per_tenant[sj.owner] / (self.arrived_jobs_per_tenant[sj.owner] + 1e-08)
                assert actual_slasr <= 1 and actual_slasr >= 0
                reward += -abs(target_slasr - actual_slasr) * 0.1
        return reward

    def reset(self, clear=True, hrt_only=False, no_new_jobs=False):
        if clear:
            self._committed_operations.clear()
            self._pending_hrt_completions.clear()
            self._settled_hrt_instances.clear()
            self._pending_be_completions.clear()
            self._settled_be_instances.clear()
            self.hrt_guardian_rejections.clear()
            self.hrt_wcet_win = 0
            self.hrt_rl_win = 0
            if self._hrt_reference is not None:
                self._hrt_reference.reset()
            self.sa_busy_until = [0] * self.num_sas
            self.sa_last_sj_executed = [0] * self.num_sas
            self.current_time = 0
            for hrt_cust in self.hrt_customers:
                hrt_customer_id = hrt_cust[0]
                self.hrt_next_arrival[hrt_customer_id] = 0
            self.last_arrival_time = 0
            self.deadline_hits_be = 0
            self.deadline_hits_hrt = 0
            self.deadline_misses_be = 0
            self.deadline_misses_hrt = 0
            self.total_arrived_jobs_be = 0
            self.total_arrived_jobs_hrt = 0
            self.imminents_lengths = []
            self.executed_sjs = 0
            self.arrived_jobs_per_tenant = [0] * self.num_tenants
            self.deadline_hits_per_tenant = [0] * self.num_tenants
            self.rq = list()
            self.imminents = set()
            hrt_all_periods = False
            if self.next_state_mode == 'hrt_only' or self.next_state_mode == 'hrt_only_all_periods':
                hrt_only = True
            if self.next_state_mode == 'hrt_only_all_periods':
                hrt_all_periods = True
            if not no_new_jobs:
                self.new_jobs_arrival(hrt_only, hrt_all_periods)
        else:
            if self._hrt_reference is not None:
                raise ReferenceScheduleError('deployment reset must clear state; periodic reservations cannot be rebased')
            offset = self.current_time
            self._committed_operations = {key: replace(op, start=op.start - offset, finish=op.finish - offset, release=op.release - offset, deadline=op.deadline - offset) for (key, op) in self._committed_operations.items()}
            pending = list(self._pending_hrt_completions.values()) + list(self._pending_be_completions.values())
            for sj in pending:
                sj.start_time -= offset
                sj.finish_time -= offset
                sj.arrival_time -= offset
                sj.absolute_deadline -= offset
            self.deadline_hits_be = 0
            self.deadline_hits_hrt = 0
            self.deadline_misses_be = 0
            self.deadline_misses_hrt = 0
            self.total_arrived_jobs_be = len(set((sj.job_id for sj in self.rq if not sj.is_hrt)))
            self.total_arrived_jobs_hrt = len(set((sj.job_id for sj in self.rq if sj.is_hrt)))
            self.executed_sjs = 0
            for j in range(len(self.rq)):
                self.rq[j].absolute_deadline -= self.current_time
                self.rq[j].arrival_time -= self.current_time
            self.sa_busy_until = [b - self.current_time for b in self.sa_busy_until]
            self.last_arrival_time -= self.current_time
            self.current_time = 0
            for hrt_cust in self.hrt_customers:
                hrt_customer_id = hrt_cust[0]
                self.hrt_next_arrival[hrt_customer_id] = 0
            if self.total_arrived_jobs_be + self.total_arrived_jobs_hrt == 0:
                self.new_jobs_arrival()
        return self.encode_state()

    def new_jobs_arrival(self, hrt_only=False, hrt_all_periods=False):
        if hrt_only or self.hrt:
            for hrt_cust in self.hrt_customers:
                (hrt_customer_id, job_id, period, hrt_deadline) = hrt_cust
                if self._hrt_arrival_horizon is not None and self.hrt_next_arrival[hrt_customer_id] >= self._hrt_arrival_horizon:
                    continue
                if self.hrt_next_arrival[hrt_customer_id] <= self.current_time:
                    self.new_job_arrival(self.hrt_next_arrival[hrt_customer_id], tenant_id=None, is_hrt=True, hrt_job_id=job_id, hrt_deadline=hrt_deadline, hrt_customer_id=hrt_customer_id)
                    k = self.rng.geometric(self.hrt_geom_p)
                    if self.hrt_geom_kmax is not None:
                        k = min(k, self.hrt_geom_kmax)
                    nondeterministic_jitter = k * 0.001 * period * self.rng.random()
                    if hrt_all_periods:
                        k = 1
                    assert nondeterministic_jitter < period
                    self.hrt_next_arrival[hrt_customer_id] = self.current_time + k * period
        if not hrt_only:
            if self.scheduling_period > 0:
                if self.uniform_arrivals:
                    for _ in range(self.jobs_per_arrival):
                        tenant_id = self.rng.integers(0, self.num_tenants) if self.num_tenants > 0 else None
                        self.new_job_arrival(self.last_arrival_time, tenant_id, is_hrt=False)
                    self.last_arrival_time += self.scheduling_period
                else:
                    while self.last_arrival_time > self.current_time - self.scheduling_period and self.last_arrival_time <= self.current_time:
                        for _ in range(self.jobs_per_arrival):
                            tenant_id = self.rng.integers(0, self.num_tenants) if self.num_tenants > 0 else None
                            self.new_job_arrival(self.last_arrival_time, tenant_id, is_hrt=False)
                        timeskip = self.distribution(self.alpha) * self.scale
                        self.last_arrival_time += timeskip
            else:
                for _ in range(self.jobs_per_arrival):
                    tenant_id = self.rng.integers(0, self.num_tenants) if self.num_tenants > 0 else None
                    self.new_job_arrival(self.current_time, tenant_id, is_hrt=False)

    def new_job_arrival(self, arrival_time, tenant_id=None, is_hrt=False, hrt_job_id=None, hrt_deadline=None, hrt_customer_id=None):
        if is_hrt:
            job_id = hrt_job_id
            slasrt = 1
            tenant_qosf = self.qos_factor
            relative_deadline = hrt_deadline
        elif tenant_id is None:
            print('NEW job without tenant_id at time', arrival_time, 'generated at', self.current_time, arrival_time <= self.current_time)
            quit()
            job_id = self.rng.choice(self.be_customers)
            slasrt = 1
            tenant_qosf = self.qos_factor
        else:
            job_id = self.rng.choice(self.tenant_jobs[tenant_id])
            slasrt = self.tenant_targets[tenant_id] if len(self.tenant_targets) > 0 else 1
            tenant_qosf = self.tenant_qos[tenant_id] if len(self.tenant_qos) > 0 else self.qos_factor
            self.arrived_jobs_per_tenant[tenant_id] += 1
        fix_factor = 1
        self.imminents.add(len(self.rq))
        if not is_hrt:
            if self.dynamic_qos is not None:
                relative_deadline = self.rng.choice(self.dynamic_qos) * self.jobs[job_id].min_latency
            else:
                relative_deadline = max(self.jobs[job_id].min_latency * tenant_qosf, self.min_deadline_period_factor * self.scheduling_period)
                relative_deadline *= fix_factor
        absolute_deadline = arrival_time + relative_deadline
        for l in range(self.jobs[job_id].num_layers):
            self.rq.append(SubJob(self.next_job_instance_id, job_id, l, relative_deadline, absolute_deadline, arrival_time, dependencies=[len(self.rq) - 1], children=[len(self.rq) + 1], owner=tenant_id, target_slasr=slasrt, is_hrt=is_hrt, hrt_customer_id=hrt_customer_id))
        self.next_job_instance_id += 1
        self.rq[-self.jobs[job_id].num_layers].dependencies = []
        self.rq[-1].children = []
        self.rq[-1].is_last = True
        if is_hrt:
            self.total_arrived_jobs_hrt += 1
        else:
            self.total_arrived_jobs_be += 1

    def sort_ready_queue(self):
        sorted_indices = sorted(range(len(self.rq)), key=lambda i: (getattr(self.rq[i], self.sort_attr), self.rq[i].layer_id))
        self.rq.sort(key=lambda d: (getattr(d, self.sort_attr), d.layer_id))
        for i in range(len(self.rq)):
            self.rq[i].dependencies = [sorted_indices.index(d) for d in self.rq[i].dependencies]
            self.rq[i].children = [sorted_indices.index(d) for d in self.rq[i].children]
        self.imminents = set((sorted_indices.index(d) for d in self.imminents))

    def encode_state(self):
        prefix = []
        if self.prepend_sa_remaining_time:
            prefix = [(*[0] * 4, *[(self.sa_busy_until[a] - self.current_time - self.latency_mean[a]) / self.latency_std[a] for a in range(self.num_sas)], *(), *(), *((0, 0) if self.num_tenants > 0 and self.enable_balancing_reward else ()), 0 if self.hrt and self.hrt_flag_provided_to_the_network else ())]
        elif len(self.rq) == 0:
            return np.zeros((1, 4 + self.num_sas + 0 + 0 + (2 if self.num_tenants > 0 and self.enable_balancing_reward else 0) + (1 if self.hrt and self.hrt_flag_provided_to_the_network else 0)), dtype=np.float32)
        return np.array(prefix + [((sj.job_id - self.job_id_mean) / self.job_id_std, (sj.layer_id - self.layer_id_mean) / self.layer_id_std, (sj.relative_deadline - self.deadline_mean) / self.deadline_std, (self.current_time - sj.arrival_time - self.deadline_mean) / self.deadline_std, *[(latency - self.latency_mean[a]) / self.latency_std[a] for (a, latency) in enumerate(self.jobs[sj.job_id].layer_latencies[sj.layer_id])], *((self.deadline_hits_per_tenant[sj.owner] / (self.arrived_jobs_per_tenant[sj.owner] + 1e-08), sj.target_slasr) if self.num_tenants > 0 and self.enable_balancing_reward else ()), *((1.0 if sj.is_hrt else 0.0,) if self.hrt and self.hrt_flag_provided_to_the_network else ())) for sj in sorted(self.rq, key=lambda d: (getattr(d, self.sort_attr), d.layer_id))], dtype=np.float32)

    def prema_step(self, action):
        if self.prepend_sa_remaining_time:
            action = action[1:]
        period_start_time = self.current_time
        if self.hrt and self.hrt_step_mode == 'deployment':
            self._prepare_hrt_reference(period_start_time)
        if self.scheduling_period > 0:
            self.current_time += self.scheduling_period
        else:
            timeskip = self.distribution(self.alpha) * self.scale
            self.current_time += timeskip
        reward = 0
        num_subjobs = action.shape[0]
        hrt_feasible = True
        is_consistent = True
        if len(self.rq) > 0:
            sa_busy_until_temp = self.sa_busy_until.copy()
            assert num_subjobs == len(self.rq), f'mismatch between actor output size and env ready queue length: {num_subjobs}/{len(self.rq)}'
            action = np.column_stack((action, np.arange(num_subjobs)))
            mapped = np.zeros(action.shape[0], dtype=bool)
            keytimes = []
            di = 0
            while self.imminents:
                self.imminents_lengths.append(len(self.imminents))
                imminents_actions = action[list(self.imminents)]
                threshold = 3
                candidates = imminents_actions[:, 0].argsort()[:threshold]
                decision = imminents_actions[imminents_actions[candidates, 1].argmin()]
                rq_sj_id = decision[-1].astype(int)
                assigned_sa = decision[1:1 + self.num_sas].argmax()
                sj = self.rq[rq_sj_id]
                job = self.jobs[sj.job_id]
                if self.use_sa_assignement_heuristic:
                    assigned_sa = np.argmin([sa_busy_until_temp[a] + job.layer_latencies[sj.layer_id][a] for a in range(self.num_sas)])
                latency = job.layer_latencies[sj.layer_id][assigned_sa]
                if len(sj.dependencies) > 0:
                    self.rq[rq_sj_id].start_time = max(sa_busy_until_temp[assigned_sa], *[self.rq[d].finish_time for d in sj.dependencies])
                else:
                    self.rq[rq_sj_id].start_time = sa_busy_until_temp[assigned_sa]
                self.rq[rq_sj_id].start_time = max(self.rq[rq_sj_id].start_time, sj.arrival_time, self._committed_predecessor_finish(sj))
                self.rq[rq_sj_id].finish_time = self.rq[rq_sj_id].start_time + latency
                self.rq[rq_sj_id].sa_id = assigned_sa
                sa_busy_until_temp[assigned_sa] = sj.finish_time
                keytimes.extend([self.rq[rq_sj_id].start_time, self.rq[rq_sj_id].finish_time])
                mapped[rq_sj_id] = True
                self.imminents.remove(rq_sj_id)
                self.imminents.update([c for c in sj.children if mapped[self.rq[c].dependencies].all()])
                di += 1
            assert di == len(self.rq), f'{di}/{len(self.rq)}'
            if self.hrt and self.hrt_step_mode == 'deployment':
                self._select_hrt_schedule(period_start_time)
                self._record_committed_subjobs(self.current_time)
            self.old_rq = deepcopy(self.rq)
            self.rq.clear()
            dependencies_conversion_table = {}
            for (rq_sj_id, sj) in enumerate(self.old_rq):
                if sj.start_time is None:
                    continue
                if sj.start_time < self.current_time:
                    self.sa_busy_until[sj.sa_id] = max(self.sa_busy_until[sj.sa_id], sj.finish_time)
                    if sj.finish_time >= self.sa_busy_until[sj.sa_id]:
                        self.sa_last_sj_executed[sj.sa_id] = (sj.job_instance_id, sj.job_id, sj.layer_id)
                    self.executed_sjs += 1
                elif self.current_time < sj.absolute_deadline:
                    self.rq.append(sj)
                    dependencies_conversion_table[rq_sj_id] = len(self.rq) - 1
                somma = 0
                factor = 1
                if self.reward_slack_factor > 0:
                    if sj.finish_time <= self.current_time:
                        somma = self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                    elif self.full_sim_reward_factor > 0:
                        somma = self.full_sim_reward_factor * self.reward_slack_factor * (sj.absolute_deadline - sj.finish_time) / sj.relative_deadline
                if sj.is_hrt:
                    factor = 5
                reward += somma * factor
                if sj.is_last:
                    tmp = self.calculate_lastsj_reward(sj)
                    reward += tmp
                    if sj.is_hrt and tmp <= self.hrt_miss_penalty:
                        hrt_feasible = False
                    if sj.is_hrt and sj.finish_time > sj.absolute_deadline:
                        hrt_feasible = False
            self.imminents.clear()
            for rq_sj_id in range(len(self.rq)):
                self.rq[rq_sj_id].dependencies = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].dependencies if d in dependencies_conversion_table]
                if len(self.rq[rq_sj_id].dependencies) == 0:
                    self.imminents.add(rq_sj_id)
                self.rq[rq_sj_id].children = [dependencies_conversion_table[d] for d in self.rq[rq_sj_id].children if d in dependencies_conversion_table]
        reward += self._settle_hrt_completions()
        if self.deadline_misses_hrt:
            hrt_feasible = False
        for a in range(self.num_sas):
            self.sa_busy_until[a] = max(self.sa_busy_until[a], self.current_time)
        if self.next_state_mode == 'noarrivals':
            rb_next_state = actor_next_state = self.encode_state()
        elif self.next_state_mode == 'standard':
            self.new_jobs_arrival()
            rb_next_state = self.encode_state()
            actor_next_state = rb_next_state
        elif self.next_state_mode == 'nonewjobs':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival()
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True)
            actor_next_state = self.encode_state()
        elif self.next_state_mode == 'hrt_only_all_periods':
            rb_next_state = self.encode_state()
            self.new_jobs_arrival(hrt_only=True, hrt_all_periods=True)
            actor_next_state = self.encode_state()
        else:
            print('Next state mode not implemented:', self.next_state_mode)
            quit(1)
        if self.normalize_reward:
            reward = reward * (self.scale / (self.alpha - 1)) / timeskip
        done = len(self.rq) == 0
        if self.imminents:
            assert max(self.imminents) < len(self.rq), f'Stale imminents index: max={max(self.imminents)}, len(rq)={len(self.rq)}'
        self.sort_ready_queue()
        return (actor_next_state, rb_next_state, reward, done, hrt_feasible)
