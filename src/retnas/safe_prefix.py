"""Safe prefix replacement for the simulator's sequential, non-preemptive jobs.

Times are absolute; the reference does not depend on the online dispatch period.
An operation belongs to the completed prefix when finish <= cut. Intervals are
half-open, so an operation starting at the cut belongs to the suffix.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
import math


class ReferenceScheduleError(RuntimeError):
    """The reference cannot safely serve the current workload/state."""


@dataclass(frozen=True)
class Operation:
    instance: int
    job: int
    layer: int
    sa: int
    start: float
    finish: float
    release: float
    deadline: float
    hrt: bool = True

    @property
    def key(self):
        return self.hrt, self.instance, self.layer

    @classmethod
    def from_subjob(cls, sj):
        return cls(int(sj.job_instance_id), int(sj.job_id), int(sj.layer_id),
                   int(sj.sa_id) if sj.sa_id is not None else None,
                   sj.start_time, sj.finish_time, sj.arrival_time,
                   sj.absolute_deadline, sj.is_hrt)


def validate_operations(operations, jobs, num_sas, complete=False):
    """Check eligibility, durations, exclusivity, and sequential precedence.

    A partial job must be a precedence-closed prefix. All included HRT
    operations must respect their deadline, including a closed job's last layer.
    """
    by_job = defaultdict(dict)
    by_sa = defaultdict(list)
    for op in operations:
        if (not isinstance(op.sa, int) or not 0 <= op.sa < num_sas
                or not isinstance(op.job, int) or not 0 <= op.job < len(jobs)
                or not isinstance(op.layer, int)
                or not 0 <= op.layer < jobs[op.job].num_layers):
            return "ineligible job/layer/accelerator"
        if not all(isinstance(t, (int, float)) and math.isfinite(t)
                   for t in (op.start, op.finish, op.release, op.deadline)):
            return "missing or non-finite operation time"
        latency = jobs[op.job].layer_latencies[op.layer][op.sa]
        if not math.isfinite(latency) or latency <= 0:
            return "ineligible accelerator latency"
        if op.start < op.release or op.finish - op.start < latency:
            return "release or execution-duration violation"
        if op.hrt and op.finish > op.deadline:
            return "HRT deadline violation"
        group = by_job[op.hrt, op.instance]
        if op.layer in group:
            return "duplicate subjob identity"
        group[op.layer] = op
        by_sa[op.sa].append(op)

    for group in by_job.values():
        layers = sorted(group)
        first = group[layers[0]]
        if layers != list(range(len(layers))):
            return "prefix is not precedence closed"
        if complete and first.hrt and len(layers) != jobs[first.job].num_layers:
            return "incomplete HRT reference job"
        for layer in layers:
            op = group[layer]
            if (op.job, op.release, op.deadline) != (
                    first.job, first.release, first.deadline):
                return "inconsistent job-instance metadata"
            if layer and group[layer - 1].finish > op.start:
                return "precedence/non-parallel execution violation"

    for intervals in by_sa.values():
        intervals.sort(key=lambda op: op.start)
        for previous, current in zip(intervals, intervals[1:]):
            if previous.finish > current.start:
                return "machine exclusivity violation"
    return None


def check_safe_prefix(reference, candidate, cut, jobs, num_sas):
    """Return (safe, reason) using the four paper conditions and clean cuts.

    Candidate contains previous commitments and the current proposal. Reference
    contains the *same actual instances*, including the history of open jobs.
    BE operations are included to prevent BE tails from blocking the HRT suffix.
    We use the successor-start comparison, not the more conservative remark.
    """
    if not math.isfinite(cut):
        return False, "non-finite cut"
    reason = validate_operations(reference, jobs, num_sas, complete=True)
    if reason:
        return False, "invalid reference: " + reason
    if any(op.hrt and op.start < cut < op.finish for op in reference):
        return False, "reference cut is not clean"
    if any(op.start is None or op.finish is None for op in candidate):
        return False, "candidate has unscheduled operations"
    if any(op.start < cut < op.finish for op in candidate):
        return False, "candidate cut is not clean (including BE tails)"
    prefix = [op for op in candidate if op.finish <= cut]
    reason = validate_operations(prefix, jobs, num_sas)
    if reason:
        return False, reason
    ref = {op.key: op for op in reference}
    cand = {op.key: op for op in prefix if op.hrt}
    if len(ref) != len(reference):
        return False, "duplicate reference identity"
    if any(key not in ref for key in cand):
        return False, "candidate HRT instance has no reference"
    if any((op.job, op.release, op.deadline) !=
           (ref[key].job, ref[key].release, ref[key].deadline)
           for key, op in cand.items()):
        return False, "candidate/reference instance metadata differ"
    # Condition 3: include all reference completions, not only starts in a tick.
    if any(op.finish <= cut and op.key not in cand for op in reference):
        return False, "candidate prefix does not cover reference prefix"
    # Condition 4: check every edge from completed to unfinished work.
    for op in cand.values():
        if op.layer + 1 == jobs[op.job].num_layers:
            continue  # A closed job was already checked against its deadline.
        successor = (True, op.instance, op.layer + 1)
        if successor not in cand:
            if successor not in ref:
                return False, "missing reference successor"
            if op.finish > ref[successor].start:
                return False, "candidate delays reference-suffix release"
    # Also certify the actual splice, including machine use by BE operations.
    suffix = [op for op in reference if op.key not in cand]
    reason = validate_operations(prefix + suffix, jobs, num_sas, complete=True)
    return (False, reason) if reason else (True, "safe prefix replacement")


def check_safe_continuation(reference, candidate, cut, jobs, num_sas,
                            future_reference=()):
    """Certify a concrete continuation, including operations crossing the cut.

    The four completed-prefix conditions still hold. Unlike the clean-cut
    theorem, this certificate explicitly retains every operation starting
    before the cut and verifies its coexistence with the reference suffix.
    Predictions starting at/after the cut are not committed by this decision.
    Future periodic reservations protect HRT instances which have not arrived.
    Durations must bound execution under the deployed interference model.
    """
    if not math.isfinite(cut):
        return False, "non-finite cut"
    reason = validate_operations(reference, jobs, num_sas, complete=True)
    if reason:
        return False, "invalid reference: " + reason
    if any(op.start is None or op.finish is None for op in candidate):
        return False, "candidate has unscheduled operations"
    if any(not math.isfinite(op.start) or not math.isfinite(op.finish)
           for op in candidate):
        return False, "non-finite candidate time"
    committed = [op for op in candidate if op.start < cut]
    reason = validate_operations(committed, jobs, num_sas)
    if reason:
        return False, reason
    ref = {op.key: op for op in reference}
    hrt = {op.key: op for op in committed if op.hrt}
    if len(ref) != len(reference):
        return False, "duplicate reference identity"
    if any(key not in ref for key in hrt):
        return False, "candidate HRT instance has no reference"
    if any((op.job, op.release, op.deadline) !=
           (ref[key].job, ref[key].release, ref[key].deadline)
           for key, op in hrt.items()):
        return False, "candidate/reference instance metadata differ"
    completed = {key: op for key, op in hrt.items() if op.finish <= cut}
    if any(op.finish <= cut and op.key not in completed for op in reference):
        return False, "candidate prefix does not cover reference prefix"
    for op in completed.values():
        if op.layer + 1 == jobs[op.job].num_layers:
            continue
        successor = (True, op.instance, op.layer + 1)
        if successor not in completed and op.finish > ref[successor].start:
            return False, "candidate delays reference-suffix release"
    suffix = [op for op in reference if op.key not in hrt]
    if any(op.start < cut for op in suffix):
        return False, "uncommitted reference suffix would start in the past"
    reason = validate_operations(committed + suffix, jobs, num_sas, complete=True)
    if reason:
        return False, reason
    # These reservations belong to jobs which may arrive in later ticks.
    # Their complete periodic schedule was validated when loading the reference.
    tails = [op for op in committed if op.finish > cut]
    by_sa = defaultdict(list)
    for op in future_reference:
        by_sa[op.sa].append(op)
    for op in tails:
        if any(op.start < reserved.finish and reserved.start < op.finish
               for reserved in by_sa[op.sa]):
            return False, "candidate tail blocks a future HRT reservation"
    return True, "safe reference continuation"


class PeriodicReference:
    """Validated periodic reservations with explicit customer/instance identity.

    Legacy YAML can be inferred only when deadline/period metadata identifies
    reservations unambiguously (interchangeable identical customers are OK).
    Sporadic arrivals use a reserved slot whose entire job is still executable;
    reservations are never independently shifted, which would break exclusivity.
    """

    def __init__(self, rows, customers, jobs, num_sas):
        self.jobs, self.num_sas = jobs, num_sas
        self.customers = {c[0]: tuple(c) for c in customers}
        if len(self.customers) != len(customers) or not customers:
            raise ReferenceScheduleError("HRT customers must have unique IDs")
        self.hyperperiod = 1
        for _, job, period, deadline in customers:
            if period <= 0 or int(period) != period or deadline <= 0:
                raise ReferenceScheduleError("invalid HRT period/deadline")
            self.hyperperiod = self.hyperperiod * int(period) // math.gcd(
                self.hyperperiod, int(period))
        groups = defaultdict(list)
        for row in rows:
            groups[row['job_instance_id']].append(row)
        self.templates = {}
        operations = []
        for instance, entries in sorted(groups.items()):
            first = entries[0]
            options = []
            for customer, job, period, deadline in customers:
                if job != first['job_id']:
                    continue
                release = first.get('arrival_time', first['absolute_deadline'] - deadline)
                if ('hrt_customer_id' in first
                        and first['hrt_customer_id'] != customer):
                    continue
                explicit_identity = ('arrival_time' in first
                                     and first.get('hrt_customer_id') == customer)
                deadline_matches = first['absolute_deadline'] == release + deadline
                if explicit_identity:
                    # EDF construction can deliberately tighten the deadline
                    # to reserve dispatch jitter. It remains a valid bound.
                    deadline_matches = release < first['absolute_deadline'] <= release + deadline
                if (0 <= release < self.hyperperiod
                        and release % period == 0
                        and deadline_matches
                        and (customer, release) not in self.templates):
                    options.append((customer, release, period, deadline))
            if not options or len({(o[1], o[3]) for o in options}) != 1:
                raise ReferenceScheduleError(
                    "cannot identify reference instance %s; regenerate with "
                    "customer and arrival metadata" % instance)
            customer, release, _, _ = options[0]
            ops = []
            for row in entries:
                if (row.get('hrt_customer_id', customer) != customer
                        or row.get('arrival_time', release) != release):
                    raise ReferenceScheduleError("inconsistent reference identity")
                ops.append(Operation(instance, row['job_id'], row['layer_id'],
                                     row['sa_id'], row['start_time'], row['finish_time'],
                                     release, row['absolute_deadline']))
            self.templates[customer, release] = sorted(ops, key=lambda op: op.layer)
            operations.extend(ops)
        for customer, _, period, _ in customers:
            if any((customer, r) not in self.templates
                   for r in range(0, self.hyperperiod, int(period))):
                raise ReferenceScheduleError("missing periodic HRT reservation")
        reason = validate_operations(operations, jobs, num_sas, complete=True)
        if reason:
            raise ReferenceScheduleError("invalid HRT reference: " + reason)
        # Project occupancy onto the hyperperiod circle. This also catches
        # conflicts between reservations separated by more than one cycle.
        cyclic = defaultdict(list)
        for op in operations:
            duration = op.finish - op.start
            if duration > self.hyperperiod:
                raise ReferenceScheduleError("reference cannot repeat: operation exceeds hyperperiod")
            start = op.start % self.hyperperiod
            end = start + duration
            cyclic[op.sa].append((start, min(end, self.hyperperiod)))
            if end > self.hyperperiod:
                cyclic[op.sa].append((0, end - self.hyperperiod))
        for intervals in cyclic.values():
            intervals.sort()
            if any(previous[1] > current[0]
                   for previous, current in zip(intervals, intervals[1:])):
                raise ReferenceScheduleError("reference cannot repeat: cyclic machine conflict")
        self.reset()

    def reset(self):
        self.bindings = {}
        self.last_slot = {}

    def future_operations(self, cut, end, next_arrivals):
        """Reservations overlapping tails, for every possible future arrival.

        Sporadic skips already drawn by the simulator can be respected. From
        each next arrival onwards, every task period is reserved again.
        """
        for customer, _, period, _ in self.customers.values():
            earliest = max(cut, next_arrivals[customer])
            first_slot = math.ceil(earliest / period)
            for slot in range(first_slot, math.ceil(end / period)):
                release = slot * period
                cycle, phase = divmod(release, self.hyperperiod)
                offset = cycle * self.hyperperiod
                for op in self.templates[customer, phase]:
                    start, finish = op.start + offset, op.finish + offset
                    if start < end and finish > cut:
                        yield replace(op, start=start, finish=finish,
                                      release=release, deadline=op.deadline + offset)

    def bind(self, sj, now):
        if sj.job_instance_id in self.bindings:
            return self.bindings[sj.job_instance_id]
        customer = getattr(sj, 'hrt_customer_id', None)
        if customer not in self.customers:
            matches = [c for c in self.customers.values()
                       if c[1] == sj.job_id and c[3] == sj.relative_deadline]
            if len(matches) != 1:
                raise ReferenceScheduleError("runtime HRT customer identity is missing")
            customer = matches[0][0]
        _, job, period, deadline = self.customers[customer]
        if job != sj.job_id or deadline != sj.relative_deadline:
            raise ReferenceScheduleError("runtime HRT workload differs from reference")
        first_slot = max(int(sj.arrival_time // period),
                         self.last_slot.get(customer, -1) + 1)
        for slot in range(first_slot, int(sj.absolute_deadline // period) + 1):
            release = slot * period
            cycle, phase = divmod(release, self.hyperperiod)
            template = self.templates[customer, phase]
            offset = cycle * self.hyperperiod
            if (template[0].start + offset < max(now, sj.arrival_time)
                    or template[-1].finish + offset > sj.absolute_deadline):
                continue
            bound = [replace(op, instance=sj.job_instance_id,
                             start=op.start + offset, finish=op.finish + offset,
                             release=sj.arrival_time, deadline=sj.absolute_deadline)
                     for op in template]
            self.bindings[sj.job_instance_id] = bound
            self.last_slot[customer] = slot
            return bound
        raise ReferenceScheduleError(
            "no executable reference reservation for HRT instance %s at %s "
            "(arrival %s, deadline %s); dispatch period/arrival pattern is "
            "incompatible with this reference" %
            (sj.job_instance_id, now, sj.arrival_time, sj.absolute_deadline))
