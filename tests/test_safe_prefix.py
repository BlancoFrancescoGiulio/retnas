import contextlib
from dataclasses import replace
import io
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from retnas.environment import Environment
from retnas.job import Job
from retnas.safe_prefix import Operation, PeriodicReference, ReferenceScheduleError, check_safe_prefix, check_safe_continuation

def workloads():
    return [Job('HRT', 5, [(2, 3), (4, 3)]), Job('BE', 1, [(1, 2)])]

def reference(instance=0, release=0, first=6, last=12, customer=10):
    return [dict(job_instance_id=instance, job_id=0, layer_id=0, sa_id=0, start_time=release + first, finish_time=release + first + 2, absolute_deadline=release + 18, arrival_time=release, hrt_customer_id=customer), dict(job_instance_id=instance, job_id=0, layer_id=1, sa_id=1, start_time=release + last, finish_time=release + last + 3, absolute_deadline=release + 18, arrival_time=release, hrt_customer_id=customer)]

def operations(rows):
    return [Operation(r['job_instance_id'], r['job_id'], r['layer_id'], r['sa_id'], r['start_time'], r['finish_time'], r['arrival_time'], r['absolute_deadline']) for r in rows]

class PaperConditionsTests(unittest.TestCase):

    def setUp(self):
        self.jobs = workloads()
        self.ref = operations(reference())

    def check(self, cand, cut, ref=None):
        return check_safe_prefix(self.ref if ref is None else ref, cand, cut, self.jobs, 2)

    def test_closed_job_may_be_slower_than_reference_if_deadline_met(self):
        cand = [replace(self.ref[0], start=9, finish=11), replace(self.ref[1], start=15, finish=18)]
        self.assertTrue(self.check(cand, 18)[0])

    def test_closed_job_deadline_is_not_relaxed_by_large_dispatch_tick(self):
        cand = [replace(self.ref[0], start=9, finish=11), replace(self.ref[1], start=16, finish=19)]
        (safe, reason) = self.check(cand, 20)
        self.assertFalse(safe)
        self.assertIn('deadline', reason)

    def test_open_interface_uses_successor_start_not_predecessor_finish(self):
        cand = [replace(self.ref[0], start=8, finish=10), self.ref[1]]
        self.assertGreater(cand[0].finish, self.ref[0].finish)
        self.assertTrue(self.check(cand, 10)[0])

    def test_reference_completion_must_be_covered(self):
        cand = [replace(self.ref[0], start=10, finish=12), replace(self.ref[1], start=12, finish=15)]
        (safe, reason) = self.check(cand, 9)
        self.assertFalse(safe)
        self.assertIn('cover', reason)

    def test_precedence_closed_prefix(self):
        cand = [replace(self.ref[0], start=6, finish=8), replace(self.ref[1], start=5, finish=8)]
        (safe, reason) = self.check(cand, 9)
        self.assertFalse(safe)
        self.assertIn('precedence', reason)

    def test_machine_exclusivity_including_best_effort(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=2, finish=5), Operation(1, 1, 0, 0, 1, 2, 0, 30, False)]
        (safe, reason) = self.check(cand, 5)
        self.assertFalse(safe)
        self.assertIn('exclusivity', reason)

    def test_eligibility_release_and_duration(self):
        for op in (replace(self.ref[0], sa=9), replace(self.ref[0], start=-1, finish=1), replace(self.ref[0], start=6, finish=7), replace(self.ref[0], finish=float('nan'))):
            with self.subTest(op=op):
                self.assertFalse(self.check([op, self.ref[1]], 18)[0])

    def test_reference_cut_crossing_policy(self):
        (safe, reason) = self.check(self.ref, 7)
        self.assertFalse(safe)
        self.assertIn('reference cut', reason)

    def test_candidate_cut_crossing_policy(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=3, finish=6)]
        self.assertIn('candidate cut', self.check(cand, 5)[1])

    def test_be_tail_cannot_block_future_hrt_suffix(self):
        cand = self.ref + [Operation(1, 1, 0, 0, 4, 7, 0, 30, False)]
        self.assertIn('candidate cut', self.check(cand, 5)[1])

    def test_half_open_boundaries(self):
        self.assertTrue(self.check(self.ref, 8)[0])
        self.assertTrue(self.check(self.ref, 12)[0])

    def test_distinct_instances_of_same_job_are_not_collapsed(self):
        other = [replace(op, instance=1, start=op.start + 20, finish=op.finish + 20, release=20, deadline=38) for op in self.ref]
        cand = self.ref + [replace(other[0], start=30, finish=32), other[1]]
        self.assertIn('cover', self.check(cand, 29, self.ref + other)[1])

class ContinuationTests(PaperConditionsTests):
    """Reuse the four paper-condition cases with the continuation certificate."""

    def check(self, cand, cut, ref=None, future=()):
        return check_safe_continuation(self.ref if ref is None else ref, cand, cut, self.jobs, 2, future)

    def test_reference_cut_crossing_policy(self):
        cand = [replace(self.ref[0], start=8, finish=10), self.ref[1]]
        self.assertIn('past', self.check(cand, 7)[1])

    def test_candidate_cut_crossing_policy(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=3, finish=6)]
        self.assertTrue(self.check(cand, 5)[0])

    def test_be_tail_cannot_block_future_hrt_suffix(self):
        cand = self.ref + [Operation(1, 1, 0, 0, 4, 7, 0, 30, False)]
        self.assertIn('exclusivity', self.check(cand, 5)[1])

    def test_dirty_reference_is_irrelevant_when_candidate_already_finished(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=2, finish=5)]
        self.assertTrue(self.check(cand, 7)[0])

    def test_be_tail_is_allowed_in_a_free_reference_gap(self):
        cand = [replace(self.ref[0], start=0, finish=2), self.ref[1], Operation(1, 1, 0, 0, 4, 7, 0, 30, False)]
        self.assertTrue(self.check(cand, 5)[0])

    def test_inflight_predecessor_must_release_reference_successor(self):
        cand = [replace(self.ref[0], start=4, finish=13), self.ref[1]]
        self.assertIn('precedence', self.check(cand, 7)[1])

    def test_inflight_layer_must_still_meet_deadline(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=3, finish=19)]
        self.assertIn('deadline', self.check(cand, 5)[1])

    def test_uncommitted_prediction_can_be_replaced_by_valid_reference(self):
        cand = [replace(self.ref[0], start=0, finish=2), replace(self.ref[1], start=12, finish=100)]
        self.assertTrue(self.check(cand, 5)[0])

    def test_candidate_tail_protects_not_yet_arrived_hrt_jobs(self):
        cand = [replace(self.ref[0], start=0, finish=2), self.ref[1], Operation(1, 1, 0, 0, 4, 27, 0, 30, False)]
        future = [replace(self.ref[0], instance=2, start=26, finish=28, release=20, deadline=38)]
        self.assertIn('future HRT', self.check(cand, 5, future=future)[1])
        self.assertTrue(self.check(cand[:-1] + [replace(cand[-1], finish=26)], 5, future=future)[0])

class ReferenceIdentityTests(unittest.TestCase):

    def setUp(self):
        self.jobs = workloads()

    def test_customers_and_repetitions_have_distinct_reservations(self):
        rows = reference(first=0, last=2) + reference(instance=1, first=6, last=8, customer=11)
        ref = PeriodicReference(rows, [(10, 0, 20, 18), (11, 0, 20, 18)], self.jobs, 2)
        a = SimpleNamespace(job_instance_id=100, job_id=0, relative_deadline=18, arrival_time=0, absolute_deadline=18, hrt_customer_id=10)
        b = SimpleNamespace(**dict(vars(a), job_instance_id=101, hrt_customer_id=11))
        c = SimpleNamespace(**dict(vars(a), job_instance_id=102, arrival_time=20, absolute_deadline=38))
        self.assertEqual(ref.bind(a, 0)[0].start, 0)
        self.assertEqual(ref.bind(b, 0)[0].start, 6)
        self.assertEqual(ref.bind(c, 20)[0].start, 20)

    def test_legacy_reference_with_unambiguous_identity(self):
        rows = [{k: v for (k, v) in r.items() if k not in ('arrival_time', 'hrt_customer_id')} for r in reference()]
        PeriodicReference(rows, [(10, 0, 20, 18)], self.jobs, 2)

    def test_future_calendar_includes_later_cycles_and_respects_known_skips(self):
        ref = PeriodicReference(reference(), [(10, 0, 20, 18)], self.jobs, 2)
        future = list(ref.future_operations(5, 55, {10: 20}))
        self.assertEqual([op.start for op in future], [26, 32, 46, 52])
        self.assertEqual([op.start for op in ref.future_operations(5, 55, {10: 40})], [46, 52])

    def test_reference_precedence_violation_is_rejected(self):
        rows = reference(last=7)
        with self.assertRaisesRegex(ReferenceScheduleError, 'precedence'):
            PeriodicReference(rows, [(10, 0, 20, 18)], self.jobs, 2)

    def test_missing_layer_or_periodic_instance_is_rejected(self):
        with self.assertRaisesRegex(ReferenceScheduleError, 'incomplete'):
            PeriodicReference(reference()[:1], [(10, 0, 20, 18)], self.jobs, 2)
        with self.assertRaisesRegex(ReferenceScheduleError, 'missing periodic'):
            PeriodicReference(reference(), [(10, 0, 20, 18), (11, 0, 30, 18)], self.jobs, 2)

    def test_late_arrival_is_never_assigned_past_timestamps(self):
        ref = PeriodicReference(reference(), [(10, 0, 20, 18)], self.jobs, 2)
        sj = SimpleNamespace(job_instance_id=100, job_id=0, relative_deadline=18, arrival_time=9, absolute_deadline=27, hrt_customer_id=10)
        with self.assertRaisesRegex(ReferenceScheduleError, 'no executable'):
            ref.bind(sj, 9)

    def test_release_jitter_can_use_an_existing_slot_without_shifting_it(self):
        ref = PeriodicReference(reference(), [(10, 0, 20, 18)], self.jobs, 2)
        sj = SimpleNamespace(job_instance_id=100, job_id=0, relative_deadline=18, arrival_time=3, absolute_deadline=21, hrt_customer_id=10)
        bound = ref.bind(sj, 3)
        self.assertEqual([op.start for op in bound], [6, 12])
        self.assertEqual(bound[-1].deadline, 21)

    def test_conflict_across_nonadjacent_cycles_is_rejected(self):
        delayed = reference(instance=1, first=27, last=32, customer=11)
        for row in delayed:
            row['absolute_deadline'] = 58
        with self.assertRaisesRegex(ReferenceScheduleError, 'cyclic machine'):
            PeriodicReference(reference() + delayed, [(10, 0, 20, 18), (11, 0, 20, 58)], self.jobs, 2)

class DeploymentTests(unittest.TestCase):

    def make_env(self, period=5):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'reference.yaml'
        path.write_text(yaml.safe_dump({'feasible_schedule': reference()}))
        jobs = workloads()
        config = SimpleNamespace(num_sas=2, all_jobs=jobs, jobs=jobs, possible_jobs=[0, 1], qos_factor=3, jobs_per_arrival=1, alpha=5, dscale=10, next_state_mode='noarrivals', prepend_sa_remaining_time=False, normalize_reward=False, full_sim_reward_factor=0, reward_slack_factor=0, scheduling_period=period, min_deadline_period_factor=2, hrt=True, hrt_step_mode='deployment', hrt_customers=[(10, 0, 20 if period <= 0 else 20 * int(period) // math.gcd(20, int(period)), 18)], wcet_schedule_path=str(path))
        env = Environment(config, seed=1)
        env.reset(no_new_jobs=True)
        env.new_job_arrival(0, is_hrt=True, hrt_job_id=0, hrt_deadline=18, hrt_customer_id=10)
        return env

    @staticmethod
    def action(env, sa_for_last=1):
        action = []
        for sj in env.rq:
            sa = sa_for_last if sj.layer_id == 1 else 0
            action.append([0 if sj.is_hrt else 2, int(sa == 0), int(sa == 1)])
        return np.array(action if action else [[0, 1, 0]], dtype=float)

    def test_clean_valid_actor_prefix_is_accepted(self):
        env = self.make_env()
        env.step(self.action(env))
        self.assertEqual(env.hrt_rl_win, 1)
        self.assertEqual(env.hrt_wcet_win, 0)
        self.assertEqual(env.deadline_hits_hrt, 1)
        self.assertFalse(env.rq)

    def test_fallback_preserves_inflight_layers_and_later_counts_completion(self):
        env = self.make_env(period=7)
        env.jobs[0].layer_latencies[0] = (2, 20)
        env.jobs[0].layer_latencies[1] = (20, 3)
        action = self.action(env)
        action[0] = [0, 0, 1]
        env.step(action)
        self.assertEqual(env.hrt_wcet_win, 1)
        committed = env._committed_operations[True, 0, 0]
        self.assertEqual((committed.start, committed.finish), (6, 8))
        self.assertEqual([sj.layer_id for sj in env.rq], [1])
        env.step(self.action(env, sa_for_last=0))
        self.assertEqual(env._committed_operations[True, 0, 0], committed)
        self.assertEqual(env._committed_operations[True, 0, 1].start, 12)
        self.assertEqual(env.deadline_hits_hrt, 0)
        env.step(self.action(env))
        self.assertEqual(env.deadline_hits_hrt, 1)
        env.step(self.action(env))
        self.assertEqual(env.deadline_hits_hrt, 1)

    def test_started_predecessor_still_constrains_next_training_step(self):
        env = self.make_env(period=1)
        env.hrt_step_mode = 'all'
        env.step(self.action(env, sa_for_last=0))
        self.assertEqual(env._committed_operations[True, 0, 0].finish, 2)
        env.step(self.action(env, sa_for_last=1))
        self.assertEqual(env.old_rq[0].start_time, 2)

    def test_reset_clears_bindings_and_committed_history(self):
        env = self.make_env(period=7)
        env.step(self.action(env))
        env.reset(no_new_jobs=True)
        self.assertFalse(env._committed_operations)
        self.assertFalse(env._hrt_reference.bindings)
        self.assertFalse(env._pending_hrt_completions)
        self.assertFalse(env._pending_be_completions)
        self.assertFalse(env.hrt_guardian_rejections)
        self.assertEqual((env.hrt_rl_win, env.hrt_wcet_win), (0, 0))

    def test_prema_uses_the_same_safe_prefix_guardian(self):
        env = self.make_env(period=7)
        env.prema_step(self.action(env))
        self.assertEqual(env.hrt_wcet_win, 0)
        self.assertEqual(env.hrt_rl_win, 1)
        self.assertEqual(env._committed_operations[True, 0, 0].finish, 2)

    def test_safe_be_last_layer_tail_is_counted_once_when_it_finishes(self):
        env = self.make_env()
        env.jobs[1].layer_latencies[0] = (3, 2)
        env.tenant_jobs = [[1]]
        env.arrived_jobs_per_tenant = [0]
        env.deadline_hits_per_tenant = [0]
        env.new_job_arrival(3, tenant_id=0)
        env.step(self.action(env))
        self.assertEqual(env.hrt_wcet_win, 0)
        self.assertEqual(env.deadline_hits_be, 0)
        self.assertEqual(len(env._pending_be_completions), 1)
        env.step(self.action(env))
        self.assertEqual(env.deadline_hits_be, 1)
        env.step(self.action(env))
        self.assertEqual(env.deadline_hits_be, 1)

    def test_empty_hrt_customer_list_needs_no_reference(self):
        env = self.make_env()
        env.hrt_customers = []
        env.wcet_schedule = None
        env.rq.clear()
        env.imminents.clear()
        env.step(self.action(env))
        self.assertIsNone(env._hrt_reference)

    def test_missing_reference_and_nonpositive_period_stop_before_dispatch(self):
        for (period, rows) in ((0, reference()), (5, [])):
            env = self.make_env(period=period)
            env.wcet_schedule = rows
            with self.assertRaises(ReferenceScheduleError):
                env.step(self.action(env))
            self.assertEqual(env.current_time, 0)
            self.assertFalse(env._committed_operations)

    def test_rebased_training_episode_preserves_predecessor_finish(self):
        env = self.make_env(period=1)
        env.hrt_step_mode = 'all'
        env.step(self.action(env))
        env.reset(clear=False)
        env.step(self.action(env))
        self.assertEqual(env.old_rq[0].start_time, 1)

    def test_generator_covers_last_tick_and_restores_legacy_model_settings(self):
        import torch
        from retnas.admission_control import test_wcet_schedule_with_model as generate_wcet
        env = self.make_env()
        env.hrt_customers = [(10, 0, 20, 18), (11, 1, 5, 4)]
        env.hrt_next_arrival[11] = 0
        cfg = SimpleNamespace(num_layers=1, policy_type='gru', num_sas=2, hrt=True, hrt_flag_provided_to_the_network=True, actor_hidden_size=4, use_packing=False, scheduling_period=5)

        class Actor:

            def __init__(self, *args, **kwargs):
                pass

            def to(self, device):
                return self

            def load_state_dict(self, checkpoint):
                pass

            def eval(self):
                pass

            def __call__(self, state, lengths):
                self_width = state.shape[-1]
                if self_width != 6:
                    raise AssertionError('legacy checkpoint received the HRT flag')
                action = torch.zeros((*state.shape[:2], 3), device=state.device)
                action[..., 0] = -state[..., 0]
                action[..., 1] = 1
                return action
        output = io.StringIO()
        arrivals = []
        original_arrivals = env.new_jobs_arrival

        def track_arrivals(*args, **kwargs):
            arrivals.append((env.current_time, env.next_state_mode, kwargs.get('hrt_only'), kwargs.get('hrt_all_periods')))
            return original_arrivals(*args, **kwargs)
        with patch('retnas.admission_control.torch.load', return_value={'core.weight_ih_l0': torch.zeros((4, 6))}), patch('retnas.admission_control.SchedulerNetworkActorRNN', Actor), patch.object(env, 'new_jobs_arrival', side_effect=track_arrivals), contextlib.redirect_stdout(output):
            (rows, feasible) = generate_wcet(cfg, 'unused', env)
        self.assertTrue(feasible, output.getvalue())
        self.assertTrue(any((row['arrival_time'] == 15 for row in rows)))
        self.assertEqual(env.deadline_hits_hrt, 5)
        self.assertEqual(env.hrt_step_mode, 'deployment')
        self.assertTrue(env.hrt_flag_provided_to_the_network)
        self.assertEqual(env.next_state_mode, 'noarrivals')
        self.assertIsNone(env._hrt_arrival_horizon)
        self.assertEqual([tick for (tick, *_) in arrivals], [0, 5, 10, 15, 20])
        self.assertTrue(all((mode == 'hrt_only_all_periods' and hrt_only and all_periods for (_, mode, hrt_only, all_periods) in arrivals)))
        self.assertEqual(sorted({row['arrival_time'] for row in rows}), [0, 5, 10, 15])

    def test_nonmultiple_dispatch_period_is_rejected_before_execution(self):
        env = self.make_env()
        env.scheduling_period = 3
        with self.assertRaisesRegex(ReferenceScheduleError, 'multiple'):
            env.step(self.action(env))
        self.assertEqual(env.current_time, 0)

    def test_generator_checks_task_period_multiples(self):
        from retnas.admission_control import test_wcet_schedule_with_model as generate_wcet
        env = self.make_env()
        env.scheduling_period = 3
        cfg = SimpleNamespace(scheduling_period=3)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ReferenceScheduleError, 'multiple'):
            generate_wcet(cfg, 'unused', env)
        self.assertEqual(env.next_state_mode, 'noarrivals')
        self.assertIsNone(env._hrt_arrival_horizon)

    def test_invalid_generated_reference_does_not_overwrite_existing_file(self):
        from retnas.admission_control import save_feasible_scheduling
        env = self.make_env()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reference.yaml'
            path.write_text('original')
            with self.assertRaises(ReferenceScheduleError):
                save_feasible_scheduling(reference(last=7), str(path), env=env)
            self.assertEqual(path.read_text(), 'original')

    def test_edf_generator_retains_identity_with_tightened_deadlines(self):
        from retnas.admission_control import test_wcet_schedule_with_edf as generate_edf
        env = self.make_env()
        cfg = SimpleNamespace(scheduling_period=5, num_sas=2)
        with contextlib.redirect_stdout(io.StringIO()):
            (rows, feasible) = generate_edf(cfg, env)
        self.assertTrue(feasible)
        self.assertTrue(all((row['hrt_customer_id'] == 10 for row in rows)))
        self.assertTrue(all((row['absolute_deadline'] == 13 for row in rows)))
        self.assertEqual(env.hrt_step_mode, 'deployment')
if __name__ == '__main__':
    unittest.main()
