"""Release-boundary regressions for workload identity and small training runs."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from retnas.buffer import ReplayBuffer
from retnas.retnas import ROOT, load_config


class ReleaseTests(unittest.TestCase):
    def test_stable_workload_ids_select_real_dnn_profiles(self):
        config = load_config(ROOT / 'configs/quick_reference.yaml')
        self.assertEqual(config.all_jobs[7].name, 'YOLOv2')
        self.assertEqual(config.all_jobs[7].num_layers, 22)
        self.assertEqual(config.all_jobs[0].num_layers, 8)
        self.assertIsNone(config.all_jobs[5])
        self.assertEqual(config.all_jobs[7].layer_latencies[0][1],
                         config.all_jobs[7].layer_latencies[0][2])

    def test_paths_do_not_depend_on_config_location(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text((ROOT / 'configs/quick_reference.yaml').read_text())
            config = load_config(path)
        self.assertEqual(Path(config.data_csv_path), ROOT / 'data')

    def test_unsupported_workload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Workload IDs'):
            load_config(ROOT / 'configs/quick_reference.yaml', {'possible_jobs': [2]})

    def test_task_period_alignment_is_checked(self):
        with self.assertRaisesRegex(ValueError, 'multiple'):
            load_config(ROOT / 'configs/quick_reference.yaml', {'scheduling_period': 3})

    def test_stage_files_use_required_modes(self):
        expected = {
            'train_wcet': ('all', 'hrt_only_all_periods', False),
            'reference': ('all', 'hrt_only_all_periods', False),
            'train_scheduler': ('all', 'nonewjobs', True),
            'deployment': ('deployment', 'nonewjobs', True),
        }
        for prefix in ('', 'quick_'):
            for stage, modes in expected.items():
                c = yaml.safe_load((ROOT / f'configs/{prefix}{stage}.yaml').read_text())['config']
                self.assertIs(c['hrt'], True)
                self.assertEqual(tuple(c[key] for key in ('hrt_step_mode', 'next_state_mode',
                                                         'hrt_flag_provided_to_the_network')), modes)

    def test_small_replay_population_samples_with_replacement(self):
        buffer = ReplayBuffer(max_size=4)
        state = np.ones((2, 3), dtype=np.float32)
        action = np.ones((2, 2), dtype=np.float32)
        buffer.push((state, state, action, 1.0, False))
        states, _, _, rewards, lengths, _, _ = buffer.sample(8)
        self.assertEqual(states.shape, (8, 2, 3))
        np.testing.assert_array_equal(lengths, np.full(8, 2))
        np.testing.assert_array_equal(rewards, np.ones((8, 1)))


if __name__ == '__main__':
    unittest.main()
