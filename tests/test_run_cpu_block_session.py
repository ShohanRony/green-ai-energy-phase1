import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import run_cpu_block_session as runner_mod


def stub_runner(calls):
    """Records every invocation instead of launching a real process -- no real model or
    instrument is ever touched by these tests."""
    def _run(pilot_py, python_exe, args):
        calls.append(args)
        if '--out' in args:
            out_idx = args.index('--out')
            out_dir = Path(args[out_idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            import csv
            with open(out_dir / 'windows.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['repeat', 'phase', 'thermal_flag'])
                w.writeheader()
                w.writerow({'repeat': 0, 'phase': 'a1', 'thermal_flag': 'False'})
        class _Result:
            returncode = 0
        return _Result()
    return _run


class ConditionListTests(unittest.TestCase):
    def test_default_excludes_known_collapsed_bnrecal(self):
        conds = runner_mod.conditions_for('mobilenet_v3_small')
        self.assertNotIn('pruned50_bnrecal', conds)
        self.assertNotIn('pruned70_bnrecal', conds)
        self.assertIn('pruned30_bnrecal', conds)

    def test_include_collapsed_flag_adds_them_back(self):
        conds = runner_mod.conditions_for('mobilenet_v3_small', include_collapsed_bnrecal=True)
        self.assertIn('pruned50_bnrecal', conds)
        self.assertIn('pruned70_bnrecal', conds)

    def test_resnet18_has_no_collapsed_exclusions(self):
        conds = runner_mod.conditions_for('resnet18')
        self.assertEqual(len(conds), 9)  # fp32_ts, fp32_eager, int8, 3 pruned, 3 bnrecal

    def test_no_fp16_anywhere(self):
        for arch in ('resnet18', 'mobilenet_v3_small', 'efficientnet_b0'):
            self.assertNotIn('fp16', runner_mod.conditions_for(arch))


class SessionOrderTests(unittest.TestCase):
    def test_same_session_same_order_deterministic(self):
        archs = ['resnet18', 'mobilenet_v3_small']
        o1 = runner_mod.session_order(1, archs)
        o2 = runner_mod.session_order(1, archs)
        self.assertEqual(o1, o2)

    def test_different_sessions_different_order(self):
        archs = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
        o1 = runner_mod.session_order(1, archs)
        o2 = runner_mod.session_order(2, archs)
        self.assertNotEqual(o1, o2)

    def test_order_is_a_permutation_not_a_subset(self):
        archs = ['resnet18']
        order = runner_mod.session_order(3, archs)
        expected = {(arch, c) for arch in archs for c in runner_mod.conditions_for(arch)}
        self.assertEqual(set(order), expected)


class PilotArgsTests(unittest.TestCase):
    def test_fp32_eager_uses_arch_not_checkpoint(self):
        args = runner_mod.pilot_args_for('resnet18', 'fp32_eager', '/tmp/out', '/tmp/ckpt')
        self.assertIn('--arch', args)
        self.assertNotIn('--checkpoint', args)

    def test_fp32_ts_loads_plain_fp32_checkpoint(self):
        args = runner_mod.pilot_args_for('resnet18', 'fp32_ts', '/tmp/out', 'checkpoints')
        self.assertIn('checkpoints/resnet18_fp32.pt', args)

    def test_pruned_condition_loads_matching_checkpoint(self):
        args = runner_mod.pilot_args_for('efficientnet_b0', 'pruned30_bnrecal', '/tmp/out', 'checkpoints')
        self.assertIn('checkpoints/efficientnet_b0_pruned30_bnrecal.pt', args)

    def test_cpu_affinity_always_requested(self):
        args = runner_mod.pilot_args_for('resnet18', 'int8', '/tmp/out', 'checkpoints')
        self.assertIn('--cpu-affinity', args)
        self.assertIn('pcores', args)


class RunSessionEndToEndTests(unittest.TestCase):
    """Exercises the full orchestration path with a stub instrument (no real pilot.py
    subprocess) and a dummy architecture list, writing only to a tempdir outside the repo,
    deleted at the end of every test."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='cpu_block_session_test_')
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch('run_cpu_block_session.check_ac_power', return_value=True)
    def test_full_session_calls_runner_once_per_condition(self, _mock_ac):
        calls = []
        log = runner_mod.run_session(
            session_n=1, architectures=['resnet18'], out_base=self.tmp,
            checkpoints_dir='checkpoints', runner=stub_runner(calls))
        self.assertEqual(len(calls), len(runner_mod.conditions_for('resnet18')))
        self.assertEqual(len(log['conditions']), len(calls))

    @patch('run_cpu_block_session.check_ac_power', return_value=True)
    def test_session_log_written_to_tempdir(self, _mock_ac):
        runner_mod.run_session(
            session_n=2, architectures=['resnet18'], out_base=self.tmp,
            checkpoints_dir='checkpoints', runner=stub_runner([]))
        log_path = Path(self.tmp) / 'session_log.json'
        self.assertTrue(log_path.exists())
        data = json.loads(log_path.read_text())
        self.assertEqual(data['session'], 2)
        self.assertEqual(data['seed'], runner_mod.CONDITION_SEED_BASE + 2)

    @patch('run_cpu_block_session.check_ac_power', return_value=False)
    def test_ac_offline_aborts_before_any_call(self, _mock_ac):
        calls = []
        with self.assertRaises(RuntimeError):
            runner_mod.run_session(
                session_n=1, architectures=['resnet18'], out_base=self.tmp,
                checkpoints_dir='checkpoints', runner=stub_runner(calls))
        self.assertEqual(calls, [])

    def test_skip_ac_check_flag_bypasses_the_guard(self):
        calls = []
        runner_mod.run_session(
            session_n=1, architectures=['resnet18'], out_base=self.tmp,
            checkpoints_dir='checkpoints', runner=stub_runner(calls), skip_ac_check=True)
        self.assertTrue(len(calls) > 0)

    @patch('run_cpu_block_session.check_ac_power', return_value=True)
    def test_nothing_written_outside_tempdir(self, _mock_ac):
        before = set(Path(self.tmp).parent.iterdir())
        runner_mod.run_session(
            session_n=1, architectures=['resnet18'], out_base=self.tmp,
            checkpoints_dir='checkpoints', runner=stub_runner([]))
        after = set(Path(self.tmp).parent.iterdir())
        self.assertEqual(before, after)  # no new sibling directories created


if __name__ == '__main__':
    unittest.main()


class ThermalVoidRerunTests(unittest.TestCase):
    def _write_csv(self, out_dir, rows):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pilot import csv_write
        csv_write(Path(out_dir) / 'windows.csv', rows)

    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_flagged_warm_condition_is_voided_and_rerun_at_end(self, _ac):
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            calls = []
            flag_count = 0
            def runner(pilot_py, python_exe, args):
                nonlocal flag_count
                calls.append(args)
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                if 'fp32_eager' in str(args) and flag_count == 0:
                    self._write_csv(out_dir, [
                        {'repeat': 0, 'phase': 'a1', 'thermal_flag': 'False'},
                        {'repeat': 1, 'phase': 'a1', 'thermal_flag': 'True'}
                    ])
                    flag_count += 1
                else:
                    self._write_csv(out_dir, [
                        {'repeat': 0, 'phase': 'a1', 'thermal_flag': 'False'},
                        {'repeat': 1, 'phase': 'a1', 'thermal_flag': 'False'}
                    ])

            log = runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)
            
            eager_runs = [c for c in calls if 'fp32_eager' in str(c)]
            self.assertEqual(len(eager_runs), 2)
            self.assertTrue('_rerun' in str(eager_runs[-1]))
            
            void_log = json.loads((Path(td) / 'void_log.json').read_text())
            self.assertEqual(len(void_log['voids']), 1)
            self.assertEqual(void_log['voids'][0]['condition'], 'resnet18_fp32_eager')

    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_flagged_cold_window_only_is_not_voided(self, _ac):
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            def runner(pilot_py, python_exe, args):
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(out_dir, [
                    {'repeat': 0, 'phase': 'a1', 'thermal_flag': 'True'},
                    {'repeat': 1, 'phase': 'a1', 'thermal_flag': 'False'}
                ])

            log = runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)
            void_log = json.loads((Path(td) / 'void_log.json').read_text())
            self.assertEqual(len(void_log['voids']), 0)

    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_missing_thermal_flag_column_raises(self, _ac):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            def runner(pilot_py, python_exe, args):
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(out_dir, [{'repeat': 0, 'phase': 'a1', 'some_other_col': 'False'}])

            with self.assertRaisesRegex(RuntimeError, "thermal_flag column missing"):
                runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)

    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_more_than_two_voids_aborts_session(self, _ac):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            def runner(pilot_py, python_exe, args):
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(out_dir, [{'repeat': 1, 'phase': 'a1', 'thermal_flag': 'True'}])

            with self.assertRaisesRegex(RuntimeError, "Session voided: more than 2 voided conditions"):
                runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)
    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_runner_ignores_low_frequency_idle_row(self, _ac):
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            def runner(pilot_py, python_exe, args):
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(out_dir, [
                    {'repeat': 0, 'phase': 'a1', 'thermal_flag': 'False'},
                    {'repeat': 1, 'phase': 'a1', 'thermal_flag': 'False'},
                    {'repeat': 1, 'phase': 'idle_before', 'thermal_flag': 'True'}
                ])

            log = runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)
            void_log = json.loads((Path(td) / 'void_log.json').read_text())
            self.assertEqual(len(void_log['voids']), 0)

    @patch('scripts.run_cpu_block_session.check_ac_power', return_value=True)
    def test_runner_raises_on_active_warm_row_with_empty_thermal_flag(self, _ac):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            def runner(pilot_py, python_exe, args):
                out_dir = Path(args[args.index('--out') + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                self._write_csv(out_dir, [
                    {'repeat': 0, 'phase': 'a1', 'thermal_flag': 'False'},
                    {'repeat': 1, 'phase': 'a1', 'thermal_flag': ''}
                ])

            with self.assertRaisesRegex(RuntimeError, "Evaluation missing"):
                runner_mod.run_session(10, ['resnet18'], td, 'checkpts', runner=runner)

class SessionGovernorCheckTests(unittest.TestCase):
    def test_failing_governor_check_stops_session(self):
        def fake_governor_check():
            raise RuntimeError("Fake governor failure")
        calls = []
        with self.assertRaisesRegex(RuntimeError, "Fake governor failure"):
            runner_mod.run_session(
                session_n=1, architectures=['resnet18'], out_base='/tmp/fake_base',
                checkpoints_dir='checkpoints', runner=stub_runner(calls),
                skip_ac_check=True, governor_check=fake_governor_check)
        self.assertEqual(calls, [])

