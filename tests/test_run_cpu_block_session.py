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
