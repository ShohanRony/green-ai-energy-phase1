"""Unit tests for M1 session runner.

Verifies:
1. Exact 23 conditions generated for standard 3 architectures.
2. mobilenet_v3_small int8 structurally excluded.
3. Deterministic order for seeds 4001-4006.
4. Checkpoint mapping.
5. Guard enforcement and void / rerun tracking.
"""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from guards import GuardError
from m1_session_runner import (
    BNRECAL_SKIP,
    CONDITION_SEED_BASE,
    M1_STRUCTURAL_EXCLUSIONS,
    checkpoint_for,
    conditions_for,
    run_m1_session,
    session_order,
)


class TestM1SessionRunner(unittest.TestCase):
    def setUp(self):
        self.architectures = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']

    def test_23_conditions_total(self):
        all_pairs = [
            (arch, cond)
            for arch in self.architectures
            for cond in conditions_for(arch)
        ]
        self.assertEqual(len(all_pairs), 23)
        self.assertEqual(len(conditions_for('resnet18')), 9)
        self.assertEqual(len(conditions_for('mobilenet_v3_small')), 6)
        self.assertEqual(len(conditions_for('efficientnet_b0')), 8)

    def test_mobilenet_int8_excluded(self):
        mb_conds = conditions_for('mobilenet_v3_small')
        self.assertNotIn('int8', mb_conds)
        self.assertIn(('mobilenet_v3_small', 'int8'), M1_STRUCTURAL_EXCLUSIONS)
        with self.assertRaises(ValueError):
            checkpoint_for('mobilenet_v3_small', 'int8')

    def test_deterministic_session_order(self):
        order1 = session_order(1, self.architectures)
        order1_again = session_order(1, self.architectures)
        self.assertEqual(order1, order1_again)
        self.assertEqual(len(order1), 23)

        # Different session produces different order
        order2 = session_order(2, self.architectures)
        self.assertNotEqual(order1, order2)
        self.assertEqual(len(order2), 23)

    def test_conditions_order_matches_x86_structure(self):
        """A6: base states first (fp32_ts, fp32_eager, [int8], pruned30,
        pruned50, pruned70), then all bnrecal states appended at the end --
        matching scripts/run_cpu_block_session.py's conditions_for exactly,
        not interleaved per-ratio."""
        resnet_conds = conditions_for('resnet18')
        self.assertEqual(
            resnet_conds,
            ['fp32_ts', 'fp32_eager', 'int8', 'pruned30', 'pruned50', 'pruned70',
             'pruned30_bnrecal', 'pruned50_bnrecal', 'pruned70_bnrecal'],
        )
        mb_conds = conditions_for('mobilenet_v3_small')
        self.assertEqual(
            mb_conds,
            ['fp32_ts', 'fp32_eager', 'pruned30', 'pruned50', 'pruned70',
             'pruned30_bnrecal'],
        )
        eff_conds = conditions_for('efficientnet_b0')
        self.assertEqual(
            eff_conds,
            ['fp32_ts', 'fp32_eager', 'int8', 'pruned30', 'pruned50', 'pruned70',
             'pruned30_bnrecal', 'pruned50_bnrecal'],
        )

    def test_checkpoint_mapping(self):
        self.assertIsNone(checkpoint_for('resnet18', 'fp32_eager'))
        self.assertEqual(checkpoint_for('resnet18', 'fp32_ts'), 'resnet18_fp32_ts.pt')
        self.assertEqual(checkpoint_for('resnet18', 'int8'), 'resnet18_int8.pt')
        self.assertEqual(checkpoint_for('efficientnet_b0', 'int8'), 'efficientnet_b0_int8.pt')
        self.assertEqual(checkpoint_for('resnet18', 'pruned50'), 'resnet18_pruned50.pt')
        self.assertEqual(checkpoint_for('resnet18', 'pruned50_bnrecal'), 'resnet18_pruned50_bnrecal.pt')

    def test_session_execution_with_rerun(self):
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session1'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            executed_conditions = []

            def mock_runner(arch, condition, out_dir, checkpoints_dir):
                label = f"{arch}_{condition}"
                executed_conditions.append(label)
                # Intentionally crash one condition on first attempt
                if label == 'resnet18_pruned30' and executed_conditions.count(label) == 1:
                    raise RuntimeError("Simulated transient failure")
                # Write a mock summary.json
                (out_dir / 'summary.json').write_text(json.dumps({'gross_j_per_image_mean': 0.05}))
                return {'thermal_flagged': False}

            now_sec = 10000.0
            boottime_sec = 10000.0 - 600.0  # 10 min uptime (< 30 min)

            log = run_m1_session(
                session_n=1,
                architectures=self.architectures,
                out_base=out_base,
                checkpoints_dir=ckpt_dir,
                condition_runner=mock_runner,
                get_boottime_fn=lambda: boottime_sec,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
                get_vm_fn=lambda: {'free_mb': 500.0},
                current_time_fn=lambda: now_sec,
            )

            self.assertEqual(log['session'], 1)
            self.assertEqual(log['seed'], 4001)
            self.assertEqual(log['total_conditions'], 23)
            # 23 successful conditions in log
            self.assertEqual(len(log['conditions']), 23)
            self.assertIn('resnet18_pruned30_rerun', log['conditions'])

            # Verify void log
            void_log = json.loads((out_base / 'void_log.json').read_text())
            self.assertEqual(len(void_log['voids']), 1)
            self.assertEqual(void_log['voids'][0]['condition'], 'resnet18_pruned30')

    def test_session_aborts_on_excessive_voids(self):
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session1'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            fail_count = 0

            def failing_runner(arch, condition, out_dir, checkpoints_dir):
                nonlocal fail_count
                fail_count += 1
                raise RuntimeError("Permanent failure")

            with self.assertRaises(RuntimeError) as ctx:
                run_m1_session(
                    session_n=1,
                    architectures=self.architectures,
                    out_base=out_base,
                    checkpoints_dir=ckpt_dir,
                    condition_runner=failing_runner,
                    get_boottime_fn=lambda: 1000.0,
                    check_ac_fn=lambda: True,
                    check_therm_fn=lambda: True,
                    current_time_fn=lambda: 1200.0,
                )
            self.assertIn("more than 2 voided conditions", str(ctx.exception))

    def test_crash_records_error_text_in_void_log(self):
        """A3: a crashed condition's exception text/traceback must be recorded
        in void_log.json, not just a crashed=True flag."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session1'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            executed = []

            def mock_runner(arch, condition, out_dir, checkpoints_dir):
                label = f"{arch}_{condition}"
                executed.append(label)
                if label == 'resnet18_pruned30' and executed.count(label) == 1:
                    raise RuntimeError("Simulated transient failure for A3 test")
                (out_dir / 'summary.json').write_text(json.dumps({'gross_j_per_image_mean': 0.05}))
                return {'thermal_flagged': False}

            run_m1_session(
                session_n=1,
                architectures=self.architectures,
                out_base=out_base,
                checkpoints_dir=ckpt_dir,
                condition_runner=mock_runner,
                get_boottime_fn=lambda: 10000.0 - 600.0,
                check_ac_fn=lambda: True,
                check_therm_fn=lambda: True,
                current_time_fn=lambda: 10000.0,
            )

            void_log = json.loads((out_base / 'void_log.json').read_text())
            self.assertEqual(len(void_log['voids']), 1)
            error_text = void_log['voids'][0]['error_text']
            self.assertIsNotNone(error_text)
            self.assertIn('RuntimeError', error_text)
            self.assertIn('Simulated transient failure for A3 test', error_text)
            self.assertIn('Traceback', error_text)

    def test_guard_error_pre_session_writes_session_abort(self):
        """A3: a GuardError raised at pre-session checks writes session_abort.json
        (reason, condition=None, uptime, time) before re-raising."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session1'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            def failing_ac_check():
                raise GuardError("Synthetic: AC power is not connected")

            with self.assertRaises(GuardError):
                run_m1_session(
                    session_n=1,
                    architectures=self.architectures,
                    out_base=out_base,
                    checkpoints_dir=ckpt_dir,
                    condition_runner=lambda **kw: None,
                    get_boottime_fn=lambda: 1000.0,
                    check_ac_fn=failing_ac_check,
                    check_therm_fn=lambda: True,
                    current_time_fn=lambda: 1200.0,
                )

            abort = json.loads((out_base / 'session_abort.json').read_text())
            self.assertEqual(abort['session'], 1)
            self.assertIn("AC power is not connected", abort['reason'])
            self.assertIsNone(abort['condition'])
            self.assertEqual(abort['time'], 1200.0)

    def test_guard_error_pre_condition_writes_session_abort(self):
        """A3: a GuardError raised at a pre-condition check (not pre-session)
        also writes session_abort.json, with the failing condition's label."""
        with TemporaryDirectory() as tmpdir:
            out_base = Path(tmpdir) / 'session1'
            ckpt_dir = Path(__file__).resolve().parent.parent.parent / 'checkpoints'

            call_count = {'ac': 0}

            def flaky_ac_check():
                call_count['ac'] += 1
                # Pass the pre-session check (call 1), fail on the first
                # pre-condition check (call 2).
                if call_count['ac'] >= 2:
                    raise GuardError("Synthetic: AC power lost before condition start")
                return True

            def mock_runner(arch, condition, out_dir, checkpoints_dir):
                (out_dir / 'summary.json').write_text(json.dumps({'gross_j_per_image_mean': 0.05}))
                return {'thermal_flagged': False}

            with self.assertRaises(GuardError):
                run_m1_session(
                    session_n=1,
                    architectures=self.architectures,
                    out_base=out_base,
                    checkpoints_dir=ckpt_dir,
                    condition_runner=mock_runner,
                    get_boottime_fn=lambda: 1000.0,
                    check_ac_fn=flaky_ac_check,
                    check_therm_fn=lambda: True,
                    current_time_fn=lambda: 1200.0,
                )

            abort = json.loads((out_base / 'session_abort.json').read_text())
            self.assertEqual(abort['session'], 1)
            self.assertIn("AC power lost before condition start", abort['reason'])
            self.assertIsNotNone(abort['condition'])


if __name__ == '__main__':
    unittest.main()
