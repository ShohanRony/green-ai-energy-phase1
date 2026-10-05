import subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import power_state
from power_watchdog import classify_transition, restore_governor

class MarkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self._orig=(power_state.STATE_DIR,power_state.STOP_MARKER,power_state.RESTORED_MARKER,power_state.RESUME_HINT)
        d=Path(self.tmp.name)/'.power_state'
        power_state.STATE_DIR,power_state.STOP_MARKER,power_state.RESTORED_MARKER,power_state.RESUME_HINT = \
            d,d/'stop',d/'restored',d/'resume_hint.txt'
    def tearDown(self):
        power_state.STATE_DIR,power_state.STOP_MARKER,power_state.RESTORED_MARKER,power_state.RESUME_HINT=self._orig
        self.tmp.cleanup()

    def test_stop_requested_false_until_marker_written(self):
        self.assertFalse(power_state.stop_requested())
        power_state.STATE_DIR.mkdir()
        power_state.STOP_MARKER.write_text('2026-10-05T00:00:00Z')
        self.assertTrue(power_state.stop_requested())

    def test_resume_hint_write_and_clear(self):
        power_state.write_resume_hint('python3 train_baseline.py --resume x.pt')
        self.assertEqual(power_state.RESUME_HINT.read_text().strip(),'python3 train_baseline.py --resume x.pt')
        power_state.clear_resume_hint()
        self.assertFalse(power_state.RESUME_HINT.exists())
        power_state.clear_resume_hint()  # idempotent, no error on a second clear

class TransitionClassifierTests(unittest.TestCase):
    def test_no_transition(self):
        self.assertIsNone(classify_transition(True,True))
        self.assertIsNone(classify_transition(False,False))
    def test_ac_lost(self):
        self.assertEqual(classify_transition(True,False),'lost')
    def test_ac_restored(self):
        self.assertEqual(classify_transition(False,True),'restored')

class RestoreGovernorTests(unittest.TestCase):
    def test_missing_sudoers_rule_fails_without_blocking(self):
        # sudo -n (non-interactive) exits nonzero rather than prompting when the
        # rule isn't installed -- confirm that's treated as a clean failure, not a hang/crash.
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1,'sudo')):
            self.assertFalse(restore_governor())
    def test_success_path(self):
        with patch('subprocess.run') as m:
            m.return_value = subprocess.CompletedProcess(args=[], returncode=0)
            self.assertTrue(restore_governor())

if __name__=='__main__':
    unittest.main()
