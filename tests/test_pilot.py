import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import integrate, summarize, check_interval_floor

class MathTests(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(integrate([(0,[9]),(1,[1]),(2,[4])],[10]),5)
    def test_power_integral(self):
        self.assertAlmostEqual(integrate([(0,[10]),(1,[20]),(3,[20])],[]),55)
    def test_reset_rejected(self):
        with self.assertRaises(RuntimeError): integrate([(0,[2]),(1,[1])],[None])
    def test_idle_duration_and_detection(self):
        rows=[]
        for i in range(31): # 31 so dropping the discarded cold rep 0 leaves a balanced 15/15 parity split
            for phase in ['idle_before','a1','a2','idle_after']:
                active=phase.startswith('a'); duration=5.2 if active else 5
                energy=duration*(20 if active else 10)
                if phase=='a2': energy += (-1)**i*.1
                rows.append(dict(device='cpu',size=32,batch=1,requested_s=5,repeat=i,phase=phase,
                                 duration_s=duration,energy_j=energy,batches=100 if active else 0))
        result=summarize(rows)[0]
        self.assertEqual(result['idle_j_mean'],50)
        self.assertAlmostEqual(result['above_idle_j_mean'],52)
        self.assertTrue(result['candidate_for_confirmation'])
        self.assertAlmostEqual(rows[1]['above_idle_fraction'],.5)
        self.assertEqual(summarize(rows[:4]),[])

class IntervalFloorTests(unittest.TestCase):
    def test_rapl_ceiling_default_rejects(self):
        with self.assertRaises(ValueError): check_interval_floor(True,0.005,0.01,False,'--x','reason')
    def test_rapl_ceiling_override_allows(self):
        check_interval_floor(True,0.005,0.01,True,'--x','reason') # must not raise
    def test_rapl_ceiling_skipped_on_other_device(self):
        check_interval_floor(False,0.005,0.01,False,'--x','reason') # device doesn't match: no-op
    def test_nvml_floor_default_rejects(self):
        with self.assertRaises(ValueError): check_interval_floor(True,0.02,0.3,False,'--y','reason')
    def test_nvml_floor_override_allows(self):
        check_interval_floor(True,0.02,0.3,True,'--y','reason') # must not raise

if __name__=='__main__': unittest.main()
