import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import integrate, summarize

class MathTests(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(integrate([(0,[9]),(1,[1]),(2,[4])],[10]),5)
    def test_power_integral(self):
        self.assertAlmostEqual(integrate([(0,[10]),(1,[20]),(3,[20])],[]),55)
    def test_reset_rejected(self):
        with self.assertRaises(RuntimeError): integrate([(0,[2]),(1,[1])],[None])
    def test_idle_duration_and_detection(self):
        rows=[]
        for i in range(30):
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

if __name__=='__main__': unittest.main()
