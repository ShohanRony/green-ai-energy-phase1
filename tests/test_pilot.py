import os, subprocess, sys, tempfile, time, types, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import Sensor, integrate, summarize, check_interval_floor, check_no_concurrent_gpu, check_platform_profile, flag_implausible_power, classify_power_regime, check_fresh_boot

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
                                 duration_s=duration,energy_j=energy,batches=100 if active else 0,power_regime=None))
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

class FakeSensor:
    def __init__(self, device, pids):
        self.device=device
        self.handle=None
        self.nv=types.SimpleNamespace(
            nvmlDeviceGetComputeRunningProcesses=lambda h: [types.SimpleNamespace(pid=p) for p in pids])

class ConcurrentGpuGuardTests(unittest.TestCase):
    def test_other_process_blocks_by_default(self):
        sensor=FakeSensor('cuda',[os.getpid(),99999])
        with self.assertRaises(RuntimeError): check_no_concurrent_gpu(sensor,allow=False)
    def test_override_allows_with_warning(self):
        sensor=FakeSensor('cuda',[os.getpid(),99999])
        check_no_concurrent_gpu(sensor,allow=True) # must not raise
    def test_only_our_own_pid_is_fine(self):
        sensor=FakeSensor('cuda',[os.getpid()])
        check_no_concurrent_gpu(sensor,allow=False) # must not raise
    def test_cpu_device_is_noop(self):
        sensor=FakeSensor('cpu',[12345])
        check_no_concurrent_gpu(sensor,allow=False) # RAPL/CPU path unaffected, must not raise

class RealConcurrentGpuGuardTest(unittest.TestCase):
    """End-to-end: a real subprocess holding the GPU, checked via a real Sensor/pynvml call.
    Skips if no CUDA GPU is present rather than failing the whole suite on CPU-only machines."""
    @classmethod
    def setUpClass(cls):
        try:
            cls.sensor=Sensor('cuda')
        except Exception as e:
            raise unittest.SkipTest(f'No CUDA GPU available: {e}')
        cls.holder=subprocess.Popen([sys.executable,'-c',
            'import torch,time\nx=torch.zeros(1000,1000).cuda()\n'
            'while True:\n x=x+1\n time.sleep(0.5)'])
        for _ in range(20): # up to ~10s for it to register a CUDA context
            if any(p.pid==cls.holder.pid for p in cls.sensor.nv.nvmlDeviceGetComputeRunningProcesses(cls.sensor.handle)):
                break
            time.sleep(0.5)
        else:
            cls.holder.kill(); raise unittest.SkipTest('Background GPU holder never registered with NVML')
    @classmethod
    def tearDownClass(cls):
        cls.holder.kill(); cls.holder.wait()
    def test_blocks_by_default_and_names_the_pid(self):
        with self.assertRaises(RuntimeError) as ctx: check_no_concurrent_gpu(self.sensor,allow=False)
        self.assertIn(str(self.holder.pid),str(ctx.exception))
    def test_override_allows_with_warning(self):
        check_no_concurrent_gpu(self.sensor,allow=True) # must not raise

class PlausibilityFlagTests(unittest.TestCase):
    def test_above_cap_is_flagged(self):
        self.assertTrue(flag_implausible_power('cuda', 85.14, 60.0)) # this GPU's real enforced cap, confirmed 2026-10-03
    def test_at_or_below_cap_is_not_flagged(self):
        self.assertFalse(flag_implausible_power('cuda', 26.44, 60.0))
        self.assertFalse(flag_implausible_power('cuda', 60.0, 60.0)) # exactly at the cap: not a violation
    def test_cpu_device_is_never_flagged(self):
        self.assertFalse(flag_implausible_power('cpu', 999.0, None)) # no GPU cap concept on the RAPL path

class PowerRegimeClassifierTests(unittest.TestCase):
    def test_pinned_at_or_above_90pct_cap(self):
        self.assertEqual(classify_power_regime('cuda', 59.9, 60.0), 'pinned')
        self.assertEqual(classify_power_regime('cuda', 54.0, 60.0), 'pinned') # exactly 90%
    def test_dip_below_90pct_cap(self):
        self.assertEqual(classify_power_regime('cuda', 53.1, 60.0), 'dip') # observed dip ceiling
        self.assertEqual(classify_power_regime('cuda', 35.0, 60.0), 'dip')
    def test_cpu_device_has_no_regime(self):
        self.assertIsNone(classify_power_regime('cpu', 999.0, None))

class FreshBootGuardTests(unittest.TestCase):
    def test_within_threshold_passes(self):
        check_fresh_boot(max_uptime_s=1e12, override=False, flag_name='--allow-stale-boot')
    def test_over_threshold_rejected_by_default(self):
        with self.assertRaises(RuntimeError): check_fresh_boot(max_uptime_s=0, override=False, flag_name='--allow-stale-boot')
    def test_over_threshold_override_allows(self):
        check_fresh_boot(max_uptime_s=0, override=True, flag_name='--allow-stale-boot') # just must not raise

class PlatformProfileGuardTests(unittest.TestCase):
    def _path(self, value):
        f=tempfile.NamedTemporaryFile(mode='w',delete=False,suffix='.profile')
        f.write(value); f.close()
        self.addCleanup(os.unlink, f.name)
        return Path(f.name)
    def test_performance_passes(self):
        self.assertEqual(check_platform_profile(self._path('performance')),'performance')
    def test_other_profile_rejected(self):
        with self.assertRaises(RuntimeError): check_platform_profile(self._path('balanced'))
    def test_missing_path_rejected(self):
        with self.assertRaises(RuntimeError): check_platform_profile(Path('/nonexistent/platform_profile'))

if __name__=='__main__': unittest.main()
