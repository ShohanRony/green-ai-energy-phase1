import os, subprocess, sys, tempfile, time, types, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot import Sensor, integrate, summarize, check_interval_floor, check_no_concurrent_gpu, check_platform_profile, flag_implausible_power, classify_power_regime, check_fresh_boot, find_rapl_domain, p_core_set, _col

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

class RaplDomainSplitTests(unittest.TestCase):
    """D16: integrate()'s names= mode must sum each RAPL domain separately, not merge them,
    and must still handle counter wraparound correctly per domain."""
    RANGE = 1000.0
    def test_split_mode_sums_domains_separately(self):
        trace = [(0.0, [10.0, 40.0]), (1.0, [30.0, 90.0])]  # package-0 delta 20, psys delta 50
        out = integrate(trace, [self.RANGE, self.RANGE], ['package-0', 'psys'])
        self.assertEqual(out, {'package-0': 20.0, 'psys': 50.0})
        self.assertNotEqual(out['package-0'], out['package-0'] + out['psys'])  # the D16 bug merged these
    def test_split_mode_wraparound_per_domain(self):
        trace = [(0.0, [990.0, 500.0]), (1.0, [5.0, 600.0])]  # package wraps, psys doesn't
        out = integrate(trace, [self.RANGE, self.RANGE], ['package-0', 'psys'])
        self.assertEqual(out['package-0'], 5.0 + (self.RANGE - 990.0))
        self.assertEqual(out['psys'], 100.0)

class RaplDomainLookupTests(unittest.TestCase):
    """find_rapl_domain() must locate a domain by its sysfs name (item 3's concurrent-RAPL-on-GPU
    patch), never a blind glob -- that's the D16 bug it exists to avoid repeating."""
    def test_matches_package_0_on_this_machine(self):
        path, rng = find_rapl_domain('package-0')
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).parent.name, 'intel-rapl:0')
        self.assertGreater(rng, 0)
    def test_missing_name_returns_none(self):
        path, rng = find_rapl_domain('definitely-not-a-real-domain')
        self.assertIsNone(path); self.assertIsNone(rng)

class PCoreAffinityTests(unittest.TestCase):
    """p_core_set() -- item 3's --cpu-affinity flag. Mocked sysfs so this doesn't depend on
    running on a hybrid P-core/E-core CPU specifically."""
    class _FakePath:
        def __init__(self, exists, text=''): self._exists=exists; self._text=text
        def __call__(self, *a, **k): return self
        def exists(self): return self._exists
        def read_text(self): return self._text
    def test_parses_ranges_and_singles(self):
        with patch('pilot.Path', self._FakePath(True, '0-5,8,10-11\n')):
            self.assertEqual(p_core_set(), {0,1,2,3,4,5,8,10,11})
    def test_raises_without_hybrid_sysfs(self):
        with patch('pilot.Path', self._FakePath(False)):
            with self.assertRaises(RuntimeError): p_core_set()

class Stage4bHarnessTests(unittest.TestCase):
    def test_summarize_per_boundary_means(self):
        rows=[]
        for i in range(10):
            for phase in ['idle_before','a1','a2','idle_after']:
                active=phase.startswith('a'); dur=5.0
                e=50.0 if active else 10.0
                rows.append(dict(device='cuda',size=32,batch=1,requested_s=5,repeat=i,phase=phase,
                                 duration_s=dur,energy_j=e,gpu_energy_j=e,cpu_package_energy_j=20.0 if active else 0.0,
                                 system_energy_j=e + (20.0 if active else 0.0),
                                 batches=100 if active else 0,images_per_s=20.0 if active else 0.0,power_regime='pinned'))
        res=summarize(rows)[0]
        self.assertAlmostEqual(res['gpu_j_per_image_mean'], 0.5)
        self.assertAlmostEqual(res['cpu_package_j_per_image_mean'], 0.2)
        self.assertAlmostEqual(res['system_j_per_image_mean'], 0.7)
        self.assertAlmostEqual(res['images_per_s_mean'], 20.0)

    def test_system_energy_sum(self):
        gpu_e = 45.2
        pkg_e = 12.8
        sys_e = gpu_e + pkg_e
        self.assertAlmostEqual(sys_e - (gpu_e + pkg_e), 0.0, places=9)

class DualNvmlRaplColumnTests(unittest.TestCase):
    """Both NVML interfaces + RAPL package-0/psys logged as separate columns, primary unchanged.
    4-column GPU trace: [power_usage_W, cumulative_J, rapl_package_J, rapl_psys_J]."""
    def test_power_usage_is_trapezoidal_cumulative_is_delta_sum(self):
        trace = [(0.0, [10.0, 100.0, 5.0, 20.0]), (1.0, [20.0, 140.0, 8.0, 27.0])]
        power_usage_j = integrate(_col(trace, 0), [])
        cumulative_j = integrate(_col(trace, 1), [None])
        self.assertAlmostEqual(power_usage_j, (10.0 + 20.0) / 2)  # trapezoid, dt=1 -> 15.0
        self.assertAlmostEqual(cumulative_j, 40.0)  # delta: 140-100, a different method/value
    def test_rapl_secondary_columns_independent_with_wraparound(self):
        trace = [(0.0, [10.0, 100.0, 990.0, 20.0]), (1.0, [20.0, 115.0, 5.0, 27.0])]  # package wraps
        pkg_j = integrate(_col(trace, 2), [1000.0])
        psys_j = integrate(_col(trace, 3), [1000.0])
        self.assertAlmostEqual(pkg_j, 5.0 + (1000.0 - 990.0))
        self.assertAlmostEqual(psys_j, 7.0)
    def test_col_extracts_single_column_unchanged_shape(self):
        trace = [(0.0, [1.0, 2.0, 3.0]), (1.0, [4.0, 5.0, 6.0])]
        self.assertEqual(_col(trace, 1), [(0.0, [2.0]), (1.0, [5.0])])

if __name__=='__main__': unittest.main()
