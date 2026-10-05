"""Local Linux energy pilot. No generated measurements or silent sensor fallback.

Power-loss resilience: checks power_state.stop_requested() between reps (never mid-rep).
If set, records how many reps are done in resume_state.json and exits cleanly. Resume with
--resume <out dir> -- all other flags (device, checkpoint, sizes, etc.) are restored from
that run's own environment.json, not re-specified.
"""
import argparse, csv, glob, json, math, os, platform, random, statistics, sys, threading, time, traceback
from pathlib import Path

import power_state

class Sensor:
    def __init__(self, device, legacy_cumulative=False):
        self.device = device
        if device == 'cpu':
            paths = glob.glob('/sys/class/powercap/intel-rapl:*/energy_uj')
            self.paths = [Path(p) for p in paths if Path(p).parent.name.count(':') == 1]
            if not self.paths: raise RuntimeError('No accessible Intel RAPL package counters; native Linux required.')
            self.ranges = [float(p.with_name('max_energy_range_uj').read_text()) / 1e6 for p in self.paths]
            self.backend = 'RAPL package energy counters'; self.power_cap_w = None
        else:
            import pynvml as nv
            self.nv = nv; nv.nvmlInit(); self.handle = nv.nvmlDeviceGetHandleByIndex(0)
            self.power_cap_w = nv.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000
            # Default is nvmlDeviceGetPowerUsage (sampled power, trapezoidal-integrated): validated
            # 2026-10-03 against live telemetry to within -0.02W (results/active_power_baseline_investigation.md).
            # nvmlDeviceGetTotalEnergyConsumption (cumulative counter) over-reports active-phase power
            # by ~30% on this GPU -- kept only behind --legacy-cumulative-counter for reproducing old numbers.
            if legacy_cumulative:
                try:
                    nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                    self.backend = 'NVML cumulative energy (legacy, --legacy-cumulative-counter)'; self.ranges = [None]
                except nv.NVMLError_NotSupported:
                    self.backend = 'NVML sampled power'; self.ranges = []
            else:
                self.backend = 'NVML sampled power'; self.ranges = []
    def read(self):
        if self.device == 'cpu': return [float(p.read_text()) / 1e6 for p in self.paths]
        if self.ranges: return [self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000]
        return [self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000]

def check_interval_floor(device_matches, interval, floor, override, flag_name, reason):
    """Reject (or warn-and-allow, behind an override flag) a polling interval below a measured floor."""
    if not device_matches or interval>=floor: return
    if not override: raise ValueError(f'{reason} Pass {flag_name} to force it anyway.')
    print(f'WARNING: {reason}', file=sys.stderr)

def check_no_concurrent_gpu(sensor, allow=False):
    """Checklist item 13: abort (or warn-and-allow) if another process besides us is using the GPU."""
    if sensor.device != 'cuda': return
    others=[proc.pid for proc in sensor.nv.nvmlDeviceGetComputeRunningProcesses(sensor.handle) if proc.pid != os.getpid()]
    if not others: return
    msg=(f'Other process(es) using the GPU: {others} — measurement would be contaminated by '
         f'concurrent GPU work (checklist item 13).')
    if not allow: raise RuntimeError(msg+' Pass --allow-concurrent-gpu to proceed anyway.')
    print(f'WARNING: {msg}', file=sys.stderr)

def check_platform_profile(path=Path('/sys/firmware/acpi/platform_profile')):
    """Checklist item 14: abort if the ACPI platform power profile isn't locked to performance."""
    if not path.exists():
        raise RuntimeError(f'{path} not found; cannot verify platform power profile is locked to performance.')
    value = path.read_text().strip()
    if value != 'performance':
        raise RuntimeError(f"Platform power profile is '{value}', not 'performance' — this is a controlled "
                            f'variable for the whole project. Set it with: echo performance | sudo tee {path}')
    return value

def flag_implausible_power(device, implied_w, power_cap_w):
    """Mark (never drop) a window whose implied power exceeds the GPU's own enforced
    hardware power cap -- a physically impossible reading, not just noise. Verified
    2026-10-03 via both `nvidia-smi -q -d POWER` and nvmlDeviceGetEnforcedPowerLimit:
    this RTX 3050's current enforced limit is 60.00W (default; max settable 95.00W)."""
    return device == 'cuda' and implied_w > power_cap_w

def classify_power_regime(device, implied_w, power_cap_w):
    """Stage 4 pre-flight (2026-10-05, results_stage4_preflight/stage4_preflight_findings.md)
    found this GPU lands in one of two distinct power states for the same config: pinned at
    its enforced cap, or an unsaturated ~55-90% of it -- never observed in between. Flag which
    one every run landed in instead of silently averaging across both; a 'dip' run is not an
    error, but a 'mixed' run (see summarize()) means reps disagreed and needs attention.
    0.9 threshold: clean margin against observed clusters (pinned >=98.8% of cap, dip <=88.5%)."""
    if device != 'cuda' or power_cap_w is None: return None
    return 'pinned' if implied_w >= 0.9 * power_cap_w else 'dip'

def check_fresh_boot(max_uptime_s, override, flag_name):
    """Stage 4 pre-flight's leading (not confirmed) hypothesis for the pin/dip split is uptime-
    since-boot, not per-invocation randomness: every dip seen so far was in a long-uptime
    session; two separate fresh reboots both measured pinned. This guard is a conservative,
    disclosed placeholder -- not a precisely measured decay boundary -- requiring a recent
    reboot before a run starts, so Stage 4's matrix doesn't silently mix regimes across a long
    session. Raises (or warns-and-allows, behind the override) if uptime exceeds the threshold."""
    uptime_s = float(Path('/proc/uptime').read_text().split()[0])
    if uptime_s <= max_uptime_s: return uptime_s
    reason = (f'System uptime is {uptime_s/60:.0f} min, over the {max_uptime_s/60:.0f}-min fresh-boot '
              f'threshold (stage4_preflight_findings.md: every observed power-regime dip happened in a '
              f'long-uptime session; reboot before starting a real matrix run).')
    if not override: raise RuntimeError(f'{reason} Pass {flag_name} to force it anyway.')
    print(f'WARNING: {reason}', file=sys.stderr)
    return uptime_s

def integrate(trace, ranges):
    energy = 0.
    for (t0, x0), (t1, x1) in zip(trace, trace[1:]):
        if not ranges: energy += (t1-t0) * (x0[0]+x1[0])/2
        else:
            for a,b,limit in zip(x0,x1,ranges):
                d = b-a
                if d < 0:
                    if limit is None: raise RuntimeError('Energy counter reset')
                    d += limit
                energy += d
    return energy

def window(sensor, seconds, interval, work=None, sync=lambda: None, allow_concurrent_gpu=False):
    check_no_concurrent_gpu(sensor, allow_concurrent_gpu)
    sync(); trace=[]; errors=[]; stop=threading.Event()
    def sample():
        try:
            start=time.perf_counter(); value=sensor.read(); end=time.perf_counter()
            trace.append(((start+end)/2,value))
        except Exception as e: errors.append(str(e)); stop.set()
    sample()
    def poll():
        while not stop.wait(interval): sample()
    thread=threading.Thread(target=poll); thread.start()
    batches=0; start=time.perf_counter()
    try:
        if work is None: time.sleep(seconds)
        else:
            while time.perf_counter()-start < seconds:
                work(); sync(); batches+=1
    finally:
        sync(); stop.set(); thread.join(); sample()
    if errors: raise RuntimeError('; '.join(errors))
    duration=trace[-1][0]-trace[0][0]
    energy=integrate(trace,sensor.ranges)
    if energy <= 0: raise RuntimeError('Nonpositive energy: stale or unavailable sensor; reject window')
    return dict(energy_j=energy,duration_s=duration,batches=batches,
                max_sample_gap_s=max(b[0]-a[0] for a,b in zip(trace,trace[1:])),
                changed_reads=sum(a[1]!=b[1] for a,b in zip(trace,trace[1:])),
                trace=trace,backend=sensor.backend,
                plausibility_flag=flag_implausible_power(sensor.device, energy/duration, sensor.power_cap_w),
                power_regime=classify_power_regime(sensor.device, energy/duration, sensor.power_cap_w))

def summarize(rows):
    # Independently paired A/A differences estimate the noise relevant to a future A/B comparison.
    out=[]
    keys=sorted({(r['device'],r['size'],r['batch'],r['requested_s']) for r in rows})
    for key in keys:
        group=[r for r in rows if (r['device'],r['size'],r['batch'],r['requested_s'])==key]
        ids=sorted({r['repeat'] for r in group}); pairs=[]; active=[]; idle=[]; net=[]; fractions=[]; diffs=[]; regimes=[]
        for rep in ids:
            d={r['phase']:r for r in group if r['repeat']==rep}
            if set(d) != {'idle_before','a1','a2','idle_after'}: continue
            # First run of each configuration is cold-cache/cold-thermal: discard from the summary
            # (checklist item 3), but still annotate it below so raw/windows logs retain it (item 12).
            cold=rep==ids[0] and len(ids)>1
            idlepower=statistics.mean(d[p]['energy_j']/d[p]['duration_s'] for p in ['idle_before','idle_after'])
            per=[]
            for p in ['a1','a2']:
                r=d[p]; gross=r['energy_j']; above=gross-idlepower*r['duration_s']
                per.append(gross/(r['batches']*key[2]))
                r['estimated_idle_j']=idlepower*r['duration_s']; r['above_idle_j']=above
                r['above_idle_fraction']=above/gross if gross else None
                r['gross_j_per_image']=per[-1]
                if not cold:
                    active.append(gross); net.append(above); fractions.append(above/gross if gross else float('nan'))
                    if r['power_regime'] is not None: regimes.append(r['power_regime'])
            if cold: continue
            # Match both idle readings to the planned window to avoid timing overshoot bias.
            idle.extend(d[p]['energy_j']/d[p]['duration_s']*key[3] for p in ['idle_before','idle_after'])
            pairs.append(statistics.mean(per)); diffs.append(per[1]-per[0])
        if len(pairs)<3: continue
        n=len(pairs); mean=statistics.mean(pairs); sd=statistics.stdev(diffs)
        mde=2.80*sd/math.sqrt(n) # normal approximation: two-sided alpha .05, power .80
        bias=abs(statistics.mean(diffs)); limit=bias+mde
        regime_set=set(regimes)
        out.append(dict(device=key[0],size=key[1],batch=key[2],window_s=key[3],pairs=n,
                        idle_j_mean=statistics.mean(idle),idle_j_sd=statistics.stdev(idle),
                        total_j_mean=statistics.mean(active),total_j_sd=statistics.stdev(active),
                        above_idle_j_mean=statistics.mean(net),above_idle_fraction_mean=statistics.mean(fractions),
                        gross_j_per_image_mean=mean,paired_difference_sd=sd,
                        paired_order_bias=statistics.mean(diffs),approx_mde_j_per_image=mde,
                        conservative_screen_fraction=limit/mean if mean>0 else None,
                        candidate_for_confirmation=bool(n>=30 and mean>0 and limit<=.05*mean),
                        power_regime=(regime_set.pop() if len(regime_set)==1 else ('mixed' if regime_set else None))))
    return out

def csv_write(path, rows):
    if not rows: return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resume',help='Path to an existing --out directory from an interrupted run. '
                                    'All other flags are restored from its environment.json, not re-specified.')
    p.add_argument('--device',choices=['cpu','cuda'])
    p.add_argument('--arch',choices=['resnet18','mobilenet_v3_small','efficientnet_b0'],default='resnet18',
                    help='Architecture to build when --checkpoint is a plain state_dict, or when no '
                         '--checkpoint is given (random-weights timing-only mode). Ignored when '
                         '--checkpoint is a self-contained TorchScript module (FP16/pruned/INT8 '
                         'checkpoints from Stage 2 are saved this way and carry their own architecture).')
    p.add_argument('--data',default='data'); p.add_argument('--download',action='store_true')
    p.add_argument('--out'); p.add_argument('--sizes',type=int,nargs='+',default=[32,64,128,224])
    p.add_argument('--batches',type=int,nargs='+',default=[1,16,64,128,256])
    p.add_argument('--windows',type=float,nargs='+',default=[5])
    p.add_argument('--repeats',type=int,default=6); p.add_argument('--interval',type=float,default=.4)
    p.add_argument('--override-fast-interval',action='store_true',
                    help='Allow --interval below the NVML counter-telescoping floor (0.3s). Logs a warning.')
    p.add_argument('--override-fast-rapl-interval',action='store_true',
                    help='Allow --interval faster than the RAPL/perf-events ceiling (100Hz / 0.01s). Logs a warning.')
    p.add_argument('--allow-concurrent-gpu',action='store_true',
                    help='Allow measurement to proceed even if another process is using the GPU. Logs a warning.')
    p.add_argument('--legacy-cumulative-counter',action='store_true',
                    help='Use nvmlDeviceGetTotalEnergyConsumption instead of the validated default '
                         'nvmlDeviceGetPowerUsage. Confirmed 2026-10-03 to over-report active-phase power '
                         'by ~30%% on this GPU (results/active_power_baseline_investigation.md). Only for '
                         'reproducing/citing pre-correction numbers. Logs a warning.')
    p.add_argument('--max-uptime-min',type=float,default=30,
                    help='Refuse to start if system uptime exceeds this (minutes) -- Stage 4 pre-flight '
                         'found every observed power-regime dip in a long-uptime session. Conservative, '
                         'disclosed placeholder, not a precisely measured boundary.')
    p.add_argument('--allow-stale-boot',action='store_true',
                    help='Allow a run past --max-uptime-min. Logs a warning.')
    p.add_argument('--threads',type=int,default=4); p.add_argument('--warmup',type=float,default=3)
    p.add_argument('--codecarbon',action='store_true',help='Run CodeCarbon concurrently with the RAPL/NVML reads on each active window.')
    p.add_argument('--codecarbon-country',default='CZE',help='ISO code for CodeCarbon offline grid-intensity lookup (energy figure itself is country-independent).')
    p.add_argument('--checkpoint'); a=p.parse_args()
    start_rep=0; rows=[]
    if a.resume:
        out=Path(a.resume)
        saved_args=json.loads((out/'environment.json').read_text())['arguments']
        resume_val=a.resume; a=argparse.Namespace(**saved_args); a.resume=resume_val  # restore everything except --resume itself
        with (out/'raw.jsonl').open() as f:
            rows=[json.loads(line) for line in f]
        resume_state=json.loads((out/'resume_state.json').read_text())
        start_rep=resume_state['next_rep']
        print(f'Resuming {out} from rep {start_rep}/{a.repeats}',file=sys.stderr)
    else:
        if not a.device or not a.out: p.error('--device and --out are required unless --resume is given')
        out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    env=dict(platform=platform.platform(),cpu=platform.processor(),arguments=vars(a),
             timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else '')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    try:
        if min(a.windows)<=0 or min(a.batches)<=0 or a.interval<=0: raise ValueError('Positive durations/batches required')
        check_interval_floor(a.device=='cuda', a.interval, 0.3, a.override_fast_interval, '--override-fast-interval',
            f'--interval {a.interval}s is below the measured NVML counter-telescoping floor (0.3-0.5s on '
            f'this RTX 3050; 20ms implies ~300W against a 60W cap).')
        check_interval_floor(a.device=='cpu', a.interval, 0.01, a.override_fast_rapl_interval, '--override-fast-rapl-interval',
            f'--interval {a.interval}s exceeds the RAPL/perf-events sampling ceiling (100Hz / 0.01s, plan checklist item 4).')
        platform_profile=check_platform_profile()
        uptime_s=check_fresh_boot(a.max_uptime_min*60, a.allow_stale_boot, '--allow-stale-boot')
        if a.device=='cuda' and a.legacy_cumulative_counter:
            print('WARNING: --legacy-cumulative-counter selected; nvmlDeviceGetTotalEnergyConsumption is '
                  'known to over-report active-phase power by ~30% on this GPU (see '
                  'results/active_power_baseline_investigation.md). Use only to reproduce/cite pre-correction numbers.',
                  file=sys.stderr)
        sensor=Sensor(a.device, legacy_cumulative=a.legacy_cumulative_counter)
        import torch, torchvision
        from torchvision import transforms
        torch.manual_seed(2026); random.seed(2026); torch.set_num_threads(a.threads)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        from train_baseline import build_model
        if a.checkpoint:
            try:
                model=torch.jit.load(a.checkpoint,map_location='cpu')  # self-contained: FP16/pruned/INT8 Stage 2 checkpoints
            except RuntimeError:
                model=build_model(a.arch)  # plain state_dict: needs the matching architecture rebuilt first
                model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
        else:
            model=build_model(a.arch)
        model=model.eval().to(a.device)
        ds=torchvision.datasets.CIFAR10(a.data,train=False,download=a.download,transform=transforms.ToTensor())
        raw=torch.stack([ds[i][0] for i in range(max(a.batches))])
        sync=torch.cuda.synchronize if a.device=='cuda' else lambda:None
        gov_paths=glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
        env.update(torch=torch.__version__,torchvision=torchvision.__version__,backend=sensor.backend,
                   gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
                   nvidia_driver_version=sensor.nv.nvmlSystemGetDriverVersion() if a.device=='cuda' else None,
                   gpu_power_cap_w=sensor.power_cap_w,
                   cpu_governors=sorted({Path(p).read_text().strip() for p in gov_paths}) if gov_paths else None,
                   platform_profile=platform_profile,uptime_s=uptime_s,
                   weights='checkpoint' if a.checkpoint else 'seeded random weights: timing pilot only')
        (out/'environment.json').write_text(json.dumps(env,indent=2))
        with torch.inference_mode():
            for rep in range(start_rep,a.repeats):
                grid=[(s,b,w) for s in a.sizes for b in a.batches for w in a.windows]; random.shuffle(grid)
                for size,batch,seconds in grid:
                    try:
                        x=torch.nn.functional.interpolate(raw[:batch],size=(size,size),mode='bilinear',align_corners=False)
                        x=(x-torch.tensor([.4914,.4822,.4465])[None,:,None,None])/torch.tensor([.247,.243,.261])[None,:,None,None]
                        try: target_dtype=next(model.parameters()).dtype  # FP16 checkpoints need the input cast to match
                        except StopIteration: target_dtype=x.dtype  # quantized models hold no plain nn.Parameter; input stays float
                        x=x.to(a.device).to(target_dtype)
                        work=lambda:model(x)
                        for phase in ['idle_before','a1','a2','idle_after']:
                            if phase.startswith('a'):
                                start=time.perf_counter()
                                while time.perf_counter()-start<a.warmup: work(); sync()
                            else: time.sleep(a.warmup) # same declared settling delay, no thermal equilibrium claim
                            cc_tracker=None
                            if a.codecarbon and phase.startswith('a'):
                                from codecarbon import OfflineEmissionsTracker
                                cc_tracker=OfflineEmissionsTracker(country_iso_code=a.codecarbon_country,
                                                                    log_level='error',save_to_file=False,
                                                                    measure_power_secs=max(a.interval,1))
                                cc_tracker.start()
                            result=window(sensor,seconds,a.interval,work if phase.startswith('a') else None,sync,a.allow_concurrent_gpu)
                            if cc_tracker is not None:
                                cc_tracker.stop()
                                result['codecarbon_energy_j']=cc_tracker.final_emissions_data.energy_consumed*3.6e6
                            row=dict(device=a.device,size=size,batch=batch,requested_s=seconds,repeat=rep,phase=phase,**result)
                            rows.append({k:v for k,v in row.items() if k!='trace'})
                            with (out/'raw.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        del x
                    except torch.cuda.OutOfMemoryError:
                        with (out/'skipped.jsonl').open('a') as f: f.write(json.dumps(dict(size=size,batch=batch,repeat=rep,reason='CUDA OOM'))+'\n')
                        torch.cuda.empty_cache()
                    finally:
                        summary=summarize(rows); csv_write(out/'summary.csv',summary); csv_write(out/'windows.csv',rows)
                (out/'resume_state.json').write_text(json.dumps(dict(next_rep=rep+1)))
                if power_state.stop_requested():
                    resume_cmd=f'python3 pilot.py --resume {out}'
                    power_state.write_resume_hint(resume_cmd)
                    print(f'Power-loss stop requested after rep {rep+1}/{a.repeats}. Resume with: {resume_cmd}',file=sys.stderr)
                    return
            power_state.clear_resume_hint()
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
