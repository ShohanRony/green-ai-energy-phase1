"""Local Linux energy pilot. No generated measurements or silent sensor fallback."""
import argparse, csv, glob, json, math, os, platform, random, statistics, sys, threading, time, traceback
from pathlib import Path

class Sensor:
    def __init__(self, device):
        self.device = device
        if device == 'cpu':
            paths = glob.glob('/sys/class/powercap/intel-rapl:*/energy_uj')
            self.paths = [Path(p) for p in paths if Path(p).parent.name.count(':') == 1]
            if not self.paths: raise RuntimeError('No accessible Intel RAPL package counters; native Linux required.')
            self.ranges = [float(p.with_name('max_energy_range_uj').read_text()) / 1e6 for p in self.paths]
            self.backend = 'RAPL package energy counters'
        else:
            import pynvml as nv
            self.nv = nv; nv.nvmlInit(); self.handle = nv.nvmlDeviceGetHandleByIndex(0)
            try:
                nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.backend = 'NVML cumulative energy'; self.ranges = [None]
            except nv.NVMLError_NotSupported:
                self.backend = 'NVML sampled power'; self.ranges = []
    def read(self):
        if self.device == 'cpu': return [float(p.read_text()) / 1e6 for p in self.paths]
        if self.ranges: return [self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000]
        return [self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000]

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

def window(sensor, seconds, interval, work=None, sync=lambda: None):
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
                trace=trace,backend=sensor.backend)

def summarize(rows):
    # Independently paired A/A differences estimate the noise relevant to a future A/B comparison.
    out=[]
    keys=sorted({(r['device'],r['size'],r['batch'],r['requested_s']) for r in rows})
    for key in keys:
        group=[r for r in rows if (r['device'],r['size'],r['batch'],r['requested_s'])==key]
        ids=sorted({r['repeat'] for r in group}); pairs=[]; active=[]; idle=[]; net=[]; fractions=[]; diffs=[]
        for rep in ids:
            d={r['phase']:r for r in group if r['repeat']==rep}
            if set(d) != {'idle_before','a1','a2','idle_after'}: continue
            idlepower=statistics.mean(d[p]['energy_j']/d[p]['duration_s'] for p in ['idle_before','idle_after'])
            # Match both idle readings to the planned window to avoid timing overshoot bias.
            idle.extend(d[p]['energy_j']/d[p]['duration_s']*key[3] for p in ['idle_before','idle_after'])
            per=[]
            for p in ['a1','a2']:
                r=d[p]; gross=r['energy_j']; above=gross-idlepower*r['duration_s']
                active.append(gross); net.append(above); fractions.append(above/gross if gross else float('nan'))
                per.append(gross/(r['batches']*key[2]))
                r['estimated_idle_j']=idlepower*r['duration_s']; r['above_idle_j']=above
                r['above_idle_fraction']=above/gross if gross else None
                r['gross_j_per_image']=per[-1]
            pairs.append(statistics.mean(per)); diffs.append(per[1]-per[0])
        if len(pairs)<3: continue
        n=len(pairs); mean=statistics.mean(pairs); sd=statistics.stdev(diffs)
        mde=2.80*sd/math.sqrt(n) # normal approximation: two-sided alpha .05, power .80
        bias=abs(statistics.mean(diffs)); limit=bias+mde
        out.append(dict(device=key[0],size=key[1],batch=key[2],window_s=key[3],pairs=n,
                        idle_j_mean=statistics.mean(idle),idle_j_sd=statistics.stdev(idle),
                        total_j_mean=statistics.mean(active),total_j_sd=statistics.stdev(active),
                        above_idle_j_mean=statistics.mean(net),above_idle_fraction_mean=statistics.mean(fractions),
                        gross_j_per_image_mean=mean,paired_difference_sd=sd,
                        paired_order_bias=statistics.mean(diffs),approx_mde_j_per_image=mde,
                        conservative_screen_fraction=limit/mean if mean>0 else None,
                        candidate_for_confirmation=bool(n>=30 and mean>0 and limit<=.05*mean)))
    return out

def csv_write(path, rows):
    if not rows: return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    p.add_argument('--data',default='data'); p.add_argument('--download',action='store_true')
    p.add_argument('--out',required=True); p.add_argument('--sizes',type=int,nargs='+',default=[32,64,128,224])
    p.add_argument('--batches',type=int,nargs='+',default=[1,16,64,128,256])
    p.add_argument('--windows',type=float,nargs='+',default=[5])
    p.add_argument('--repeats',type=int,default=6); p.add_argument('--interval',type=float,default=.4)
    p.add_argument('--override-fast-interval',action='store_true',
                    help='Allow --interval below the NVML counter-telescoping floor (0.3s). Logs a warning.')
    p.add_argument('--threads',type=int,default=4); p.add_argument('--warmup',type=float,default=3)
    p.add_argument('--codecarbon',action='store_true',help='Run CodeCarbon concurrently with the RAPL/NVML reads on each active window.')
    p.add_argument('--codecarbon-country',default='CZE',help='ISO code for CodeCarbon offline grid-intensity lookup (energy figure itself is country-independent).')
    p.add_argument('--checkpoint'); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    env=dict(platform=platform.platform(),cpu=platform.processor(),arguments=vars(a),
             timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else '')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    rows=[]
    try:
        if min(a.windows)<=0 or min(a.batches)<=0 or a.interval<=0: raise ValueError('Positive durations/batches required')
        if a.device=='cuda' and a.interval<0.3:
            if not a.override_fast_interval:
                raise ValueError(f'--interval {a.interval}s is below the measured NVML counter-telescoping '
                                  f'floor (0.3-0.5s on this RTX 3050; 20ms implies ~300W against a 60W cap). '
                                  f'Pass --override-fast-interval to force it anyway.')
            print(f'WARNING: --interval {a.interval}s is below the 0.3s NVML floor; '
                  f'readings will be inflated by the counter-telescoping artifact.', file=sys.stderr)
        sensor=Sensor(a.device)
        import torch, torchvision
        from torchvision import transforms
        torch.manual_seed(2026); random.seed(2026); torch.set_num_threads(a.threads)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=torchvision.models.resnet18(weights=None,num_classes=10)
        model.conv1=torch.nn.Conv2d(3,64,3,1,1,bias=False); model.maxpool=torch.nn.Identity()
        if a.checkpoint: model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
        model=model.eval().to(a.device)
        ds=torchvision.datasets.CIFAR10(a.data,train=False,download=a.download,transform=transforms.ToTensor())
        raw=torch.stack([ds[i][0] for i in range(max(a.batches))])
        sync=torch.cuda.synchronize if a.device=='cuda' else lambda:None
        env.update(torch=torch.__version__,torchvision=torchvision.__version__,backend=sensor.backend,
                   gpu=torch.cuda.get_device_name(0) if a.device=='cuda' else None,
                   weights='checkpoint' if a.checkpoint else 'seeded random weights: timing pilot only')
        (out/'environment.json').write_text(json.dumps(env,indent=2))
        with torch.inference_mode():
            for rep in range(a.repeats):
                grid=[(s,b,w) for s in a.sizes for b in a.batches for w in a.windows]; random.shuffle(grid)
                for size,batch,seconds in grid:
                    try:
                        x=torch.nn.functional.interpolate(raw[:batch],size=(size,size),mode='bilinear',align_corners=False)
                        x=(x-torch.tensor([.4914,.4822,.4465])[None,:,None,None])/torch.tensor([.247,.243,.261])[None,:,None,None]
                        x=x.to(a.device)
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
                            result=window(sensor,seconds,a.interval,work if phase.startswith('a') else None,sync)
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
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc()); raise

if __name__=='__main__': main()
