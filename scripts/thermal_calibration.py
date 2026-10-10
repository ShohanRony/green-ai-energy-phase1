#!/home/shohan/green-ai-research/measurement_env/bin/python3
import sys, time, json, datetime, csv, os, argparse, subprocess
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from pilot import check_ac_power, check_platform_profile, check_governor, p_core_set, ThermalGuard
from train_baseline import build_model

def get_cmd_output(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return e.output.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes-per-model', type=float, default=15.0)
    parser.add_argument('--out', type=str, default=str(repo_root / f'calibration/thermal_{datetime.datetime.now().strftime("%Y%m%d")}'))
    args = parser.parse_args()

    if not check_ac_power():
        raise RuntimeError("AC power is offline")
    prof = check_platform_profile()
    check_governor()

    import glob
    gov_paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    govs = [Path(p).read_text().strip() for p in gov_paths] if gov_paths else []
    gov_counts = {g: govs.count(g) for g in set(govs)}
    gov_str = ", ".join(f"{v} {k}" for k, v in gov_counts.items())

    drv_paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_driver')
    drv = Path(drv_paths[0]).read_text().strip() if drv_paths else ""

    epp_paths = glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference')
    epp = Path(epp_paths[0]).read_text().strip() if epp_paths else ""

    powerprofilesctl_out = get_cmd_output("powerprofilesctl get")
    thermald_active = get_cmd_output("systemctl is-active thermald")
    git_hash = get_cmd_output("git rev-parse HEAD")

    pcores = sorted(list(p_core_set()))
    os.sched_setaffinity(0, pcores)
    torch.set_num_threads(4)
    
    guard = ThermalGuard('cpu', require_thresholds=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    env = {
        'kernel': os.uname().release,
        'cpu_model': [l.split(':')[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if 'model name' in l][0],
        'governor': gov_str,
        'scaling_driver': drv,
        'energy_performance_preference': epp,
        'platform_profile': prof,
        'powerprofilesctl': powerprofilesctl_out,
        'thermald_active': thermald_active,
        'thermal_sensor_type': guard.temp_type,
        'git_hash': git_hash,
        'start_utc': start_utc,
        'end_utc': "",
        'input_shape': [1, 3, 32, 32],
        'threads': 4,
        'minutes_per_model': args.minutes_per_model,
        'ambient_note': "",
        'invalid_reason': ""
    }
    env_path = out_dir / 'environment.json'
    env_path.write_text(json.dumps(env, indent=2))
    
    csv_path = out_dir / 'windows.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_utc', 'model', 'window_idx', 'duration_s', 'n_samples', 'mean_freq_mhz', 'min_freq_mhz', 'core_throttle_delta', 'pkg_throttle_delta', 'pkg_throttle_ms_delta', 'pkg_temp_max_c'])
        
        for model_name in ['resnet18_fp32.pt', 'resnet18_int8.pt']:
            torch.manual_seed(2026)
            x = torch.randn(1, 3, 32, 32)
            if 'int8' in model_name:
                model = torch.jit.load(str(repo_root / f'checkpoints/{model_name}'))
            else:
                m = build_model('resnet18')
                m.load_state_dict(torch.load(str(repo_root / f'checkpoints/{model_name}'), map_location='cpu', weights_only=True))
                m.eval()
                model = torch.jit.trace(m, x)
            model.eval()
            
            start_time = time.perf_counter()
            window_idx = 0
            
            while time.perf_counter() - start_time < args.minutes_per_model * 60:
                win_start = time.perf_counter()
                trace = []
                
                # Check guard conditions
                if not check_ac_power():
                    env['invalid_reason'] = "AC power lost during run"
                    env_path.write_text(json.dumps(env, indent=2))
                    raise RuntimeError("AC power lost")
                try:
                    check_governor()
                except RuntimeError as e:
                    env['invalid_reason'] = f"Governor check failed: {e}"
                    env_path.write_text(json.dumps(env, indent=2))
                    raise
                
                trace.append(guard.sample())
                last_sample = time.perf_counter()

                while time.perf_counter() - win_start < 5:
                    with torch.inference_mode():
                        model(x)
                    now = time.perf_counter()
                    if now - last_sample >= 0.4:
                        trace.append(guard.sample())
                        last_sample = now
                
                trace.append(guard.sample())

                eval_res = guard.evaluate(trace, active=True)
                
                def fmt(v):
                    return "" if v is None else v

                writer.writerow([
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    model_name,
                    window_idx,
                    time.perf_counter() - win_start,
                    len(trace),
                    fmt(eval_res.get('thermal_pcore_mean_freq_mhz')),
                    fmt(eval_res.get('thermal_pcore_min_freq_mhz')),
                    fmt(eval_res.get('thermal_core_throttle_delta')),
                    fmt(eval_res.get('thermal_pkg_throttle_count_delta')),
                    fmt(eval_res.get('thermal_pkg_throttle_time_ms_delta')),
                    fmt(eval_res.get('thermal_pkg_temp_max_c'))
                ])
                f.flush()
                window_idx += 1

    env['end_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env_path.write_text(json.dumps(env, indent=2))

if __name__ == '__main__':
    main()
