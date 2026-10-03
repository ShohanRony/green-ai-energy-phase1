import pynvml as nv, time, json, sys

nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(0)
out_path = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

with open(out_path, 'w') as f:
    while True:
        t = time.perf_counter()
        row = dict(
            t=t,
            sm_clock_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            mem_clock_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            power_mw=nv.nvmlDeviceGetPowerUsage(h),
            pstate=nv.nvmlDeviceGetPerformanceState(h),
        )
        f.write(json.dumps(row)+'\n'); f.flush()
        time.sleep(interval)
