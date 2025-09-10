# mem_probe.py
import os, sys, time, gc
import tracemalloc

# ---- RSS/VMS via psutil if present; fallback to /proc/self/status ----
try:
    import psutil
    _PROC = psutil.Process(os.getpid())

    def _rss_vms():
        m = _PROC.memory_info()
        return m.rss, m.vms
except Exception:
    def _rss_vms():
        rss = vms = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                    elif line.startswith("VmSize:"):
                        vms = int(line.split()[1]) * 1024
        except Exception:
            pass
        return rss, vms

def _fmt_bytes(n):
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

class MemTracker:
    """
    Tracks OS RSS/VMS and Python allocations (tracemalloc).
    Usage:
        mt = MemTracker(start_tracemalloc=True)
        mt.check("after first read")
        ...
        mt.check("after parse", top=10)
    """
    def __init__(self, start_tracemalloc=True, n_frames=25):
        self._started_tm = False
        self.snap_prev = None
        if start_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(n_frames)
            self._started_tm = True

    def check(self, tag="", top=5, gc_collect=False):
        if gc_collect:
            gc.collect()

        rss, vms = _rss_vms()
        msg = [f"[MEM] {tag}  RSS={_fmt_bytes(rss)}  VMS={_fmt_bytes(vms)}"]

        snap = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
        if snap is not None:
            if self.snap_prev is None:
                stats = snap.statistics("lineno")
                total = sum(s.size for s in stats)
                msg.append(f"[PY] total tracked={_fmt_bytes(total)} (first snapshot)")
                # top alloc sites
                for s in stats[:top]:
                    msg.append(f"      {s.traceback.format()[-1].strip()}  +{_fmt_bytes(s.size)} in {s.count} blocks")
            else:
                diff = snap.compare_to(self.snap_prev, "lineno")
                grow = [d for d in diff if d.size_diff > 0]
                total_grow = sum(d.size_diff for d in grow)
                msg.append(f"[PY] delta since last check: +{_fmt_bytes(total_grow)}")
                for d in grow[:top]:
                    tb_last = d.traceback.format()[-1].strip()
                    msg.append(f"      {tb_last}  +{_fmt_bytes(d.size_diff)} in {d.count_diff} blocks")
            self.snap_prev = snap

        print("\n".join(msg))
        
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    def _rss_vms():
        m = _PROC.memory_info()
        return m.rss, m.vms
except Exception:
    def _rss_vms():
        # Linux fallback
        rss = vms = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):  rss = int(line.split()[1]) * 1024
                    if line.startswith("VmSize:"): vms = int(line.split()[1]) * 1024
        except Exception:
            pass
        return rss, vms

def _fmt_bytes(n):
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

class EpochMemLogger:
    def __init__(self, use_tracemalloc=True, n_frames=25, csv_path=None, logger=None):
        self.logger = logger
        self.csv_path = csv_path
        if use_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(n_frames)
        if csv_path and not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("ts,epoch,rss_bytes,vms_bytes,py_current,py_peak,tag\n")

    def log(self, epoch:int, tag:str="end_of_epoch", gc_collect=False):
        if gc_collect:
            gc.collect()

        rss, vms = _rss_vms()
        if tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
        else:
            cur = peak = -1

        msg = (f"[MEM] {tag} e{epoch}  RSS={_fmt_bytes(rss)}  "
               f"VMS={_fmt_bytes(vms)}  PY(cur/peak)={_fmt_bytes(max(cur,0))}/"
               f"{_fmt_bytes(max(peak,0)) if peak>=0 else 'N/A'}")
        print(msg) if self.logger is None else self.logger.info(msg)

        if self.csv_path:
            with open(self.csv_path, "a") as f:
                f.write(f"{time.time():.3f},{epoch},{rss},{vms},{cur},{peak},{tag}\n")