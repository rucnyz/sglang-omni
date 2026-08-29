import sqlite3, sys
con=sqlite3.connect(sys.argv[1]); cur=con.cursor()
mid={n:i for i,n in cur.execute("SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS")}
def series(name):
    i=mid.get(name)
    if i is None: return []
    return [(t,v) for t,v in cur.execute("SELECT timestamp,value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp",(i,))]
gr=series("GR Active [Throughput %]")
sm=series("SMs Active [Throughput %]")
# active span = samples where GR-active>5 (GPU actually running compute) → drain-excluded
active_idx=set(k for k,(t,v) in enumerate(gr) if v>5)
def avg(s, idxs=None):
    vals=[v for k,(t,v) in enumerate(s) if (idxs is None or k in idxs)]
    return sum(vals)/len(vals) if vals else 0.0
busy_frac=len(active_idx)/len(gr) if gr else 0
print(f"  busy-fraction (GR>5%): {busy_frac*100:.0f}%   [drain/idle excluded below]")
for name in ["SMs Active [Throughput %]","GR Active [Throughput %]","Tensor Active [Throughput %]","Compute Warps in Flight [Throughput %]","DRAM Read Bandwidth [Throughput %]"]:
    s=series(name)
    print(f"  {name:<42} session {avg(s):5.1f}%   active-span {avg(s,active_idx):5.1f}%")
