import gzip, json, sys, glob, collections
for f in sorted(glob.glob(sys.argv[1])):
    try:
        with gzip.open(f) as fp: tr=json.load(fp)
    except Exception as e: print(f,"ERR",e); continue
    evs=tr.get("traceEvents",tr) if isinstance(tr,dict) else tr
    pname=""; kern=collections.Counter(); kcnt=collections.Counter()
    for e in evs:
        if e.get("ph")=="M" and e.get("name")=="process_name":
            pname=e.get("args",{}).get("name","")
        cat=e.get("cat","")
        if cat in ("kernel","gpu_kernel") or (cat=="" and "dur" in e and e.get("name","").startswith(("void ","ampere","cutlass","sm90","sm100","triton","_"))):
            if "dur" in e: kern[e["name"][:70]]+=e["dur"]; kcnt[e["name"][:70]]+=1
    tot=sum(kern.values())
    print(f"\n=== {f.split('/')[-1]}  proc='{pname}'  total_kernel_us={tot:.0f} ({len(kern)} distinct) ===")
    for name,us in kern.most_common(8):
        print(f"  {us/max(tot,1)*100:5.1f}%  {us/1000:8.1f}ms  x{kcnt[name]:<5} {name}")
