import urllib.request, json
from packaging.version import Version
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/sgl-kernel/json"))
rel = d["releases"]
vs = sorted((v for v in rel if not Version(v).is_prerelease), key=Version)
print("newest 12 by version:", vs[-12:])
for v in vs[-6:]:
    files = rel[v]
    rp = files[0].get("requires_python")
    print("\n%s  requires_python=%s" % (v, rp))
    for f in files[:8]:
        print("   ", f["filename"])
