"""Two processes delta-index the same edit at the same moment (the hook's
freshen and the session's auto-refresh do exactly this). Count live docs
for the edited path afterwards: 1 is right, 2 is a duplicate."""
import os
import json, subprocess, sys, time, threading
DATA = os.path.expanduser("~/.indexio")
MARKF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "SPEC-P9.md")
EXE = sys.argv[1]
def reindex():
    subprocess.run([EXE, "reindex", "--data-dir", DATA, "--repo", "indexio", "--worktree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def live_docs(mark):
    out = subprocess.run([EXE, "search", "--data-dir", DATA, "--json", "--limit", "20", mark], capture_output=True, text=True, encoding="utf-8").stdout
    try:
        return len(json.loads(out)["hits"])
    except Exception:
        return -1
results = []
for i in range(4):
    mark = "concurrentprobe_%d_%d" % (int(time.time()), i)
    with open(MARKF, "a", encoding="utf-8") as f:
        f.write("\nProbe marker %s.\n" % mark)
    ts = [threading.Thread(target=reindex) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    results.append(live_docs(mark))
print(EXE.split("\\")[-3] if "target" in EXE else "deployed", "live docs per probe (1 = correct):", results)
# clean up
s = open(MARKF, encoding="utf-8").read()
lines = [l for l in s.split("\n") if not l.startswith("Probe marker concurrentprobe_")]
open(MARKF, "w", encoding="utf-8", newline="\n").write("\n".join(lines).rstrip("\n") + "\n")
reindex()
