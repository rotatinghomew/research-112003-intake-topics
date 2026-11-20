"""调试 method_pick 候选过滤。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.groups import build_groups
from appeal.taxonomy import Taxonomy
from appeal.simulation import run_simulation, _desk_tolerance

b = load_bundle('data')
tax = Taxonomy(b.topics)
groups = build_groups(b)
results = run_simulation(b, tax, groups)
r1 = results[0]
print('tol:', _desk_tolerance(b, 27))
print('\n阈值表:')
for m, d in r1.thresholds.items():
            print(' ', m, d)
print('\n审计分 keys:')
for m, d in r1.audit_scores.items():
            print(' ', m, sorted(d))
print('\npick:', r1.method_pick)
# 手工走一遍候选逻辑
for tkey in sorted(r1.trained):
    policy, topic = tkey.split(':', 1)
    desk = tax.desk_for(policy, topic)
    if desk is None:
        continue
    for method in ('rule', 'nbayes', 'logreg'):
        sc = r1.audit_scores[method].get(tkey)
        in_thr = tkey in r1.thresholds[method]
        if sc and (sc.support_pos > 0):
            print(tkey, method, 'desk=', desk, 'in_thr=', in_thr,
                  'tp=', sc.tp, 'fp=', sc.fp, 'pos=', sc.support_pos)
