"""调试第二窗 grid_thresholds：v2:refund 的 nbayes 为什么无阈值。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.evaluation import score_method, grid_thresholds
from appeal.groups import build_groups, time_folds
from appeal.labels_view import LabelView
from appeal.taxonomy import Taxonomy
from appeal.trainer import train_window

b = load_bundle('data')
tax = Taxonomy(b.topics)
g = build_groups(b)
ranks = time_folds(b)
view = LabelView(b.labels)
tr_ids, sel_ids = set(), set()
for root, members in g.members.items():
    f = max(ranks[m] for m in members)
    if f in {0, 1, 2}:
        tr_ids.update(members)
    elif f == 4:
        sel_ids.update(members)
tr = {k: b.letters[k] for k in tr_ids}
sel = {k: b.letters[k] for k in sel_ids}
print('选型件:', len(sel), sorted(sel_ids))
trained = train_window(b, tax, view, 34, tr)
labels41 = view.verdicts_by_letter(41)
for method in ('nbayes', 'logreg'):
    print('\n', method)
    for thr in (0.3, 0.5, 0.7, 0.9):
        sc = score_method(trained, method, {'v2:refund': thr, 'v2:contact': thr},
                          sel, labels41, tax)
        for k in ('v2:refund', 'v2:contact'):
            print(' ', thr, k, sc[k].as_row())
    print(' grid ->', grid_thresholds(trained, method, sel, labels41, tax, {}))
