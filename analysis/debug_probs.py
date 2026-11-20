"""调试 nbayes/logreg 在第二窗审计件上的概率与阈值。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.features import split_segments
from appeal.groups import build_groups, time_folds
from appeal.labels_view import LabelView
from appeal.taxonomy import Taxonomy
from appeal.trainer import train_window

b = load_bundle('data')
tax = Taxonomy(b.topics)
g = build_groups(b)
ranks = time_folds(b)
view = LabelView(b.labels)
ids = set()
for root, members in g.members.items():
    if max(ranks[m] for m in members) in {0, 1, 2, 3}:
        ids.update(members)
tr = {k: b.letters[k] for k in ids}
trained = train_window(b, tax, view, 34, tr)

for lid in ('x207', 'x209', 'x206'):
    lt = b.letters[lid]
    print(lid, lt.text)
    for tkey in ('v2:refund', 'v2:contact'):
        tt = trained[tkey]
        for method in ('nbayes', 'logreg'):
            model = tt.models[method]
            for seg in split_segments(lt.text):
                p = model.predict_segment(seg)
                print(f'  {tkey} {method:7s} seg={seg[:10]} prob={p.prob:.3f} cov={p.coverage:.2f}')
    print()
