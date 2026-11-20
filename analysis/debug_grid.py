"""调试 grid_thresholds 内部。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.evaluation import score_method
from appeal.groups import build_groups, time_folds
from appeal.labels_view import LabelView
from appeal.taxonomy import Taxonomy
from appeal.trainer import train_window

b = load_bundle('data')
tax = Taxonomy(b.topics)
groups = build_groups(b)
ranks = time_folds(b)
view = LabelView(b.labels)

train_ids, select_ids = set(), set()
for root, members in groups.members.items():
    f = max(ranks[m] for m in members)
    if f == 0:
        train_ids.update(members)
    elif f == 1:
        select_ids.update(members)
train_letters = {k: b.letters[k] for k in train_ids}
select_letters = {k: b.letters[k] for k in select_ids}
print('选型折叠件数:', len(select_letters), sorted(select_ids)[:8])

trained = train_window(b, tax, view, 26, train_letters)
labels34 = view.verdicts_by_letter(34)
print('day34 选型件标签示例:')
for lid in sorted(select_ids)[:8]:
    print(' ', lid, b.letters[lid].arrived, labels34.get(lid))

# 手工阈值试算
sc = score_method(trained, 'rule', {'v2:refund': 0.8}, select_letters, labels34, tax)
print('\nrule v2:refund @0.8:', sc['v2:refund'].as_row())
sc2 = score_method(trained, 'rule', {'v2:refund': 0.8, 'v2:contact': 0.8},
                   select_letters, labels34, tax)
for k in sc2:
    print(k, sc2[k].as_row())
