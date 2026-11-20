"""探针：检查首窗训练产物与一个 v2 评估件的预测细节。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.evaluation import predict_letter_topics
from appeal.groups import build_groups, time_folds
from appeal.labels_view import LabelView
from appeal.taxonomy import Taxonomy
from appeal.trainer import train_window

b = load_bundle('data')
tax = Taxonomy(b.topics)
groups = build_groups(b)
ranks = time_folds(b)
view = LabelView(b.labels)

train_ids = set()
for root, members in groups.members.items():
    if max(ranks[m] for m in members) == 0:
        train_ids.update(members)
train_letters = {k: b.letters[k] for k in train_ids}
trained = train_window(b, tax, view, 26, train_letters)

for key in sorted(trained):
    tt = trained[key]
    print(key, 'pos', tt.n_pos, 'neg', tt.n_neg,
          'models', [m for m in tt.models if tt.models[m] is not None],
          'rule_phrases', len(tt.models['rule'].phrase_priors))

# v2 评估件 x174（day35, refund yes）
lid = 'x174'
lt = b.letters[lid]
print('\n', lid, lt.arrived, lt.text, tax.policy_on(lt.arrived))
print('标签(day42可见):', view.verdicts_by_letter(42).get(lid))
for method in ('rule', 'nbayes', 'logreg'):
    preds = predict_letter_topics(trained, lt, 'v2', method)
    for t, p in sorted(preds.items()):
        print(f'  {method:7s} {t:9s} prob={p.prob:.3f} oov={p.oov} cov={p.coverage:.2f}')
