"""端到端调试：检查模拟产物的阈值与 v2 审计件打分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.evaluation import predict_letter_topics
from appeal.groups import build_groups
from appeal.labels_view import LabelView
from appeal.taxonomy import Taxonomy
from appeal.simulation import run_simulation

b = load_bundle('data')
tax = Taxonomy(b.topics)
groups = build_groups(b)
results = run_simulation(b, tax, groups)
r1 = results[0]
print('阈值:')
for m, d in r1.thresholds.items():
    print(' ', m, d)

tt = r1.trained['v2:refund']
print('\nv2:refund 模型:', [m for m in tt.models if tt.models[m] is not None],
      '证据', tt.evidence_sources)
lt = b.letters['x174']
for m in ('rule', 'nbayes', 'logreg'):
    model = tt.models[m]
    if model is None:
        print(m, '无模型')
        continue
    from appeal.features import split_segments
    for s in split_segments(lt.text):
        p = model.predict_segment(s)
        print(f'  {m}: prob={p.prob} oov={p.oov} cov={p.coverage}')

# 直接看预测聚合
print('\n聚合预测:')
for m in ('rule', 'nbayes', 'logreg'):
    print(m, {k: (round(v.prob, 3) if v.prob is not None else None, v.oov)
              for k, v in predict_letter_topics(r1.trained, lt, 'v2', m).items()})
