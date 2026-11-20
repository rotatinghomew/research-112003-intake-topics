"""标定 OOV 阈值：训练语料内覆盖率分布 vs 已知新表达（x241 两段）。"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from appeal.dataio import load_bundle
from appeal.features import FeatureVocab, split_segments
from appeal.groups import build_groups, time_folds
from appeal.taxonomy import Taxonomy

b = load_bundle('data')
tax = Taxonomy(b.topics)
groups = build_groups(b)
ranks = time_folds(b)

# 首窗训练折叠（组感知）
train_ids = set()
for root, members in groups.members.items():
    if max(ranks[m] for m in members) == 0:
        train_ids.update(members)
train_letters = {k: b.letters[k] for k in train_ids}
vocab = FeatureVocab().fit([s for lt in train_letters.values()
                            for s in split_segments(lt.text)])

covs = sorted((vocab.coverage(s), lt.letter, s)
              for lt in train_letters.values() for s in split_segments(lt.text))
print('训练段覆盖率：min', covs[0][0], 'p5', covs[len(covs) // 20][0],
      '中位', covs[len(covs) // 2][0], 'min5:')
for c, lid, s in covs[:5]:
    print(' ', round(c, 3), lid, s)

print('\n全部 12 个标准短语覆盖率:')
seen = set()
for lt in train_letters.values():
    for s in split_segments(lt.text):
        seen.add(s)
for s in sorted(seen):
    print(' ', round(vocab.coverage(s), 3), s)

print('\n在途新表达 x241:')
for s in split_segments(b.letters['x241'].text):
    print(' ', round(vocab.coverage(s), 3), s)
print('在途标准表达 x242/x245:')
for lid in ('x242', 'x245'):
    for s in split_segments(b.letters[lid].text):
        print(' ', lid, round(vocab.coverage(s), 3), s)
