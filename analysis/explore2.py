"""第二轮探索：标签覆盖口径、短语-主题对应、时点可用性。"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


def read(name):
    with (DATA / f'{name}.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


letters = {r['letter']: r for r in read('letters')}
labels = read('labels')

per_letter = defaultdict(list)
for r in labels:
    if r['verdict'] != 'retracted':
        per_letter[r['letter']].append((r['topic'], r['verdict']))
cover = Counter(len(v) for v in per_letter.values())
print('== 每封有效结论条数分布 ==', dict(cover))
one = [k for k, v in per_letter.items() if len(v) == 1]
print('只复核一项的来件:', len(one))
print('其中结论为 yes 的:', sum(1 for k in one if per_letter[k][0][1] == 'yes'))
print('总有效行:', sum(cover[k] * k for k in cover), ' 唯一来件:', len(per_letter))

# 短语 -> 主题（用 yes 标签对齐；多主题信含两段）
phrase_topic = defaultdict(Counter)
for lid, rows in per_letter.items():
    yes_topics = {t for t, v in rows if v == 'yes'}
    segs = letters[lid]['text'].replace('麻烦帮我查这件事，', '').split('；')
    for seg in segs:
        for t in yes_topics:
            phrase_topic[seg][t] += 1
print('\n== 短语 -> yes 主题计数 ==')
for p, c in phrase_topic.items():
    print(dict(c), p)

# 各时点、各主题可用的 yes/no 证据（known <= 时点）
for cutoff in (27, 35, 42, 48, 54):
    av = defaultdict(Counter)
    for r in labels:
        if int(r['known']) <= cutoff:
            av[(r['policy'], r['topic'])][r['verdict']] += 1
    print(f'\n== known <= {cutoff} ==')
    for k in sorted(av):
        print(k, dict(av[k]))

# 各 fold 内、在各时点已经有标签的来件数
folds = {'first.select': (27, 34), 'first.audit': (35, 42),
         'second.select': (35, 41), 'second.audit': (42, 47),
         'fresh48': (48, 50)}
for name, (lo, hi) in folds.items():
    members = [k for k, r in letters.items() if lo <= int(r['arrived']) <= hi]
    labeled = [k for k in members if k in per_letter]
    print(f'\n{name}: 来件 {len(members)}, 有任何有效标签 {len(labeled)}')

# x241/x242 等最新来件是否已有标签
print('\n== 48 日及以后来件的标签 ==')
for k, r in letters.items():
    if int(r['arrived']) >= 48:
        print(k, r['arrived'], per_letter.get(k, '无标签'))
