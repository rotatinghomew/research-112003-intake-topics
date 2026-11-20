"""只读探索：核对资料分布，供设计决策使用。用 python3 -I 运行。"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


def read(name):
    with (DATA / f'{name}.csv').open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


letters = {r['letter']: r for r in read('letters')}
labels = read('labels')
forwarding = read('forwarding')
windows = read('windows')

# 1. 标签：每主题 yes/no/retracted 数（按 policy）
pol_topic = defaultdict(Counter)
for r in labels:
    pol_topic[(r['policy'], r['topic'])][r['verdict']] += 1
print('== 标签按 policy/topic ==')
for k in sorted(pol_topic):
    print(k, dict(pol_topic[k]))

# 2. 每封来件的人工覆盖主题数；只复核一项的占比
per_letter = defaultdict(list)
for r in labels:
    if r['verdict'] != 'retracted':
        per_letter[r['letter']].append((r['topic'], r['verdict']))
cover = Counter(len(v) for v in per_letter.items())
print('\n== 每封有效结论条数分布 ==', dict(cover))
one_item = [k for k, v in per_letter.items() if len(v) == 1]
print('只复核一项的来件数:', len(one_item))

# 3. 文本短语清单（按分号切）
phrases = Counter()
for r in letters.values():
    for seg in r['text'].replace('麻烦帮我查这件事，', '').split('；'):
        phrases[seg] += 1
print('\n== 短语种类 ==', len(phrases))
for p, c in phrases.most_common():
    print(f'{c:4d}  {p}')

# 4. 模板 vs 自由写法
print('\n== expression ==', Counter(r['expression'] for r in letters.values()))
print('== 模板开头 ==', Counter(r['text'].startswith('麻烦帮我查这件事') for r in letters.values()))

# 5. 多主题来件（以分号切出两段）
multi = [k for k, r in letters.items() if '；' in r['text']]
print('\n== 含两段的来件数 ==', len(multi))

# 6. 转发组
parent = {r['copy']: r['parent'] for r in forwarding}
groups = defaultdict(list)
for r in letters:
    cur = r
    while cur in parent:
        cur = parent[cur]
    groups[cur].append(r)
print('\n== 转发组 ==', len(groups), '个根；含副本的组:')
for root, members in groups.items():
    if len(members) > 1:
        days = [letters[m]['arrived'] for m in members]
        print(root, members, 'arrived', days)

# 7. 窗口内来件量
print('\n== 各窗口来件量 ==')
for w in windows:
    for phase, lo, hi in [('train', 1, int(w['train_before']) - 1),
                          ('select', int(w['select_from']), int(w['select_to'])),
                          ('audit', int(w['audit_from']), int(w['audit_to'])),
                          ('fresh', int(w['fresh_from']), None)]:
        if hi is None:
            n = sum(1 for r in letters.values() if int(r['arrived']) >= lo)
        else:
            n = sum(1 for r in letters.values() if lo <= int(r['arrived']) <= hi)
        print(w['name'], phase, n)

# 8. 标签 known 日与来件日的滞后
lag = Counter()
for r in labels:
    if r['verdict'] != 'retracted':
        lag[int(r['known']) - int(letters[r['letter']]['arrived'])] += 1
print('\n== 标签滞后(known-arrived) ==', dict(sorted(lag.items())))

# 9. v2 新子类首批 known 日
for t in ('identity', 'promise'):
    ks = sorted(int(r['known']) for r in labels if r['topic'] == t and r['verdict'] == 'yes')
    print(f'== {t} yes 标签 known 日: 最早 {ks[0]}, 至 34 日累计 {sum(1 for k in ks if k <= 34)}, 至 41 日 {sum(1 for k in ks if k <= 41)}')

# 10. 新来件 x241-x245 的结构
print('\n== 末日来件 ==')
for k in ('x241', 'x242', 'x243', 'x244', 'x245'):
    print(k, letters[k])
