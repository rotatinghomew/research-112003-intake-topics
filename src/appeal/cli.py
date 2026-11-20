"""命令行入口：运行滚动选型并生成分送方案与协调员视图。

用法（项目根目录）：
    python3 -I run_pipeline.py --data data --out reports --today 50
全程只读 data/，只写 out/ 指定目录；不访问网络、不调用外部模型服务。
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from .dataio import load_bundle
from .evaluation import slice_rows
from .groups import build_groups, window_roles
from .routing import route_week
from .simulation import run_simulation
from .taxonomy import Taxonomy

SCORE_FIELDS = ['window', 'phase', 'policy', 'topic', 'method', 'tp', 'fp',
                'fn', 'tn', 'support_pos', 'support_neg', 'miss_rate',
                'misroute_rate', 'misroute_upper95', 'coverage', 'oov_blocks']
SLICE_FIELDS = ['window', 'phase', 'policy', 'topic', 'method', 'expression',
                'multi_topic', 'tp', 'fp', 'fn', 'tn', 'miss_rate',
                'misroute_rate', 'coverage']


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _explain(trained, tkey: str, method: str) -> list[dict]:
    """提取真正决定分案的特征：logreg 权重、nbayes 对数似然比、规则短语。"""
    policy, topic = tkey.split(':', 1)
    tt = trained[tkey]
    model = tt.models[method]
    rows = []
    if method == 'logreg':
        scored = [((g, model.weights[i])) for g, i in tt.vocab.index.items()]
        scored = [(g, w) for g, w in scored]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        for gram, weight in scored[:12]:
            rows.append((f'gram「{gram}」', round(weight, 4)))
    elif method == 'nbayes':
        scored = [(g, model.log_pos[i] - model.log_neg[i])
                  for g, i in tt.vocab.index.items()]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        for gram, lr in scored[:12]:
            rows.append((f'gram「{gram}」', round(lr, 4)))
    else:
        for phrase, prior in sorted(model.phrase_priors.items(),
                                    key=lambda x: -x[1]):
            rows.append((f'短语「{phrase}」', round(prior, 4)))
    return [{'policy': policy, 'topic': topic, 'method': method,
             'feature': feat, 'weight': w} for feat, w in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='申诉主题分类选型与当周分送')
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='reports')
    ap.add_argument('--today', type=int, default=50,
                    help='分送模拟当日（默认 50，即资料最新在途日）')
    args = ap.parse_args(argv)

    bundle = load_bundle(args.data)
    taxonomy = Taxonomy(bundle.topics)
    groups = build_groups(bundle)
    results = run_simulation(bundle, taxonomy, groups)
    roles_by_window = window_roles(bundle)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 报告 1：选型与独立复核明细 ----
    score_rows, select_rows, pick_rows, audit_slice_rows, select_slice_rows = \
        [], [], [], [], []
    for res in results:
        for method, scores in res.audit_scores.items():
            for sc in scores.values():
                row = sc.as_row()
                row.update(window=res.name, phase='audit')
                score_rows.append(row)
            audit_slice_rows += slice_rows(scores, res.name, 'audit')
        for method, scores in res.select_scores.items():
            for sc in scores.values():
                row = sc.as_row()
                row.update(window=res.name, phase='select')
                select_rows.append(row)
            select_slice_rows += slice_rows(scores, res.name, 'select')
        for key, method in sorted(res.method_pick.items()):
            policy, topic = key.split(':', 1)
            sc = res.audit_scores[method][key]
            pick_rows.append({
                'window': res.name, 'policy': policy, 'topic': topic,
                'desk': taxonomy.desk_for(policy, topic), 'method': method,
                'threshold': res.thresholds[method].get(key),
                'audit_coverage': sc.coverage,
                'audit_misroute_rate': sc.misroute_rate,
                'audit_misroute_upper95': round(sc.misroute_upper, 4),
                'audit_support_pos': sc.support_pos,
                'audit_support_neg': sc.support_neg, 'oov_blocks': sc.oov_blocks,
            })
    _write_csv(out / 'audit_scores.csv', score_rows, SCORE_FIELDS)
    _write_csv(out / 'select_scores.csv', select_rows, SCORE_FIELDS)
    _write_csv(out / 'slice_scores.csv',
               audit_slice_rows + select_slice_rows, SLICE_FIELDS)
    _write_csv(out / 'method_selection.csv', pick_rows,
               ['window', 'policy', 'topic', 'desk', 'method', 'threshold',
                'audit_coverage', 'audit_misroute_rate', 'audit_misroute_upper95',
                'audit_support_pos', 'audit_support_neg', 'oov_blocks'])

    # ---- 报告 2：桌级准入（累计独立证据） ----
    desk_rows = []
    for res in results:
        for desk, ev in sorted(res.desk_evidence.items()):
            row = {'window': res.name, 'desk': desk}
            row.update(ev)
            desk_rows.append(row)
    _write_csv(out / 'desk_eligibility.csv', desk_rows,
               ['window', 'desk', 'tp', 'fp', 'support_pos', 'min_support',
                'misroute_upper95', 'coverage', 'tolerance_registered', 'eligible'])

    # ---- 报告 3：转发泄漏审计（每窗口角色 + 组归并 + 关系确认日） ----
    leak_rows = []
    for res in results:
        roles = roles_by_window[res.name]
        for root, members in groups.members.items():
            if len(members) <= 1:
                continue
            for m in sorted(members, key=lambda x: bundle.letters[x].arrived):
                leak_rows.append({
                    'window': res.name, 'root': root, 'letter': m,
                    'arrived': bundle.letters[m].arrived,
                    'role': roles.get(m), 'parent': groups._parent.get(m, ''),
                    'forward_known': groups.known_day.get(m, ''),
                    'excluded_from_train': int(any(
                        mm in {x for row in res.excluded_groups for x in row}
                        for mm in members)),
                })
    _write_csv(out / 'forwarding_audit.csv', leak_rows,
               ['window', 'root', 'letter', 'arrived', 'role', 'parent',
                'forward_known', 'excluded_from_train'])

    # ---- 报告 4：当周分送（用最新窗口的选型结果） ----
    latest = results[-1]
    eligible = {d for d, ev in latest.desk_evidence.items() if ev['eligible']}
    live_thresholds = {k: latest.thresholds[m][k]
                       for k, m in latest.method_pick.items()}
    fresh_from = next(w.fresh_from for w in bundle.windows
                      if w.name == latest.name)
    fresh = {lid: lt for lid, lt in bundle.letters.items()
             if lt.arrived >= fresh_from}
    outcome = route_week(bundle, taxonomy, latest.trained, latest.method_pick,
                         live_thresholds, eligible, args.today, fresh,
                         bundle.assignments)
    route_rows = [{
        'letter': r.letter, 'topic': r.topic, 'desk': r.desk,
        'auto': int(r.auto), 'state': r.state, 'reason': r.reason,
        'base_reason': r.base_reason,
        'prob': r.prob, 'oov': int(r.oov), 'urgent': int(r.urgent),
        'arrived': r.arrived,
    } for r in outcome.routes]
    _write_csv(out / 'routing_plan.csv', route_rows,
               ['letter', 'arrived', 'urgent', 'topic', 'desk', 'auto',
                'state', 'reason', 'base_reason', 'prob', 'oov'])

    # ---- 报告 5：协调员视图（逐件清单，含紧急标记、底层与最终待分原因） ----
    coord_rows = [{
        'letter': r.letter, 'topic': r.topic, 'desk': r.desk,
        'state': r.state, 'reason': r.reason, 'base_reason': r.base_reason,
        'urgent': int(r.urgent),
    } for r in outcome.routes]
    _write_csv(out / 'coordinator_view.csv', coord_rows,
               ['letter', 'topic', 'desk', 'state', 'reason', 'base_reason',
                'urgent'])

    # ---- 报告 6：分案决定性特征（最新窗口选中模型） ----
    expl_rows = []
    for tkey, method in latest.method_pick.items():
        expl_rows += _explain(latest.trained, tkey, method)
    _write_csv(out / 'feature_explanations.csv', expl_rows,
               ['policy', 'topic', 'method', 'feature', 'weight'])

    # 控制台摘要
    print(f'窗口 {results[0].name} / {results[1].name} 选型完成')
    for res in results:
        print(f'-- {res.name}: 训练 {res.train_count} 件，'
              f'排除转发组 {len(res.excluded_groups)} 个，'
              f'入选自动主题 {len(res.method_pick)} 个')
        for desk, ev in sorted(res.desk_evidence.items()):
            flag = '可自动' if ev['eligible'] else '暂不自动'
            print(f'   桌 {desk:8s} {flag}  累计送件 {ev["tp"] + ev["fp"]} '
                  f'(误 {ev["fp"]}) 上界 {ev["misroute_upper95"]} 覆盖 {ev["coverage"]}')
    print(f'当周分送: 送达 {len(outcome.sent)} 条主题，待分 {len(outcome.waiting)} 条')
    print(f'报告已写入 {out}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
