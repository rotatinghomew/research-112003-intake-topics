"""按时间滚动的选型模拟：训练段 -> 选型段 -> 独立复核段 -> 在途段。

泄漏防线：
- 每个窗口独立定义来件角色（滚动窗口时间段重叠，同一来件可在不同窗口
  承担不同角色），转发组按窗口角色整组归并，防止同一来件换渠道进验证；
- 训练集排除任何“组内有成员落在本窗口选型/复核段”的转发组，
  即使该组的根件在训练段也不例外；
- 标签严格按 known 日可见性过滤，retracted 在撤销日后移除；
- 选型段只用于阈值/办法选择，独立复核段从不参与任何调参，
  因此复核段上的结果才是可对外声称的独立表现。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .dataio import DataBundle
from .evaluation import TopicScore, grid_thresholds, score_method, wilson_upper
from .groups import (ForwardGroups, ROLE_AUDIT, ROLE_SELECT, ROLE_TRAIN,
                     group_aware_members, window_roles)
from .labels_view import LabelView
from .models import CANDIDATES
from .taxonomy import Taxonomy
from .trainer import train_window


@dataclass
class WindowResult:
    name: str
    train_day: int
    train_count: int
    excluded_groups: list
    audit_scores: dict[str, dict[str, TopicScore]]  # method -> key -> score
    select_scores: dict[str, dict[str, TopicScore]]
    thresholds: dict[str, dict]                      # method -> key -> threshold/None
    method_pick: dict[str, str]                      # 'policy:topic' -> method
    desk_evidence: dict[str, dict]                  # desk -> 累计证据与准入
    trained: dict = field(repr=False, default=None)


def _desk_tolerance(bundle: DataBundle, day: int) -> dict[str, float]:
    out = {}
    for lim in bundle.limits:
        if lim.known <= day and lim.tolerance is not None:
            out[lim.desk] = lim.tolerance  # 后登记的同桌记录覆盖前者
    return out


def _desk_min_support(bundle: DataBundle, day: int) -> dict[str, int]:
    out = {}
    for lim in bundle.limits:
        if lim.known <= day and lim.min_support is not None:
            out[lim.desk] = lim.min_support
    return out


def run_simulation(bundle: DataBundle, taxonomy: Taxonomy,
                   groups: ForwardGroups) -> list[WindowResult]:
    roles_by_window = window_roles(bundle)
    view = LabelView(bundle.labels)
    results: list[WindowResult] = []
    # 跨窗口累计的独立复核证据（同一 desk 的自动送件只在复核段累计）
    pooled: dict[str, dict[str, int]] = {}

    for w in bundle.windows:
        roles = roles_by_window[w.name]

        # 评估角色（选型/复核）涉及的转发组，其全部成员都不得进入训练
        eval_letters: set[str] = set()
        eval_groups: list[list[str]] = []
        for role in (ROLE_SELECT, ROLE_AUDIT):
            ids, absorbed = group_aware_members(groups, roles, role)
            eval_letters.update(ids)
            eval_groups.extend(absorbed)

        # 训练成员：角色=train，且其转发组没有任何成员落在评估角色
        train_letters: dict = {}
        excluded: list[list[str]] = []
        for root, members in groups.members.items():
            if any(m in eval_letters for m in members):
                if len(members) > 1:
                    excluded.append(sorted(members))
                continue
            train_members = [m for m in members if roles.get(m) == ROLE_TRAIN]
            for m in train_members:
                train_letters[m] = bundle.letters[m]

        select_ids, _ = group_aware_members(groups, roles, ROLE_SELECT)
        audit_ids, _ = group_aware_members(groups, roles, ROLE_AUDIT)
        select_letters = {k: bundle.letters[k] for k in select_ids}
        audit_letters = {k: bundle.letters[k] for k in audit_ids}

        train_day = w.train_before - 1
        trained = train_window(bundle, taxonomy, view, train_day, train_letters)
        # 标签按各阶段结束日的可见性过滤：复核段来件的结论要到 audit_to 才齐
        select_labels = view.verdicts_by_letter(w.select_to)
        audit_labels = view.verdicts_by_letter(w.audit_to)
        # 容忍度/名额是第 44 日才登记的治理口径：在此之前没有自动准入依据，
        # 选型仍正常进行（选办法、定阈值），但桌级准入只能在口径登记后判定。
        tol_audit = _desk_tolerance(bundle, w.audit_to)

        thresholds, select_scores, audit_scores = {}, {}, {}
        for method in CANDIDATES:
            thresholds[method] = grid_thresholds(
                trained, method, select_letters, select_labels, taxonomy)
            select_scores[method] = score_method(
                trained, method, thresholds[method], select_letters,
                select_labels, taxonomy)
            audit_scores[method] = score_method(
                trained, method, thresholds[method], audit_letters,
                audit_labels, taxonomy)

        # 逐主题选办法：容忍度/最小支持量在“跨独立复核批次的累计池”上判定，
        # 单批复核样本小，Wilson 上界天然偏宽，不能用单批否决一个零误送的办法。
        # 本环节只做办法比较：同主题下取误送最少、覆盖最高的候选。
        method_pick: dict[str, str] = {}
        for tkey in sorted(trained):
            policy, topic = tkey.split(':', 1)
            desk = taxonomy.desk_for(policy, topic)
            if desk is None:
                continue
            candidates = []
            for method in CANDIDATES:
                sc = audit_scores[method].get(tkey)
                if sc is None or tkey not in thresholds[method]:
                    continue
                if sc.support_pos == 0 or (sc.tp + sc.fp) == 0:
                    continue
                # 排序键：误送数最少 -> 覆盖最高 -> 误送上界最低
                candidates.append(((sc.fp, -sc.coverage, sc.misroute_upper), method))
            if candidates:
                candidates.sort()
                method_pick[tkey] = candidates[0][1]

        # 桌级累计证据：按选中办法把复核段的自动送件结果并入池
        desk_evidence = {}
        for tkey, method in method_pick.items():
            policy, topic = tkey.split(':', 1)
            desk = taxonomy.desk_for(policy, topic)
            sc = audit_scores[method][tkey]
            agg = pooled.setdefault(desk, {'tp': 0, 'fp': 0, 'pos': 0})
            agg['tp'] += sc.tp
            agg['fp'] += sc.fp
            agg['pos'] += sc.support_pos
        for desk, agg in pooled.items():
            sent = agg['tp'] + agg['fp']
            min_sup = _desk_min_support(bundle, w.audit_to).get(desk)
            tolerance = tol_audit.get(desk)  # 口径未登记时为 None，不准入
            upper = wilson_upper(agg['fp'], sent)
            desk_evidence[desk] = {
                'tp': agg['tp'], 'fp': agg['fp'], 'support_pos': agg['pos'],
                'misroute_upper95': round(upper, 4),
                'min_support': min_sup,
                'tolerance_registered': tolerance is not None,
                'coverage': round(agg['tp'] / agg['pos'], 4) if agg['pos'] else None,
                'eligible': (tolerance is not None and min_sup is not None
                             and sent >= min_sup and upper <= tolerance),
            }

        results.append(WindowResult(
            name=w.name, train_day=train_day, train_count=len(train_letters),
            excluded_groups=excluded, audit_scores=audit_scores,
            select_scores=select_scores, thresholds=thresholds,
            method_pick=method_pick, desk_evidence=desk_evidence, trained=trained))
    return results
