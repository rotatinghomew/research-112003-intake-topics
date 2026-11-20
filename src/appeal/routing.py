"""当周分送引擎：把新来件的各主题送到已验证可承接的桌面或人工补读位。

硬规则：
- 只有独立复核中通过桌级容忍度与最小支持量的桌面才自动接件；
- 未登录表达、证据不足的新子类、v2 上位类 claim 一律转 manual；
- 一封信的不同主题可分别处置：一个主题先送达，不等同封所有主题都确定；
- 同一来件对同一桌面只占一个接件位（多主题同桌合并）；
- 窗口满或缺席不丢件，未分送主题进 waiting 并带原因，紧急件优先；
- 已签收（received 且有 ack）的历史分送保留，只对未确认（waiting）部分重新分送；
- 预测结果不是标签；本模块不产出任何 verdict。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .dataio import AssignmentRow, DataBundle
from .features import split_segments
from .taxonomy import MANUAL_DESK, Taxonomy

# waiting / 转人工原因码
R_OOV = 'oov_expression'            # 未登录表达
R_NO_METHOD = 'no_verified_method'  # 主题尚无通过独立复核的自动办法
R_PARENT_ONLY = 'parent_only_topic'  # 只有上位类证据，不能指定新桌面
R_NO_DESK = 'desk_not_in_scope'     # 目录中该主题当前不挂桌
R_FULL = 'desk_window_full'         # 窗口名额已满
R_ABSENT = 'desk_unavailable'       # 桌面缺席/无有效窗口


@dataclass
class Route:
    letter: str
    topic: str
    desk: str
    auto: bool
    state: str          # received / waiting
    reason: str         # 最终原因（送达方式或等待原因）
    base_reason: str    # 主题判定层面的底层原因（OOV/无办法/上位类等）
    prob: float | None
    oov: bool
    urgent: bool
    arrived: int


@dataclass
class RoutingOutcome:
    routes: list[Route] = field(default_factory=list)
    # (desk, 窗口起点) -> 占用该窗口的来件集合（同信同桌一位）
    desk_letters: dict = field(default_factory=dict)

    @property
    def sent(self):
        return [r for r in self.routes if r.state == 'received']

    @property
    def waiting(self):
        return [r for r in self.routes if r.state == 'waiting']


def active_limits(bundle: DataBundle, today: int) -> dict[str, list]:
    """desk -> 当日有效且已登记的窗口记录。"""
    out: dict[str, list] = {}
    for lim in bundle.limits:
        if lim.known <= today and lim.from_day <= today <= lim.through:
            out.setdefault(lim.desk, []).append(lim)
    return out


def _desk_quota(bundle: DataBundle, today: int, desk: str):
    windows = active_limits(bundle, today).get(desk)
    if not windows:
        return None
    return sorted(windows, key=lambda x: x.from_day)[-1]


def _prob(pred):
    if pred is None or pred.prob is None:
        return None
    return round(pred.prob, 4)


def _safe_predict(trained, letter, policy, method_pick):
    """按各主题选中的办法逐请求段预测；未选中办法的主题不产出概率。"""
    merged = {}
    for tkey, method in method_pick.items():
        pol, topic = tkey.split(':', 1)
        if pol != policy:
            continue
        tt = trained.get(tkey)
        if tt is None or tt.models.get(method) is None:
            continue
        best = None
        for seg in split_segments(letter.text):
            pred = tt.models[method].predict_segment(seg)
            if pred.prob is None:
                continue  # 规则模型弃权段
            if best is None or pred.prob > best.prob:
                best = pred
        if best is not None:
            merged[topic] = best
    return merged


def _decide_topic(taxonomy: Taxonomy, policy: str, topic: str, pred,
                  method_pick: dict[str, str], thresholds: dict[str, float | None],
                  eligible_desks: set[str]) -> tuple[bool, str, object]:
    """返回 (是否自动, 原因码, 预测对象)。不自动时原因码说明转人工/待分理由。

    原因优先级：表达层（OOV）> 目录层（无桌/上位类）> 准入层（无验证办法），
    让协调员先看到最具体、最可操作的拦截原因。
    """
    tkey = f'{policy}:{topic}'
    desk = taxonomy.desk_for(policy, topic)
    if desk is None:
        if taxonomy.is_parent_only(policy, topic):
            return False, R_PARENT_ONLY, pred
        return False, R_NO_DESK, pred
    # 表达层先判：即便桌面未准入，OOV 也是更具体的人工补读理由
    if pred is not None and pred.oov:
        return False, R_OOV, pred
    if tkey not in method_pick or desk not in eligible_desks:
        return False, R_NO_METHOD, pred
    if pred is None:
        return False, R_NO_METHOD, pred
    thr = thresholds.get(tkey, 1.01)
    if thr is None:
        if pred.prob is None:  # 规则模型弃权
            return False, R_NO_METHOD, pred
    elif pred.prob < thr:
        return False, R_NO_METHOD, pred
    return True, 'auto_verified', pred


def route_week(bundle: DataBundle, taxonomy: Taxonomy, trained: dict,
               method_pick: dict[str, str], thresholds: dict[str, float | None],
               eligible_desks: set[str], today: int,
               fresh_letters: dict,
               seed_assignments: list[AssignmentRow] | None = None) -> RoutingOutcome:
    outcome = RoutingOutcome()
    seed_assignments = seed_assignments or []

    # 1) 已签收分送原样保留并占名额；waiting 种子件留待本轮重分
    locked_topics: set[tuple[str, str]] = set()
    pending_seed: list[AssignmentRow] = []
    for a in seed_assignments:
        if a.state == 'received' and a.ack is not None:
            locked_topics.add((a.letter, a.topic))
            outcome.desk_letters.setdefault(a.desk, set()).add(a.letter)
            outcome.routes.append(Route(
                a.letter, a.topic, a.desk, a.desk != MANUAL_DESK, 'received',
                'acknowledged', 'acknowledged', None, False,
                bundle.letters[a.letter].urgent, a.issued))
        else:
            pending_seed.append(a)

    def try_occupy(desk: str, letter_id: str) -> tuple[bool, str | None]:
        """返回 (是否占到位, 失败原因码)。同信同桌只占一位。"""
        win = _desk_quota(bundle, today, desk)
        if win is None:
            return False, R_ABSENT
        occupied = outcome.desk_letters.setdefault(desk, set())
        if letter_id in occupied:
            return True, None
        if len(occupied) >= win.quota:
            return False, R_FULL
        occupied.add(letter_id)
        return True, None

    # 2) 紧急优先、再按到达日、再按编号稳定排序
    ordered = sorted(fresh_letters.values(),
                     key=lambda lt: (not lt.urgent, lt.arrived, lt.letter))
    seed_waiting = {(a.letter, a.topic) for a in pending_seed}

    for letter in ordered:
        lid = letter.letter
        policy = taxonomy.policy_on(letter.arrived)
        preds = _safe_predict(trained, letter, policy, method_pick)

        # 需要承接的主题：模型识别 ∪ 种子 waiting，剔除已签收
        wanted = set(preds) | {t for (l, t) in seed_waiting if l == lid}
        wanted = {t for t in wanted if (lid, t) not in locked_topics}

        # 兜底：模型对所有主题都弃权且无种子待办时，整封转人工补读，
        # 绝不允许出现“没有任何桌面承接”的来件。
        if not wanted and not any(l == lid for (l, _t) in locked_topics):
            win = _desk_quota(bundle, today, MANUAL_DESK)
            ok, fail_code = try_occupy(MANUAL_DESK, lid)
            outcome.routes.append(Route(
                lid, '_letter_', MANUAL_DESK, False,
                'received' if ok else 'waiting',
                R_NO_METHOD if ok else fail_code, R_NO_METHOD, None, False,
                letter.urgent, letter.arrived))
            continue

        # 逐主题定目标桌，再按桌聚合（同信同桌合并为一个接件位）
        desk_items: dict[str, list[tuple[str, bool, str, object]]] = {}
        for topic in sorted(wanted):
            auto, reason, pred = _decide_topic(
                taxonomy, policy, topic, preds.get(topic), method_pick,
                thresholds, eligible_desks)
            desk = taxonomy.desk_for(policy, topic) if auto else MANUAL_DESK
            if desk is None:
                desk, auto, reason = MANUAL_DESK, False, R_PARENT_ONLY
            desk_items.setdefault(desk, []).append((topic, auto, reason, pred))

        for desk, items in desk_items.items():
            ok, fail_code = try_occupy(desk, lid)
            any_auto = any(a for _, a, _, _ in items)
            for topic, auto, reason, pred in items:
                if ok:
                    outcome.routes.append(Route(
                        lid, topic, desk, any_auto and auto, 'received',
                        'auto_verified' if any_auto and auto else reason,
                        reason,
                        _prob(pred), False if pred is None else pred.oov,
                        letter.urgent, letter.arrived))
                else:
                    # 满员/缺席：不改投别的桌，保留待分；底层原因(base_reason)
                    # 记录该主题为何需要这个桌，协调员可同时看到两层原因。
                    outcome.routes.append(Route(
                        lid, topic, desk, False, 'waiting', fail_code, reason,
                        _prob(pred), False if pred is None else pred.oov,
                        letter.urgent, letter.arrived))

    # 3) 不在本轮 fresh 列表内的种子 waiting 件，重新评估一次
    for a in pending_seed:
        if a.letter in fresh_letters or (a.letter, a.topic) in locked_topics:
            continue
        letter = bundle.letters[a.letter]
        policy = taxonomy.policy_on(letter.arrived)
        preds = _safe_predict(trained, letter, policy, method_pick)
        auto, reason, pred = _decide_topic(
            taxonomy, policy, a.topic, preds.get(a.topic), method_pick,
            thresholds, eligible_desks)
        desk = taxonomy.desk_for(policy, a.topic) if auto else MANUAL_DESK
        if desk is None:
            desk, auto, reason = MANUAL_DESK, False, R_PARENT_ONLY
        ok, fail_code = try_occupy(desk, a.letter)
        final_reason = ('auto_verified' if ok and auto
                        else reason if ok else fail_code)
        outcome.routes.append(Route(
            a.letter, a.topic, desk, auto,
            'received' if ok else 'waiting', final_reason, reason,
            _prob(pred), False if pred is None else pred.oov,
            letter.urgent, letter.arrived))
    return outcome
