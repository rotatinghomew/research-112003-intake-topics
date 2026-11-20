"""转发关系分组，防止同一来件换渠道泄漏到验证集。

根来件由 forwarding.csv 显式登记，不由文本是否相同判断。
组的评估折叠取组内成员最晚到达的折叠：只要组里有一件落在评估段，
整组都不进入任何更早的训练/选型段（最严格口径，宁可少训练不泄漏）。
"""
from .dataio import DataBundle, ForwardRow


class ForwardGroups:
    def __init__(self, rows: list[ForwardRow], letters):
        self._parent: dict[str, str] = {}
        for r in rows:
            if r.state != 'confirmed':
                continue
            self._parent[r.copy] = r.parent
        self.letters = letters
        # 根 -> 成员
        self.members: dict[str, list[str]] = {}
        for lid in letters:
            root = self.root_of(lid)
            self.members.setdefault(root, []).append(lid)
        # 关系确认日（用于时点说明），copy -> known
        self.known_day = {r.copy: r.known for r in rows if r.state == 'confirmed'}

    def root_of(self, letter_id: str) -> str:
        cur = letter_id
        seen = set()
        while cur in self._parent and cur not in seen:
            seen.add(cur)
            cur = self._parent[cur]
        return cur

    def fold_of_group(self, letter_id: str, fold_rank: dict[str, int]) -> int:
        """fold_rank: 来件 -> 折叠序号（数字越大越晚）。返回组内最大折叠号。"""
        root = self.root_of(letter_id)
        return max(fold_rank[m] for m in self.members[root])

    def chains(self) -> list[list[str]]:
        return [sorted(m, key=lambda x: self.letters[x].arrived)
                for m in self.members.values() if len(m) > 1]


def build_groups(bundle: DataBundle) -> ForwardGroups:
    return ForwardGroups(bundle.forwarding, bundle.letters)


def time_folds(bundle: DataBundle) -> dict[str, int]:
    """保留作调试用的粗折叠；正式模拟改用 window_roles 按窗口独立定角色。

    滚动窗口时间段重叠：同一来件在首窗可能是复核件、在次窗又是选型件，
    单一折叠号无法表达双重角色，因此角色判定必须按窗口分别进行。
    """
    w1, w2 = bundle.windows
    rank: dict[str, int] = {}
    for lid, lt in bundle.letters.items():
        d = lt.arrived
        if d < w1.train_before:
            rank[lid] = 0
        elif w1.select_from <= d <= w1.select_to:
            rank[lid] = 1
        elif w1.audit_from <= d <= w1.audit_to:
            rank[lid] = 2
        elif w2.select_from <= d <= w2.select_to:
            rank[lid] = 4
        elif w2.audit_from <= d <= w2.audit_to:
            rank[lid] = 5
        else:
            rank[lid] = 6
    return rank


# 窗口内角色
ROLE_TRAIN = 'train'
ROLE_SELECT = 'select'
ROLE_AUDIT = 'audit'
ROLE_FRESH = 'fresh'
ROLE_OUTSIDE = 'outside'


def window_roles(bundle: DataBundle) -> dict[str, dict[str, str]]:
    """window_name -> letter -> 该窗口下的角色。

    每个窗口独立按自身边界判定，允许同一来件在不同窗口承担不同角色。
    """
    roles: dict[str, dict[str, str]] = {}
    for w in bundle.windows:
        per: dict[str, str] = {}
        for lid, lt in bundle.letters.items():
            d = lt.arrived
            if d < w.train_before:
                per[lid] = ROLE_TRAIN
            elif w.select_from <= d <= w.select_to:
                per[lid] = ROLE_SELECT
            elif w.audit_from <= d <= w.audit_to:
                per[lid] = ROLE_AUDIT
            elif d >= w.fresh_from:
                per[lid] = ROLE_FRESH
            else:
                # 训练截点后、选型前等无定义区间（本资料无），不参与该窗口
                per[lid] = ROLE_OUTSIDE
        roles[w.name] = per
    return roles


def group_aware_members(groups: ForwardGroups, roles: dict[str, str],
                        role: str) -> tuple[list[str], list[list[str]]]:
    """按组感知返回某角色成员；整组只要有一件属于该角色，整组归入，
    同时返回被整体吸收的多成员组（用于泄漏审计）。"""
    member_ids: list[str] = []
    absorbed: list[list[str]] = []
    for root, members in groups.members.items():
        roles_here = {roles.get(m, ROLE_OUTSIDE) for m in members}
        if role in roles_here:
            member_ids.extend(members)
            if len(members) > 1:
                absorbed.append(sorted(members))
    return sorted(member_ids), absorbed
