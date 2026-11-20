"""目录（主题体系）版本与承办桌映射。

规则要点：
- 来件按到达日适用当时生效的 policy 版本；
- v2 把 v1 的 claim 拆成 identity/promise，v2 下 claim 仅为上位类、不挂桌，
  旧判读只有上位类证据，不能强行指定新桌面（parent-only 禁送）；
- refund/contact 在两个版本下承办桌不变，证据可以沿时间累积。
"""
from .dataio import TopicRow

ROOT = 'issues'
MANUAL_DESK = 'manual'
TEMPLATE_PREFIX = '麻烦帮我查这件事，'


class Taxonomy:
    def __init__(self, topic_rows: list[TopicRow]):
        self._rows = topic_rows
        self.policies = sorted({r.policy for r in topic_rows},
                               key=lambda p: min(r.starts for r in topic_rows if r.policy == p))
        # policy -> 生效日
        self.starts = {p: min(r.starts for r in topic_rows if r.policy == p)
                       for p in self.policies}
        # policy -> topic -> desk（空串表示不挂桌）
        self.desk_of: dict[str, dict[str, str]] = {p: {} for p in self.policies}
        self.parent_of: dict[str, dict[str, str]] = {p: {} for p in self.policies}
        for r in topic_rows:
            self.desk_of[r.policy][r.topic] = r.desk
            self.parent_of[r.policy][r.topic] = r.parent
        # 每个主题的全量历史版本主题集合（用于证据继承判断）
        self.topics = sorted({r.topic for r in topic_rows})

    def policy_on(self, day: int) -> str:
        """day 当天适用的政策版本（starts 当日即生效）。"""
        active = self.policies[0]
        for p in self.policies:
            if self.starts[p] <= day:
                active = p
        return active

    def desk_for(self, policy: str, topic: str) -> str | None:
        """返回承办桌；None 表示该主题在此版本下不能自动定桌。"""
        desk = self.desk_of[policy].get(topic)
        return desk or None

    def is_parent_only(self, policy: str, topic: str) -> bool:
        """该主题在该版本下是否只是上位类（有记录但不挂桌）。"""
        return topic in self.desk_of[policy] and not self.desk_of[policy][topic]

    def children_of(self, policy: str, parent: str) -> list[str]:
        return sorted(t for t, par in self.parent_of[policy].items() if par == parent)
