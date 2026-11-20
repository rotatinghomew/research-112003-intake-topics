"""时点标签视图与 PU（正例-未标注）训练样本构造。

核心口径：
- 只采信 known 日已可见、且未被同项 retracted 撤销的人工结论；
- yes 是正例、明确 no 是负例；标签部门没复核的主题保持“未知”，
  绝不作为负例（资料中只复核一项的来件其唯一结论全部为 yes）；
- 结论自带 policy 版本，训练 v2 新子类时不允许借用 v1 上位类旧判读。
"""
from .dataio import DataBundle, LabelRow


class LabelView:
    def __init__(self, labels: list[LabelRow]):
        self._labels = labels

    def visible(self, day: int) -> list[LabelRow]:
        """day 日（含）可见且仍然有效的逐项结论。"""
        # 先收集该时点前的撤销键
        retracted = {
            (r.letter, r.topic)
            for r in self._labels
            if r.known <= day and r.verdict == 'retracted'
        }
        return [
            r for r in self._labels
            if r.known <= day and r.verdict in ('yes', 'no')
            and (r.letter, r.topic) not in retracted
        ]

    def verdicts_by_letter(self, day: int) -> dict[str, dict[str, tuple[str, str]]]:
        """letter -> topic -> (verdict, policy)。"""
        out: dict[str, dict[str, tuple[str, str]]] = {}
        for r in self.visible(day):
            out.setdefault(r.letter, {})[r.topic] = (r.verdict, r.policy)
        return out


def topic_samples(bundle: DataBundle, view: LabelView, day: int, policy: str, topic: str):
    """返回某主题在某 policy 下的 (正例来件, 负例来件)。

    只收该 policy、该主题上有明确人工结论的来件；无结论者不参与（PU）。
    """
    positives, negatives = [], []
    for r in view.visible(day):
        if r.policy != policy or r.topic != topic:
            continue
        if r.verdict == 'yes':
            positives.append(r.letter)
        elif r.verdict == 'no':
            negatives.append(r.letter)
    return sorted(positives), sorted(negatives)


def all_topics_for_policy(bundle: DataBundle, policy: str) -> list[str]:
    return sorted({r.topic for r in bundle.topics if r.policy == policy})
