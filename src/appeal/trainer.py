"""按时点训练各主题分类器。

训练纪律：
- 只允许使用训练折叠内来件（转发整组折叠由 groups 模块统一裁定）；
- 只采信 train_day 已可见且未撤销的人工结论；
- PU 采样：有明确 yes/no 才进训练，未复核主题不充负例；
- 多主题来件按请求段做弱归因：能被高精度规则短语归属的段只归该主题，
  归属不了时回退为该来件全部 yes 主题共享，并在报告中注明这是弱标注；
- v2 新子类只用 v2 结论训练，v1 claim 旧判读不进入 identity/promise。
"""
from __future__ import annotations

from .dataio import DataBundle, Letter
from .features import FeatureVocab, split_segments
from .labels_view import LabelView
from .models import (LOGREG, NBAYES, RULE, RuleModel,
                     mine_phrase_rules, train_logreg, train_naive_bayes)
from .taxonomy import Taxonomy

DEFAULT_OOV = 0.9
# 依据 analysis/calibrate_oov.py：训练语料内标准短语覆盖率均为 1.0，
# 已知新表达（x241 两段）覆盖率仅 0.05/0.0，0.9 位于两者之间且有充分安全边界。


class TrainedTopic:
    """一个主题在一个 policy 下、三个候选办法的成品模型与阈值占位。"""

    def __init__(self, topic: str, policy: str, models: dict, vocab: FeatureVocab,
                 n_pos: int, n_neg: int, oov_threshold: float,
                 evidence_sources: list[str] | None = None):
        self.topic = topic
        self.policy = policy
        self.models = models
        self.vocab = vocab
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.oov_threshold = oov_threshold
        # 证据血缘，例如 ['v2:refund', 'v1:refund(继承)']，供报告审计
        self.evidence_sources = evidence_sources or []
        # 选型阶段确定，初始为 None
        self.thresholds: dict[str, float] = {}


def _segments_for_topic(topic: str, policy: str, letter: Letter,
                        labels_by_letter: dict, rules: dict[str, float]) -> list[str]:
    """把一封 yes 来件的请求段归因到指定主题（多示例弱标注）。"""
    segs = split_segments(letter.text)
    if len(segs) == 1:
        return segs
    matched = [s for s in segs if any(p in s for p in rules)]
    if matched:
        own = [s for s in matched if any(p in s for p in rules)]
        return own if own else segs
    return segs  # 无规则可依的回退：整封弱正例


def train_window(bundle: DataBundle, taxonomy: Taxonomy, view: LabelView,
                 train_day: int, train_letters: dict[str, Letter],
                 oov_threshold: float = DEFAULT_OOV) -> dict[str, TrainedTopic]:
    """train_day: 训练截点（只用 known <= train_day 的结论）。"""
    labels_by_letter = view.verdicts_by_letter(train_day)
    trained: dict[str, TrainedTopic] = {}

    # 共享词表：OOV 覆盖率必须相对“训练语料里出现过的全部表达”计算，
    # 用单主题小词表会让通用汉字 bigram 被误判为未登录。
    all_segments = [seg for lt in train_letters.values()
                    for seg in split_segments(lt.text)]
    shared_vocab = FeatureVocab().fit(all_segments)
    # 当前及历史 policy 的全部主题都可能需要模型（旧来件在评估段仍按其 policy 评估）
    for policy in taxonomy.policies:
        for topic in taxonomy.topics:
            if topic not in taxonomy.desk_of[policy]:
                continue
            pos_ids, neg_ids, sources = [], [], []
            for lid, lt in train_letters.items():
                mark = labels_by_letter.get(lid, {}).get(topic)
                if mark is None:
                    continue  # PU：未复核不进训练
                verdict, vpolicy = mark
                if vpolicy == policy:
                    (pos_ids if verdict == 'yes' else neg_ids).append(lid)
                elif (vpolicy != policy and verdict == 'yes'
                      and _inheritable(taxonomy, vpolicy, policy, topic)):
                    # 仅当主题同名且承办桌两版未变时，旧版正例可作证据延续；
                    # 新子类（identity/promise）不满足继承条件，绝不借用 v1 claim。
                    pos_ids.append(lid)
                    if f'{vpolicy}:{topic}(继承)' not in sources:
                        sources.append(f'{vpolicy}:{topic}(继承)')
            sources.insert(0, f'{policy}:{topic}')

            # 先在训练语料上挖规则，再做段归因
            rules = mine_phrase_rules(topic, train_letters, labels_by_letter, policy)
            # 继承场景下规则可能为空（新版暂无标签），仅当证据可继承时补旧版规则
            if not rules and _inheritable(taxonomy, _previous_policy(taxonomy, policy),
                                          policy, topic):
                rules = mine_phrase_rules(topic, train_letters, labels_by_letter,
                                          _previous_policy(taxonomy, policy))
            pos_segments, neg_segments = [], []
            for lid in pos_ids:
                pos_segments += _segments_for_topic(
                    topic, policy, train_letters[lid], labels_by_letter, rules)
            for lid in neg_ids:
                # no 来件的所有段都是该主题的反例证据（别的主题短语恰好证明“不是本主题”）
                neg_segments += split_segments(train_letters[lid].text)

            # 模型向量与 OOV 覆盖率都建立在共享词表上
            vocab = shared_vocab
            models: dict = {}
            models[RULE] = RuleModel(topic, rules, vocab, oov_threshold)
            if pos_segments and neg_segments:
                models[NBAYES] = train_naive_bayes(
                    topic, pos_segments, neg_segments, vocab,
                    oov_threshold=oov_threshold)
                models[LOGREG] = train_logreg(
                    topic, pos_segments, neg_segments, vocab,
                    oov_threshold=oov_threshold)
            else:
                # 证据不足以训练判别模型（如新子类早期零负例）：只保留规则，显式标注不可自动
                models[NBAYES] = None
                models[LOGREG] = None
            trained[f'{policy}:{topic}'] = TrainedTopic(
                topic, policy, models, vocab, len(pos_ids), len(neg_ids),
                oov_threshold, evidence_sources=sources)
    return trained


def _previous_policy(taxonomy: Taxonomy, policy: str) -> str:
    idx = taxonomy.policies.index(policy)
    return taxonomy.policies[idx - 1] if idx > 0 else policy


def _inheritable(taxonomy: Taxonomy, old_policy: str, new_policy: str,
                  topic: str) -> bool:
    """旧版同主题证据能否延续到新版：主题在两版都存在且承办桌一致（非空）。

    v1 claim -> v2 identity/promise 这类拆分不满足条件：
    旧判读只有上位主题证据，不能强行指定新桌面。
    """
    if topic not in taxonomy.desk_of[old_policy] or topic not in taxonomy.desk_of[new_policy]:
        return False
    old_desk = taxonomy.desk_for(old_policy, topic)
    new_desk = taxonomy.desk_for(new_policy, topic)
    return old_desk is not None and old_desk == new_desk
