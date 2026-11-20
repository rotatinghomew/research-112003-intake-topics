"""评估：按时间折叠的独立复核、漏判/误判切片、桌级容忍度与覆盖。

指标口径（全部只在有人工逐项结论的主题上计算，空缺不参与）：
- 漏判 miss：人工 yes，但模型未达自动分送条件（阈值不足或 OOV 拦截）；
- 误送 misroute：模型自动分送，但人工结论为 no；
- 覆盖 coverage：自动分送的 yes 数 / 全部 yes 数；
- 容忍度判断用误送率的 Wilson 单侧 95% 上界，而非点估计，避免小样本乐观偏差；
- 切片：模板/自由写法、单主题/多主题来件分别报告。
"""
from __future__ import annotations
import math

from .features import split_segments
from .taxonomy import Taxonomy

_NO_THRESHOLD = 1.01  # 未选型主题：任何概率都达不到，等同不自动送件


def wilson_upper(k: int, n: int, z: float = 1.645) -> float:
    """误送率的 Wilson 单侧上界（95% 置信，z=1.645）。n=0 时返回 1.0。"""
    if n == 0:
        return 1.0
    phat = k / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / (1 + z * z / n))


class TopicScore:
    def __init__(self, policy: str, topic: str, method: str):
        self.policy = policy
        self.topic = topic
        self.method = method
        self.tp = self.fp = self.fn = self.tn = 0
        self.oov_blocks = 0          # 因未登录表达被拦截的 yes（计入漏判并单列）
        self.slices: dict[str, list[int]] = {}  # slice -> [tp, fp, fn, tn]
        self.examples: list[dict] = []

    @property
    def support_pos(self) -> int:
        return self.tp + self.fn

    @property
    def support_neg(self) -> int:
        return self.fp + self.tn

    @property
    def miss_rate(self) -> float | None:
        return self.fn / self.support_pos if self.support_pos else None

    @property
    def misroute_rate(self) -> float | None:
        sent = self.tp + self.fp
        return self.fp / sent if sent else None

    @property
    def coverage(self) -> float | None:
        return self.tp / self.support_pos if self.support_pos else None

    @property
    def misroute_upper(self) -> float:
        return wilson_upper(self.fp, self.tp + self.fp)

    def as_row(self) -> dict:
        return {
            'policy': self.policy, 'topic': self.topic, 'method': self.method,
            'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn,
            'support_pos': self.support_pos, 'support_neg': self.support_neg,
            'miss_rate': _fmt(self.miss_rate),
            'misroute_rate': _fmt(self.misroute_rate),
            'misroute_upper95': round(self.misroute_upper, 4),
            'coverage': _fmt(self.coverage), 'oov_blocks': self.oov_blocks,
        }


def _fmt(x):
    return None if x is None else round(x, 4)


def slice_rows(scores: dict[str, 'TopicScore'], window: str, phase: str) -> list[dict]:
    """把各主题的模板/自由写法 × 单/多主题切片计数展开成行。"""
    rows = []
    for key, sc in sorted(scores.items()):
        for slice_name, (tp, fp, fn, tn) in sorted(sc.slices.items()):
            expression, multi = slice_name.split(':', 1)
            pos, sent = tp + fn, tp + fp
            rows.append({
                'window': window, 'phase': phase, 'policy': sc.policy,
                'topic': sc.topic, 'method': sc.method,
                'expression': expression, 'multi_topic': multi,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'miss_rate': _fmt(fn / pos) if pos else None,
                'misroute_rate': _fmt(fp / sent) if sent else None,
                'coverage': _fmt(tp / pos) if pos else None,
            })
    return rows


def _slice_key(letter) -> str:
    multi = '多主题' if '；' in letter.text else '单主题'
    return f'{letter.expression}:{multi}'


def predict_letter_topics(trained: dict, letter, policy: str, method: str):
    """对一封来件逐请求段预测，返回 topic -> 聚合 Prediction（同主题取最大概率）。

    此处只产出概率与 OOV 判断，阈值闸门由调用方按选型结果施加。
    """
    out: dict = {}
    for seg in split_segments(letter.text):
        for tkey, tt in trained.items():
            if tt.policy != policy:
                continue
            model = tt.models.get(method)
            if model is None:
                continue
            pred = model.predict_segment(seg)
            if pred.prob is None:
                continue  # 规则模型对未命中段弃权
            prev = out.get(tt.topic)
            if prev is None or pred.prob > prev.prob:
                out[tt.topic] = pred
    return out


def score_method(trained: dict, method: str, thresholds: dict[str, float],
                 letters: dict, labels_by_letter: dict, taxonomy: Taxonomy,
                 collect_examples: int = 6) -> dict[str, TopicScore]:
    """在一个评估折叠上打分。thresholds: 'policy:topic' -> 阈值。"""
    scores: dict[str, TopicScore] = {}

    def get(policy, topic):
        key = f'{policy}:{topic}'
        if key not in scores:
            scores[key] = TopicScore(policy, topic, method)
        return scores[key]

    for lid, letter in letters.items():
        marks = labels_by_letter.get(lid, {})
        if not marks:
            continue
        policy = taxonomy.policy_on(letter.arrived)
        decisions = predict_letter_topics(trained, letter, policy, method)
        for topic, (verdict, vpolicy) in marks.items():
            if vpolicy != policy:
                continue
            sc = get(policy, topic)
            sl = _slice_key(letter)
            sc.slices.setdefault(sl, [0, 0, 0, 0])
            pred = decisions.get(topic)
            thr = thresholds.get(f'{policy}:{topic}', _NO_THRESHOLD)
            if thr is None:
                # 规则模型阈值：弃权（无预测）即不送件，命中即按其高精度先验送件
                auto = pred is not None and not pred.oov
            else:
                auto = bool(pred is not None and pred.prob >= thr and not pred.oov)
            if verdict == 'yes':
                if auto:
                    sc.tp += 1
                    sc.slices[sl][0] += 1
                else:
                    sc.fn += 1
                    sc.slices[sl][2] += 1
                    if pred is not None and pred.oov:
                        sc.oov_blocks += 1
                if len(sc.examples) < collect_examples and (
                        verdict == 'yes' and not auto):
                    sc.examples.append({'letter': lid, 'kind': 'miss',
                                        'prob': None if pred is None else round(pred.prob, 3),
                                        'oov': None if pred is None else pred.oov,
                                        'coverage': None if pred is None else round(pred.coverage, 3),
                                        'reasons': [] if pred is None else pred.reasons[:3]})
            else:  # 明确 no
                if auto:
                    sc.fp += 1
                    sc.slices[sl][1] += 1
                    if len(sc.examples) < collect_examples:
                        sc.examples.append({'letter': lid, 'kind': 'misroute',
                                            'prob': round(pred.prob, 3),
                                            'oov': pred.oov,
                                            'coverage': round(pred.coverage, 3),
                                            'reasons': pred.reasons[:3]})
                else:
                    sc.tn += 1
                    sc.slices[sl][3] += 1
    return scores


def grid_thresholds(trained: dict, method: str, letters: dict,
                    labels_by_letter: dict, taxonomy: Taxonomy,
                    grid=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> dict[str, float]:
    """只在选型折叠上定阈值（容忍度闸门留给独立复核段）：

    - 规则模型弃权语义：不命中即不送件，阈值固定为 None，只评估其覆盖；
    - 概率模型：在选型折叠上取误送点估计为零的最低阈值（保守），
      若所有阈值都有误送则取误送数最少者中覆盖最高的；
    - 没有正例证据的主题不产出阈值。
    """
    chosen: dict[str, float] = {}
    for tkey, tt in trained.items():
        if tt.models.get(method) is None:
            continue
        policy, topic = tkey.split(':', 1)
        desk = taxonomy.desk_for(policy, topic)
        if desk is None:
            continue  # 上位类不自动定桌
        if method == 'rule':
            # 规则模型：有高精度短语即可，命中与否本身就是开关
            if tt.models['rule'].phrase_priors:
                chosen[tkey] = None
            continue
        best, best_key = None, None
        for thr in grid:
            sc = score_method(trained, method, {tkey: thr}, letters,
                              labels_by_letter, taxonomy, collect_examples=0).get(tkey)
            if sc is None or sc.support_pos == 0 or (sc.tp + sc.fp) == 0:
                continue
            # 优先误送为 0，再看覆盖；同为 0 误送取最低阈值（覆盖最大）
            key = (sc.fp, -sc.coverage)
            if best_key is None or key < best_key:
                best_key, best = key, thr
        if best is not None:
            chosen[tkey] = best
    return chosen
