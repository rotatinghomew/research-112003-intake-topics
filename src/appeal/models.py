"""三个候选分类器，全部本地实现、可解释，不依赖外部模型服务。

候选办法（每个主题独立训练一个二分类器，天然支持多标签来件）：
1. rule    人工维护的短语/词典规则：透明、可审计，作为强基线；
2. nbayes  字符 n-gram 多项式朴素贝叶斯：小样本稳健、输出带加权证据；
3. logreg  字符 n-gram 逻辑回归（批量梯度下降）：给出真正决定分案的特征权重。

训练样本严格 PU：只有明确 yes/no 的来件进入训练，未复核主题不充负例。
所有分类器同时给出“未登录表达”提示，OOV 时不允许高置信自动分送。
"""
from __future__ import annotations
import math

from .features import FeatureVocab, split_segments

RULE = 'rule'
NBAYES = 'nbayes'
LOGREG = 'logreg'
CANDIDATES = (RULE, NBAYES, LOGREG)


class Prediction:
    __slots__ = ('topic', 'prob', 'oov', 'coverage', 'reasons')

    def __init__(self, topic, prob, oov, coverage, reasons):
        self.topic = topic
        self.prob = prob          # 该主题成立的估计概率
        self.oov = oov            # 是否判定为未登录表达
        self.coverage = coverage  # gram 覆盖率
        self.reasons = reasons    # 可解释证据 [(描述, 贡献), ...]


# ---------------------------------------------------------------- 规则词典
class RuleModel:
    """短语规则：关键词命中即给正例证据；规则由训练标签自动挖掘并人工可读。

    挖掘口径：在训练段里统计每个候选短语带某主题 yes 标签的条件占比，
    只保留高精度短语；规则文件可读、可由承办员修订。
    """

    name = RULE

    def __init__(self, topic: str, phrase_priors: dict[str, float],
                 vocab: FeatureVocab, oov_threshold: float):
        self.topic = topic
        self.phrase_priors = phrase_priors  # 短语 -> 训练集中该主题正例率
        self.vocab = vocab
        self.oov_threshold = oov_threshold

    def predict_segment(self, segment: str) -> Prediction:
        hits = [(p, prior) for p, prior in self.phrase_priors.items() if p in segment]
        if hits:
            prob = max(prior for _, prior in hits)
            reasons = [(f'命中规则短语「{p}」', prior) for p, prior in hits]
        else:
            # 无规则命中表示“本规则不支持”，而不是“主题成立概率为 0”：
            # 规则模型不产出负概率，未命中段在多段聚合时被忽略。
            prob = None
            reasons = [('无规则短语命中（规则模型弃权）', 0.0)]
        coverage = self.vocab.coverage(segment)
        oov = coverage < self.oov_threshold
        return Prediction(self.topic, prob, oov, coverage, reasons)


def mine_phrase_rules(topic: str, train_letters, labels_by_letter, policy: str,
                      min_precision: float = 0.8) -> dict[str, float]:
    """从训练来件挖掘高精度短语。labels_by_letter: letter -> topic -> (verdict, policy)。"""
    stats: dict[str, list[int]] = {}
    for lid, lt in train_letters.items():
        marked = labels_by_letter.get(lid, {}).get(topic)
        if marked is None or marked[1] != policy:
            continue  # PU：该主题未复核则不参与
        positive = 1 if marked[0] == 'yes' else 0
        for seg in split_segments(lt.text):
            entry = stats.setdefault(seg, [0, 0])
            entry[0] += positive
            entry[1] += 1
    return {p: pos / total for p, (pos, total) in stats.items()
            if total > 0 and pos / total >= min_precision}


# ---------------------------------------------------------------- 朴素贝叶斯
class NaiveBayes:
    name = NBAYES

    def __init__(self, topic: str, vocab: FeatureVocab, pos_params, neg_params,
                 prior_pos: float, oov_threshold: float):
        self.topic = topic
        self.vocab = vocab
        self.log_pos, self.pos_denom = pos_params
        self.log_neg, self.neg_denom = neg_params
        self.prior_pos = prior_pos
        self.oov_threshold = oov_threshold

    def predict_segment(self, segment: str) -> Prediction:
        vec = self.vocab.vector(segment)
        log_p = math.log(self.prior_pos)
        log_n = math.log(1.0 - self.prior_pos)
        reasons = []
        for idx, count in vec.items():
            wp = count * self.log_pos[idx]
            wn = count * self.log_neg[idx]
            log_p += wp
            log_n += wn
            gram = self.vocab_index_gram(idx)
            reasons.append((f'gram「{gram}」×{int(count)}', wp - wn))
        # 数值稳定归一化
        m = max(log_p, log_n)
        p_pos = math.exp(log_p - m) / (math.exp(log_p - m) + math.exp(log_n - m))
        coverage = self.vocab.coverage(segment)
        reasons.sort(key=lambda x: abs(x[1]), reverse=True)
        return Prediction(self.topic, p_pos, coverage < self.oov_threshold,
                          coverage, reasons[:6])

    def vocab_index_gram(self, idx: int) -> str:
        for g, i in self.vocab.index.items():
            if i == idx:
                return g
        return '?'


def train_naive_bayes(topic: str, pos_segments: list[str], neg_segments: list[str],
                      vocab: FeatureVocab, alpha: float = 1.0,
                      oov_threshold: float = 0.6) -> NaiveBayes:
    size = len(vocab.index)
    pos_counts = [0.0] * size
    neg_counts = [0.0] * size
    for segs, target in ((pos_segments, pos_counts), (neg_segments, neg_counts)):
        for seg in segs:
            for idx, count in vocab.vector(seg).items():
                target[idx] += count
    pos_denom = sum(pos_counts) + alpha * size
    neg_denom = sum(neg_counts) + alpha * size
    log_pos = [math.log((c + alpha) / pos_denom) for c in pos_counts]
    log_neg = [math.log((c + alpha) / neg_denom) for c in neg_counts]
    n_pos, n_neg = len(pos_segments), len(neg_segments)
    prior = (n_pos + 1.0) / (n_pos + n_neg + 2.0) if n_pos + n_neg else 0.5
    return NaiveBayes(topic, vocab, (log_pos, pos_denom), (log_neg, neg_denom),
                      prior, oov_threshold)


# ---------------------------------------------------------------- 逻辑回归
class LogisticModel:
    name = LOGREG

    def __init__(self, topic: str, weights: list[float], bias: float,
                 vocab: FeatureVocab, oov_threshold: float, train_count: int):
        self.topic = topic
        self.weights = weights
        self.bias = bias
        self.vocab = vocab
        self.oov_threshold = oov_threshold
        self.train_count = train_count

    def predict_segment(self, segment: str) -> Prediction:
        vec = self.vocab.vector(segment)
        z = self.bias
        reasons = []
        for idx, count in vec.items():
            contrib = self.weights[idx] * count
            z += contrib
            gram = next((g for g, i in self.vocab.index.items() if i == idx), '?')
            reasons.append((f'gram「{gram}」×{int(count)}', contrib))
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        coverage = self.vocab.coverage(segment)
        reasons.sort(key=lambda x: abs(x[1]), reverse=True)
        return Prediction(self.topic, prob, coverage < self.oov_threshold,
                          coverage, [(d, c) for d, c in reasons[:6]])


def train_logreg(topic: str, pos_segments: list[str], neg_segments: list[str],
                 vocab: FeatureVocab, epochs: int = 200, lr: float = 0.15,
                 l2: float = 0.01, oov_threshold: float = 0.6) -> LogisticModel:
    """全批量梯度下降，特征按 L2 归一化长度，固定随机无关、结果可复现。"""
    size = len(vocab.index)
    weights = [0.0] * size
    bias = 0.0
    data = [(vocab.vector(seg), 1.0) for seg in pos_segments]
    data += [(vocab.vector(seg), 0.0) for seg in neg_segments]
    # L2 归一化
    normed = []
    for vec, y in data:
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        normed.append(({k: v / norm for k, v in vec.items()}, y))
    n = max(1, len(normed))
    for _ in range(epochs):
        grad_w = [0.0] * size
        grad_b = 0.0
        for vec, y in normed:
            z = bias + sum(weights[k] * v for k, v in vec.items())
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - y
            grad_b += err
            for k, v in vec.items():
                grad_w[k] += err * v
        for k in range(size):
            weights[k] -= lr * (grad_w[k] / n + l2 * weights[k])
        bias -= lr * grad_b / n
    return LogisticModel(topic, weights, bias, vocab, oov_threshold, len(normed))
