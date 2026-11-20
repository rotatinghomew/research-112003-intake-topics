"""特征抽取：字符 n-gram、请求段切分、未登录表达（OOV）检测。

设计上把一封来件拆成若干“请求段”（分号分隔），分类在段级别进行，
一封信因此可以携带多个主题且互不吞没。模板前缀是渠道固定话术，
不是业务表达，先剥离再抽取。
"""
from .taxonomy import TEMPLATE_PREFIX


def split_segments(text: str) -> list[str]:
    body = text
    if body.startswith(TEMPLATE_PREFIX):
        body = body[len(TEMPLATE_PREFIX):]
    return [seg.strip() for seg in body.split('；') if seg.strip()]


def char_ngrams(segment: str, n_min: int = 2, n_max: int = 3) -> list[str]:
    feats: list[str] = []
    for n in range(n_min, n_max + 1):
        if len(segment) >= n:
            feats.extend(segment[i:i + n] for i in range(len(segment) - n + 1))
    return feats


class FeatureVocab:
    """只从训练语料建立词表；评估/在途来件里的未见 gram 自然不计入。"""

    def __init__(self, n_min: int = 2, n_max: int = 3):
        self.n_min = n_min
        self.n_max = n_max
        self.index: dict[str, int] = {}
        self.train_grams: set[str] = set()

    def fit(self, segments: list[str]) -> 'FeatureVocab':
        for seg in segments:
            for g in char_ngrams(seg, self.n_min, self.n_max):
                self.train_grams.add(g)
                if g not in self.index:
                    self.index[g] = len(self.index)
        return self

    def vector(self, segment: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for g in char_ngrams(segment, self.n_min, self.n_max):
            idx = self.index.get(g)
            if idx is not None:
                out[idx] = out.get(idx, 0.0) + 1.0
        return out

    def coverage(self, segment: str) -> float:
        """段内 gram 在训练语料出现过的比例；新写法该值显著偏低。"""
        grams = char_ngrams(segment, self.n_min, self.n_max)
        if not grams:
            return 0.0
        return sum(1 for g in grams if g in self.train_grams) / len(grams)
