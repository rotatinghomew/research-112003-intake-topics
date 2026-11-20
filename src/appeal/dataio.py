"""CSV 资料载入与不可变数据记录。

空字段保持为空字符串（业务上空表示“没有记录”，不是零）。
所有整数字段在边界处统一转换，避免业务代码散落 int() 调用。
"""
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class Letter:
    letter: str
    arrived: int
    channel: str
    expression: str  # template / free
    urgent: bool
    text: str


@dataclass(frozen=True)
class LabelRow:
    letter: str
    topic: str
    policy: str
    known: int
    verdict: str  # yes / no / retracted


@dataclass(frozen=True)
class ForwardRow:
    copy: str
    parent: str
    known: int
    state: str


@dataclass(frozen=True)
class TopicRow:
    policy: str
    topic: str
    parent: str
    starts: int
    desk: str  # 空字符串表示该版本下不直接挂承办桌


@dataclass(frozen=True)
class LimitRow:
    desk: str
    known: int
    from_day: int
    through: int
    quota: int
    tolerance: float | None  # manual 人工补读桌没有误送率口径
    min_support: int | None


@dataclass(frozen=True)
class AssignmentRow:
    letter: str
    topic: str
    desk: str
    issued: int
    ack: int | None
    state: str  # received / waiting


@dataclass(frozen=True)
class Window:
    name: str
    train_before: int
    select_from: int
    select_to: int
    audit_from: int
    audit_to: int
    fresh_from: int


@dataclass(frozen=True)
class DataBundle:
    letters: dict[str, Letter]
    labels: list[LabelRow]
    forwarding: list[ForwardRow]
    topics: list[TopicRow]
    limits: list[LimitRow]
    assignments: list[AssignmentRow]
    windows: list[Window]


def _int(value: str) -> int:
    return int(value)


def _opt_int(value: str) -> int | None:
    return int(value) if value else None


def _opt_float(value: str) -> float | None:
    return float(value) if value else None


def load_bundle(data_dir: str | Path) -> DataBundle:
    data_dir = Path(data_dir)

    def read(name: str) -> list[dict[str, str]]:
        with (data_dir / f'{name}.csv').open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    letters = {
        r['letter']: Letter(
            letter=r['letter'], arrived=_int(r['arrived']), channel=r['channel'],
            expression=r['expression'], urgent=r['urgent'] == '1', text=r['text'],
        )
        for r in read('letters')
    }
    labels = [
        LabelRow(r['letter'], r['topic'], r['policy'], _int(r['known']), r['verdict'])
        for r in read('labels')
    ]
    forwarding = [
        ForwardRow(r['copy'], r['parent'], _int(r['known']), r['state'])
        for r in read('forwarding')
    ]
    topics = [
        TopicRow(r['policy'], r['topic'], r['parent'], _int(r['starts']), r['desk'])
        for r in read('topics')
    ]
    limits = [
        LimitRow(r['desk'], _int(r['known']), _int(r['from_day']), _int(r['through']),
                 _int(r['quota']), _opt_float(r['tolerance']), _opt_int(r['min_support']))
        for r in read('limits')
    ]
    assignments = [
        AssignmentRow(r['letter'], r['topic'], r['desk'], _int(r['issued']),
                      _opt_int(r['ack']), r['state'])
        for r in read('assignments')
    ]
    windows = [
        Window(r['name'], _int(r['train_before']), _int(r['select_from']),
               _int(r['select_to']), _int(r['audit_from']), _int(r['audit_to']),
               _int(r['fresh_from']))
        for r in read('windows')
    ]
    return DataBundle(letters, labels, forwarding, topics, limits, assignments, windows)
