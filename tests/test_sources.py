"""核对业务资料的格式与可关联性。"""
import csv
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parents[1] / 'data'
TABLES = {'letters': ('letter', 'arrived', 'channel', 'expression', 'urgent', 'text'), 'forwarding': ('copy', 'parent', 'known', 'state'), 'labels': ('letter', 'topic', 'policy', 'known', 'verdict'), 'topics': ('policy', 'topic', 'parent', 'starts', 'desk'), 'limits': ('desk', 'known', 'from_day', 'through', 'quota', 'tolerance', 'min_support'), 'assignments': ('letter', 'topic', 'desk', 'issued', 'ack', 'state'), 'windows': ('name', 'train_before', 'select_from', 'select_to', 'audit_from', 'audit_to', 'fresh_from')}
RELATION = ('assignments', 'letter', 'letters', 'letter')


class SourceContract(unittest.TestCase):
    def read(self, table):
        with (DATA / f'{table}.csv').open(encoding='utf-8', newline='') as file:
            return list(csv.DictReader(file))

    def test_shape(self):
        for name, columns in TABLES.items():
            with self.subTest(table=name):
                rows = self.read(name)
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(tuple(row), columns)
                    self.assertNotIn(None, row.values())

    def test_references(self):
        left, left_key, right, right_key = RELATION
        self.assertLessEqual({row[left_key] for row in self.read(left)}, {row[right_key] for row in self.read(right)})

    def test_source_variation(self):
        self.assertEqual({row['topic'] for row in self.read('topics') if row['parent'] == 'claim'}, {'identity', 'promise'})
        self.assertEqual({row['state'] for row in self.read('assignments') if row['letter'] == 'x241'}, {'received', 'waiting'})
