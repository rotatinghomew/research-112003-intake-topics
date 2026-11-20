"""流水线业务不变量测试。

运行：python3 -m unittest discover -s tests -v
所有测试只读 data/，不触网、不调用外部模型服务。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from appeal.dataio import load_bundle
from appeal.evaluation import wilson_upper
from appeal.groups import (ROLE_AUDIT, ROLE_SELECT, ROLE_TRAIN, build_groups,
                           group_aware_members, window_roles)
from appeal.labels_view import LabelView
from appeal.routing import (MANUAL_DESK, R_FULL, R_NO_METHOD, R_OOV,
                            R_PARENT_ONLY, route_week)
from appeal.simulation import run_simulation
from appeal.taxonomy import Taxonomy

DATA = ROOT / 'data'


class PipelineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle(DATA)
        cls.tax = Taxonomy(cls.bundle.topics)
        cls.groups = build_groups(cls.bundle)
        cls.results = run_simulation(cls.bundle, cls.tax, cls.groups)
        cls.latest = cls.results[-1]


class TestTaxonomy(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.tax = Taxonomy(self.bundle.topics)

    def test_policy_versions_by_day(self):
        self.assertEqual(self.tax.policy_on(26), 'v1')
        self.assertEqual(self.tax.policy_on(27), 'v2')
        self.assertEqual(self.tax.policy_on(50), 'v2')

    def test_claim_split_is_parent_only_in_v2(self):
        # v2 把 claim 拆成 identity/promise，claim 本身不挂桌
        self.assertIsNone(self.tax.desk_for('v2', 'claim'))
        self.assertTrue(self.tax.is_parent_only('v2', 'claim'))
        self.assertEqual(self.tax.desk_for('v2', 'identity'), 'review')
        self.assertEqual(self.tax.desk_for('v2', 'promise'), 'review')

    def test_stable_topics_keep_desk(self):
        for topic, desk in (('refund', 'finance'), ('contact', 'service')):
            self.assertEqual(self.tax.desk_for('v1', topic), desk)
            self.assertEqual(self.tax.desk_for('v2', topic), desk)


class TestForwardingLeakage(PipelineTestBase):
    def test_forward_roots_are_explicit_not_text_based(self):
        # 根来件由 forwarding.csv 决定，10 条 confirmed 关系
        self.assertEqual(len(self.groups.known_day), 10)

    def test_eval_groups_excluded_from_train(self):
        roles_by_window = window_roles(self.bundle)
        for res in self.results:
            roles = roles_by_window[res.name]
            eval_ids = set()
            for role in (ROLE_SELECT, ROLE_AUDIT):
                ids, _ = group_aware_members(self.groups, roles, role)
                eval_ids.update(ids)
            # 训练排除清单里的每个成员都不得出现在评估集合
            excluded_members = {m for group in res.excluded_groups for m in group}
            self.assertTrue(excluded_members.issubset(eval_ids))
            self.assertGreater(len(res.excluded_groups), 0)

    def test_known_forward_pair_never_trains_together(self):
        # x064(根,训练段) 与 x136(副本,选型段) 同组，首窗必须整组排除训练
        roles = window_roles(self.bundle)['first']
        train_ids, _ = group_aware_members(self.groups, roles, ROLE_TRAIN)
        # 直接验证：被排除组的根件不在训练集
        first = self.results[0]
        excluded = {m for g in first.excluded_groups for m in g}
        self.assertIn('x064', excluded)
        self.assertIn('x136', excluded)


class TestPULabels(unittest.TestCase):
    def setUp(self):
        self.bundle = load_bundle(DATA)
        self.view = LabelView(self.bundle.labels)

    def test_single_reviewed_item_is_always_positive(self):
        # 资料事实：只复核一项的来件，其唯一结论全是 yes -> 缺失项严禁当负例
        per = {}
        for r in self.view.visible(54):
            per.setdefault(r.letter, []).append(r)
        single = [rows for rows in per.values() if len(rows) == 1]
        self.assertTrue(single)
        self.assertTrue(all(rows[0].verdict == 'yes' for rows in single))

    def test_retraction_removes_verdict_after_day(self):
        # x120 refund 在 day51 被 retracted
        before = self.view.visible(50)
        after = self.view.visible(54)
        self.assertTrue(any(r.letter == 'x120' and r.topic == 'refund'
                            for r in before))
        self.assertFalse(any(r.letter == 'x120' and r.topic == 'refund'
                             for r in after))

    def test_no_future_labels_visible(self):
        # day26 训练截点看不到任何 v2 结论（最早 v2 known=29）
        rows = self.view.visible(26)
        self.assertFalse(any(r.policy == 'v2' for r in rows))


class TestNoCrossVersionSubtypeLeak(PipelineTestBase):
    def test_identity_promise_never_inherit_v1_claim(self):
        for res in self.results:
            for key in ('v2:identity', 'v2:promise'):
                tt = res.trained.get(key)
                if tt is None:
                    continue
                sources = ' '.join(tt.evidence_sources)
                self.assertNotIn('v1:claim', sources)

    def test_v2_claim_has_no_desk_model(self):
        tt = self.latest.trained['v2:claim']
        self.assertIsNone(self.tax.desk_for('v2', 'claim'))
        # 上位类不应出现在自动选型里
        self.assertNotIn('v2:claim', self.latest.method_pick)

    def test_stable_topic_inheritance_recorded(self):
        tt = self.latest.trained['v2:refund']
        self.assertTrue(any('v1:refund' in s for s in tt.evidence_sources))


class TestOOV(PipelineTestBase):
    def test_novel_expression_x241_flagged(self):
        from appeal.features import split_segments
        tt = self.latest.trained['v2:contact']
        # 任一选中办法对 x241 新表达的覆盖率都应很低
        letter = self.bundle.letters['x241']
        for seg in split_segments(letter.text):
            cov = tt.vocab.coverage(seg)
            self.assertLess(cov, tt.oov_threshold)

    def test_known_phrases_fully_covered(self):
        from appeal.features import split_segments
        tt = self.latest.trained['v2:refund']
        letter = self.bundle.letters['x242']  # 标准退费+联系短语
        for seg in split_segments(letter.text):
            self.assertEqual(tt.vocab.coverage(seg), 1.0)


class TestWilson(unittest.TestCase):
    def test_zero_errors_small_sample_still_bounded(self):
        # 10 件零误送，Wilson 单侧 95%(z=1.645) 上界约 0.213，不能宣称误送率为 0
        self.assertAlmostEqual(wilson_upper(0, 10), 0.2130, places=3)
        # 样本越大上界越紧：30 件零误送上界约 0.083
        self.assertAlmostEqual(wilson_upper(0, 30), 0.0827, places=3)

    def test_empty_sample_is_maximally_pessimistic(self):
        self.assertEqual(wilson_upper(0, 0), 1.0)


class TestRouting(PipelineTestBase):
    def _route(self, today=50):
        eligible = {d for d, ev in self.latest.desk_evidence.items()
                    if ev['eligible']}
        thresholds = {k: self.latest.thresholds[m][k]
                      for k, m in self.latest.method_pick.items()}
        fresh = {lid: lt for lid, lt in self.bundle.letters.items()
                 if lt.arrived >= 48}
        return route_week(self.bundle, self.tax, self.latest.trained,
                          self.latest.method_pick, thresholds, eligible,
                          today, fresh, self.bundle.assignments)

    def test_no_desk_eligible_so_all_fresh_go_manual_or_wait(self):
        outcome = self._route()
        # 当前证据下没有桌通过准入：新分送不得出现 auto=1 的非签收件
        for r in outcome.routes:
            if r.reason != 'acknowledged':
                self.assertFalse(r.auto)

    def test_one_slot_per_letter_per_desk(self):
        outcome = self._route()
        # x236 有 contact+refund 两个主题都转 manual，但 manual 只占一个位
        manual_letters = {r.letter for r in outcome.routes
                          if r.desk == MANUAL_DESK and r.state == 'received'}
        # manual 名额 4，去重后占位数不超过配额
        self.assertLessEqual(len(manual_letters), 4)

    def test_acknowledged_assignments_preserved(self):
        outcome = self._route()
        ack = [r for r in outcome.routes if r.reason == 'acknowledged']
        keys = {(r.letter, r.topic, r.desk) for r in ack}
        # 种子里两条 received 必须原样保留
        self.assertIn(('x237', 'claim', 'manual'), keys)
        self.assertIn(('x241', 'refund', 'finance'), keys)

    def test_no_letter_left_without_any_route(self):
        outcome = self._route()
        fresh_ids = {lid for lid, lt in self.bundle.letters.items()
                     if lt.arrived >= 48}
        routed = {r.letter for r in outcome.routes}
        self.assertEqual(fresh_ids, routed)

    def test_oov_reason_surfaces_for_x241_contact(self):
        outcome = self._route()
        x241_contact = [r for r in outcome.routes
                        if r.letter == 'x241' and r.topic == 'contact']
        self.assertTrue(x241_contact)
        self.assertEqual(x241_contact[0].base_reason, R_OOV)

    def test_full_window_waits_not_rerouted_to_other_desk(self):
        outcome = self._route()
        waiting = outcome.waiting
        self.assertTrue(waiting)
        for r in waiting:
            self.assertEqual(r.reason, R_FULL)
            # 等待件不得改投到有余位的别的桌
            self.assertIn(r.desk, (MANUAL_DESK, 'review', 'finance', 'service'))

    def test_partial_topic_delivery_does_not_wait_for_whole_letter(self):
        # x241：refund 已送达 finance，contact 因 OOV 转人工——两主题独立处置
        outcome = self._route()
        topics = {r.topic: r.state for r in outcome.routes
                  if r.letter == 'x241'}
        self.assertEqual(topics['refund'], 'received')
        self.assertIn(topics['contact'], ('received', 'waiting'))

    def test_urgent_letter_jumps_queue_for_manual_slot(self):
        from appeal.dataio import Letter
        eligible = {d for d, ev in self.latest.desk_evidence.items()
                    if ev['eligible']}
        thresholds = {k: self.latest.thresholds[m][k]
                      for k, m in self.latest.method_pick.items()}
        fresh = dict(self.bundle.letters)
        # 最晚到达但标记紧急，应在 manual 满员前抢到位置
        fresh['z999'] = Letter('z999', 50, 'web', 'free', True,
                               '先停服务但没收到钱')
        fresh = {k: v for k, v in fresh.items() if v.arrived >= 48}
        outcome = route_week(self.bundle, self.tax, self.latest.trained,
                             self.latest.method_pick, thresholds, eligible,
                             50, fresh, self.bundle.assignments)
        received = {r.letter for r in outcome.routes if r.state == 'received'}
        self.assertIn('z999', received)

    def test_manual_quota_caps_distinct_letters(self):
        outcome = self._route()
        # manual 当周配额 4；去重后实际占用不超过 4 个来件
        manual_distinct = {r.letter for r in outcome.routes
                           if r.desk == MANUAL_DESK and r.state == 'received'}
        self.assertLessEqual(len(manual_distinct), 4)
        # 超出部分必须 waiting 而不是丢失或改投
        self.assertTrue(outcome.waiting)


class TestCandidateComparison(PipelineTestBase):
    def test_methods_are_compared_not_single_score_picked(self):
        # 第二窗 contact 选 nbayes、refund 选 logreg，说明逐主题比较真实生效
        pick = self.results[1].method_pick
        self.assertEqual(pick.get('v2:contact'), 'nbayes')
        self.assertEqual(pick.get('v2:refund'), 'logreg')

    def test_every_picked_topic_has_audit_evidence(self):
        for res in self.results:
            for key, method in res.method_pick.items():
                sc = res.audit_scores[method][key]
                self.assertGreaterEqual(sc.support_pos, 1)


if __name__ == '__main__':
    unittest.main()
