"""Deterministic sampling contracts; no accuracy expectations."""
import unittest
from collections import Counter
from sampling import build_schedule, build_history


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.idx = {}
        for seq in ('a', 'b'):
            for frame in range(0, 50, 5):
                self.idx[f'{seq}_{frame}'] = dict(seq=seq, frame=frame,
                    exist=int(frame < 40), attr={'TC': int(frame < (10 if seq=='a' else 25))})

    def test_epoch_size_negative_protection_and_sequence_quota(self):
        for epoch in range(10):
            a = build_schedule(self.idx, 'global', 0, epoch)
            b = build_schedule(self.idx, 'quota', 0, epoch)
            for order in (a, b):
                self.assertEqual(len(order), len(self.idx))
                self.assertLessEqual(max(Counter(order).values()), 2)
                self.assertEqual(Counter(s for s in order if not self.idx[s]['exist']),
                                 Counter(s for s in self.idx if not self.idx[s]['exist']))
            self.assertEqual(sum(self.idx[s]['attr']['TC'] for s in a),
                             sum(self.idx[s]['attr']['TC'] for s in b))
            self.assertEqual(Counter(self.idx[s]['seq'] for s in b),
                             Counter(r['seq'] for r in self.idx.values()))

    def test_deterministic_and_rotates(self):
        self.assertEqual(build_schedule(self.idx, 'quota', 0, 0),
                         build_schedule(self.idx, 'quota', 0, 0))
        self.assertNotEqual(build_schedule(self.idx, 'quota', 0, 0),
                            build_schedule(self.idx, 'quota', 0, 1))

    def test_uniform_has_all_frames_once(self):
        self.assertEqual(Counter(build_schedule(self.idx, 'uniform', 0, 0)),Counter(self.idx.keys()))

    def test_history_is_causal_same_sequence_and_label_independent(self):
        history=build_history(self.idx)
        for stem, triplet in history.items():
            self.assertEqual(triplet[-1], stem)
            for previous in triplet:
                self.assertEqual(self.idx[stem]['seq'], self.idx[previous]['seq'])
                self.assertLessEqual(self.idx[previous]['frame'],self.idx[stem]['frame'])
        self.assertEqual(history['a_0'], ['a_0','a_0','a_0'])
        self.assertEqual(history['a_5'], ['a_0','a_0','a_5'])
        self.assertEqual(history['a_10'], ['a_0','a_5','a_10'])
        changed={s:dict(r,exist=1-r['exist'],attr={}) for s,r in self.idx.items()}
        self.assertEqual(history,build_history(changed))


if __name__=='__main__':
    unittest.main()
