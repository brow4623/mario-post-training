import unittest

from evaluate_persona import aligned_scores, paired_delta
from judge import text_hash


class Alignment(unittest.TestCase):
    def test_rejects_missing_duplicate_stale_or_mixed_scores(self):
        outputs = [{'id': 'a', 'output': 'answer'}]
        score = {'id': 'a', 'text_sha256': text_hash('answer'), 'config_sha256': 'config'}
        self.assertEqual(aligned_scores(outputs, [score], 'config'), [score])
        for bad in ([], [score, score], [{**score, 'id': 'b'}],
                    [{**score, 'text_sha256': 'stale'}], [{**score, 'config_sha256': 'other'}]):
            with self.assertRaises(ValueError):
                aligned_scores(outputs, bad, 'config')

    def test_delta_direction_and_empty_subset(self):
        self.assertIsNone(paired_delta([], []))
        self.assertEqual(paired_delta([1, 2], [2, 3]), {'mean': 1, 'ci95': [1., 1.]})


if __name__ == '__main__':
    unittest.main()
