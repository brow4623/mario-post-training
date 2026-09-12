from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from eval import resume_rows, validate_rows
from verifier import extract_answer, is_correct
from compare import mask_math, paired_outputs


class EvaluationTests(unittest.TestCase):
    def test_comparison_alignment(self):
        rows = [{'id': 'a', 'question': 'q', 'answer': '2'}]
        prediction = {'id': 'a', 'question': 'q', 'gold': '2', 'output': '#### 2'}
        self.assertEqual(paired_outputs(rows, [prediction]), [prediction])
        for bad in ([], [prediction, prediction], [{**prediction, 'gold': '3'}]):
            with self.assertRaises(ValueError):
                paired_outputs(rows, bad)

    def test_math_mask(self):
        masked = mask_math('I add <<1+1=2>>2 coins!\n#### 2')
        self.assertNotIn('2', masked)
        self.assertIn('coins', masked)

    def test_full_numeric_value(self):
        self.assertEqual(extract_answer('#### 1/2'), Fraction(1, 2))
        self.assertFalse(is_correct('#### 1/2', '1'))
        for text in ('#### 2+3', '#### 2 apples', '#### 1,00', '#### 1/0',
                     '#### 2\nmore text', r'\boxed{2+3}', r'\boxed{2'):
            self.assertIsNone(extract_answer(text), text)

    def test_boxed_and_terminal_markers(self):
        self.assertTrue(is_correct(r'\boxed{\frac{1}{2}}', '0.5'))
        self.assertTrue(is_correct(r'\boxed{1,250}', '1250'))
        self.assertTrue(is_correct('#### -0.5', '-1/2'))
        self.assertTrue(is_correct('#### 7\nActually #### 8', '8'))
        self.assertFalse(is_correct('Answer 8', '8'))
        self.assertTrue(is_correct('Answer 8', '8', lenient=True))

    def test_invalid_datasets(self):
        for rows in ([], [{'id': 'a', 'answer': '2'}] * 2, [{'id': 'a', 'answer': 'bad'}]):
            with self.assertRaises(ValueError):
                validate_rows(rows)

    def test_resume_matches_exact_source(self):
        rows = [{'id': 'a', 'question': 'Question?', 'answer': '2'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.jsonl'
            self.assertEqual(resume_rows(path, rows), [])
            saved = {'id': 'a', 'question': 'Question?', 'gold': '2', 'output': '#### 2'}
            path.write_text(json.dumps(saved) + '\n', encoding='utf-8')
            self.assertEqual(resume_rows(path, rows), [saved])
            rows[0]['question'] = 'Changed'
            with self.assertRaises(ValueError):
                resume_rows(path, rows)
