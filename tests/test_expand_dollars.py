import unittest

from openvoice.text.english import _dollars_re, _expand_dollars


class ExpandDollarsTests(unittest.TestCase):
    def expand(self, text):
        match = _dollars_re.search(text)
        self.assertIsNotNone(match)
        return _expand_dollars(match)

    def test_one_decimal_place_is_tens_of_cents(self):
        self.assertEqual(self.expand("$1.5"), "1 dollar, 50 cents")
        self.assertEqual(self.expand("$0.5"), "50 cents")
        self.assertEqual(self.expand("$10.2"), "10 dollars, 20 cents")

    def test_two_decimal_places_stay_literal_cents(self):
        self.assertEqual(self.expand("$1.50"), "1 dollar, 50 cents")
        self.assertEqual(self.expand("$1.05"), "1 dollar, 5 cents")
        self.assertEqual(self.expand("$2.05"), "2 dollars, 5 cents")

    def test_whole_dollars(self):
        self.assertEqual(self.expand("$1"), "1 dollar")
        self.assertEqual(self.expand("$2"), "2 dollars")


if __name__ == "__main__":
    unittest.main()
