import unittest

from ttl_cache import TTLCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class TTLCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache = TTLCache(self.clock)

    def test_value_is_available_before_expiry(self) -> None:
        self.cache.put("language", "python", 5)
        self.clock.now = 104.999

        self.assertEqual(self.cache.get("language"), "python")

    def test_value_is_expired_at_exact_deadline(self) -> None:
        self.cache.put("language", "python", 5)
        self.clock.now = 105.0

        with self.assertRaises(KeyError):
            self.cache.get("language")
        self.assertEqual(len(self.cache), 0)

    def test_missing_key_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.cache.get("missing")


if __name__ == "__main__":
    unittest.main()
