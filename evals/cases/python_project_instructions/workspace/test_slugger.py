import unittest

from slugger import make_slug


class SluggerTests(unittest.TestCase):
    def test_simple_title(self) -> None:
        self.assertEqual(make_slug("Hello World"), "hello-world")


if __name__ == "__main__":
    unittest.main()
