import logging
import unittest

from logging_utils import configure_logging


class LoggingConfigTests(unittest.TestCase):
    def test_configure_logging_does_not_override_existing_handlers_by_default(self):
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        old_level = root.level
        sentinel = logging.StreamHandler()
        try:
            root.handlers = [sentinel]
            root.setLevel(logging.WARNING)
            configure_logging(level="DEBUG", force=False)
            self.assertIs(root.handlers[0], sentinel)
            self.assertEqual(root.level, logging.WARNING)
        finally:
            root.handlers = old_handlers
            root.setLevel(old_level)


if __name__ == "__main__":
    unittest.main()
