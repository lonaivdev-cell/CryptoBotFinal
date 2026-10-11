import logging
import unittest

from tests.fakes import REAL_LOOKING_KEY, REAL_LOOKING_SECRET

from botservice.safety import (
    REDACTED,
    RedactingFilter,
    Redactor,
    is_loopback_host,
    is_placeholder,
    keys_configured,
    live_gate,
    sentence,
)
from botservice.settings import ServiceSettings, SettingsError


class PlaceholderKeyTest(unittest.TestCase):
    def test_shipped_placeholders_are_detected(self):
        # Every value the README and .env.example have ever shipped, plus the usual suspects.
        for value in (
            None, "", "   ", "your_key_here", "your_secret_here", "YOUR_KEY_HERE", "PLACEHOLDER",
            "PLACEHOLDER_BINANCE_API_KEY", "placeholder-change-me", "changeme", "<your binance key>",
            "${BINANCE_API_KEY}", "xxxxxxxxxxxxxxxxxxxxxxxxxxxx", "****************", '"your_key_here"',
            "short", "a key with spaces in it but long", "todo", "replace-with-real-key-please",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_placeholder(value))

    def test_real_looking_values_are_keys(self):
        for value in (REAL_LOOKING_KEY, REAL_LOOKING_SECRET, "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A"):
            with self.subTest(value=value):
                self.assertFalse(is_placeholder(value))

    def test_both_key_and_secret_must_be_real(self):
        self.assertTrue(keys_configured(REAL_LOOKING_KEY, REAL_LOOKING_SECRET))
        self.assertFalse(keys_configured(REAL_LOOKING_KEY, "your_secret_here"))
        self.assertFalse(keys_configured("PLACEHOLDER", REAL_LOOKING_SECRET))
        self.assertFalse(keys_configured("", ""))


class LiveGateTest(unittest.TestCase):
    LIVE_ENV = {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": "1"}

    def test_default_environment_simulates(self):
        gate = live_gate({}, keys_ok=True)
        self.assertEqual(gate.mode, "simulation")
        self.assertTrue(gate.simulate_only)
        self.assertIsNone(gate.reason)

    def test_live_needs_both_switches_and_real_keys(self):
        self.assertEqual(live_gate(self.LIVE_ENV, keys_ok=True).mode, "live")
        self.assertFalse(live_gate(self.LIVE_ENV, keys_ok=True).simulate_only)

    def test_each_missing_piece_keeps_simulation_and_says_why(self):
        cases = [
            ({"LLAMA_SENTIMENT_SIMULATE_ONLY": "False"}, True, "CRYPTOBOT_ALLOW_LIVE is not 1"),
            ({"CRYPTOBOT_ALLOW_LIVE": "1"}, True, "LLAMA_SENTIMENT_SIMULATE_ONLY is not False"),
            (self.LIVE_ENV, False, "placeholders"),
        ]
        for env, keys_ok, why in cases:
            with self.subTest(env=env, keys_ok=keys_ok):
                gate = live_gate(env, keys_ok=keys_ok)
                self.assertEqual(gate.mode, "simulation")
                self.assertIn(why, gate.reason)

    def test_allow_live_must_be_exactly_one(self):
        for allow in ("true", "yes", "on", "2", "01", " ", "1 # sure"):
            with self.subTest(allow=allow):
                env = {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": allow}
                self.assertEqual(live_gate(env, keys_ok=True).mode, "simulation")
        env = {"LLAMA_SENTIMENT_SIMULATE_ONLY": "False", "CRYPTOBOT_ALLOW_LIVE": " 1 "}
        self.assertEqual(live_gate(env, keys_ok=True).mode, "live")

    def test_simulate_only_needs_an_explicit_false_word(self):
        for value in ("False", "false", "FALSE", "0", "no", "off"):
            with self.subTest(value=value):
                env = {"LLAMA_SENTIMENT_SIMULATE_ONLY": value, "CRYPTOBOT_ALLOW_LIVE": "1"}
                self.assertEqual(live_gate(env, keys_ok=True).mode, "live")
        # Typos and anything else keep the safety switch on.
        for value in ("Flase", "nope", "True", "1", "", "fals"):
            with self.subTest(value=value):
                env = {"LLAMA_SENTIMENT_SIMULATE_ONLY": value, "CRYPTOBOT_ALLOW_LIVE": "1"}
                self.assertEqual(live_gate(env, keys_ok=True).mode, "simulation")


class LoopbackTest(unittest.TestCase):
    def test_loopback_hosts(self):
        for host in ("127.0.0.1", "127.0.0.53", "::1", "[::1]", "localhost", "LOCALHOST"):
            self.assertTrue(is_loopback_host(host), host)
        for host in ("0.0.0.0", "::", "192.168.1.10", "100.64.0.2", "loserver", "example.com", "", "localhost.evil.com"):
            self.assertFalse(is_loopback_host(host), host)

    def test_settings_refuse_a_non_loopback_bind(self):
        for bind in ("0.0.0.0", "::", "100.64.0.7", "loserver"):
            with self.subTest(bind=bind):
                with self.assertRaises(SettingsError) as ctx:
                    ServiceSettings.from_env({"CRYPTOBOT_CONTROL_BIND": bind})
                self.assertIn("loopback", str(ctx.exception))
        self.assertEqual(ServiceSettings.from_env({"CRYPTOBOT_CONTROL_BIND": "::1"}).bind, "::1")

    def test_settings_defaults_and_validation(self):
        s = ServiceSettings.from_env({})
        self.assertEqual((s.bind, s.port, s.token, s.autostart), ("127.0.0.1", 8097, None, False))
        self.assertEqual(ServiceSettings.from_env({"CRYPTOBOT_CONTROL_PORT": "9001"}).port, 9001)
        for env in ({"CRYPTOBOT_CONTROL_PORT": "eighty"}, {"CRYPTOBOT_CONTROL_PORT": "70000"},
                    {"CRYPTOBOT_PAPER_USDT": "-5"}, {"CRYPTOBOT_AUTOSTART": "maybe"},
                    {"CRYPTOBOT_CONTROL_TOKEN": "short"}):
            with self.subTest(env=env):
                with self.assertRaises(SettingsError):
                    ServiceSettings.from_env(env)


class RedactionTest(unittest.TestCase):
    def test_redactor_replaces_known_secrets_only(self):
        r = Redactor([REAL_LOOKING_KEY, REAL_LOOKING_SECRET, "", None, "x"])
        text = f"binance said: apiKey={REAL_LOOKING_KEY} sig for {REAL_LOOKING_SECRET}; x marks the spot"
        out = r(text)
        self.assertNotIn(REAL_LOOKING_KEY, out)
        self.assertNotIn(REAL_LOOKING_SECRET, out)
        self.assertEqual(out.count(REDACTED), 2)
        self.assertIn("x marks the spot", out)  # a short "secret" never blanks ordinary text

    def test_logging_filter_redacts_args_and_tracebacks(self):
        r = Redactor([REAL_LOOKING_KEY])
        logger = logging.getLogger("test.redaction")
        logger.propagate = False
        records = []

        class Keep(logging.Handler):
            def emit(self, record):
                records.append(self.format(record))

        handler = Keep()
        handler.addFilter(RedactingFilter(r))
        logger.addHandler(handler)
        try:
            logger.error("key is %s", REAL_LOOKING_KEY)
            try:
                raise RuntimeError(f"rejected {REAL_LOOKING_KEY}")
            except RuntimeError:
                logger.exception("boom")
        finally:
            logger.removeHandler(handler)
        joined = "\n".join(records)
        self.assertNotIn(REAL_LOOKING_KEY, joined)
        self.assertIn("RuntimeError", joined)
        self.assertGreaterEqual(joined.count(REDACTED), 2)

    def test_sentence_is_one_bounded_line(self):
        out = sentence("a\n  b\tc " + "z" * 500, limit=50)
        self.assertEqual(len(out), 50)
        self.assertTrue(out.startswith("a b c "))


if __name__ == "__main__":
    unittest.main()
