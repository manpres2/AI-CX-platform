"""Provider contract tests without downloading speech models."""
import ast
import asyncio
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKENDS = ("app/main.py", "bot-template/app/main_bot.py",
            "techsupport-voice-bot/app/main_tech.py")


def functions_from(path, names):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in names]
    return compile(ast.Module(body=selected, type_ignores=[]), path, "exec")


class ProviderTests(unittest.TestCase):
    def test_dispatch_and_preset_payloads(self):
        for path in BACKENDS:
            with self.subTest(backend=path):
                calls = []
                def post(url, **kwargs):
                    calls.append((url, kwargs))
                    return types.SimpleNamespace(content=b"qwen", raise_for_status=lambda: None)
                cfg = {"tts_mode": "local", "tts_local_engine": "qwen3",
                       "tts_cloud": {"qwen3": {"voice_mode": "preset", "speaker": "Aiden"}}}
                ns = {"asyncio": asyncio, "httpx": types.SimpleNamespace(post=post),
                      "_normalize_for_speech": lambda text: text,
                      "load_provider_config": lambda: cfg, "synthesize": lambda text: b"kokoro",
                      "_TTS_CLOUD_ENGINES": {"veena": lambda text, settings: b"veena"}}
                exec(functions_from(path, {"synthesize_active", "synthesize_qwen3"}), ns)
                self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"qwen")
                self.assertEqual(calls[-1][0], "http://127.0.0.1:8020/tts")
                self.assertEqual(calls[-1][1]["json"]["speaker"], "Aiden")
                cfg["tts_local_engine"] = "kokoro"
                self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"kokoro")
                cfg.update(tts_mode="cloud", tts_cloud_engine="veena")
                self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"veena")
                cfg["tts_cloud_engine"] = "unknown"
                with self.assertRaises(ValueError):
                    asyncio.run(ns["synthesize_active"]("hello"))
                with self.assertRaises(ValueError):
                    ns["synthesize_qwen3"]("hello", {"voice_mode": "clone"})

    def test_legacy_qwen_config_migrates_to_local(self):
        import json
        import logging
        for path in BACKENDS:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
            default = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "DEFAULT_PROVIDER_CONFIG"
                                   for t in n.targets))
            with tempfile.TemporaryDirectory() as tmp:
                config = pathlib.Path(tmp) / "providers.json"
                config.write_text(json.dumps({"tts_mode": "cloud", "tts_cloud_engine": "qwen3",
                                             "tts_cloud": {"qwen3": {"speaker": "Aiden"}}}))
                ns = {"json": json, "log": logging.getLogger(__name__), "PROVIDER_FILE": config,
                      "DEFAULT_PROVIDER_CONFIG": ast.literal_eval(default)}
                exec(functions_from(path, {"load_provider_config"}), ns)
                result = ns["load_provider_config"]()
                self.assertEqual((result["tts_mode"], result["tts_local_engine"]), ("local", "qwen3"))
                self.assertEqual(result["tts_cloud"]["qwen3"]["speaker"], "Aiden")


class ServiceTests(unittest.TestCase):
    def test_http_contract(self):
        import numpy as np
        import qwen3_tts_server as service
        from fastapi.testclient import TestClient
        class Model:
            def generate_custom_voice(self, **kwargs):
                return [np.array([0, .5, -.5])], 24000
        with patch.object(service, "_get_model", return_value=Model()), patch.object(service, "API_KEY", ""):
            client = TestClient(service.app)
            for speaker in service.SPEAKERS:
                response = client.post("/tts", json={"text": "Hello", "speaker": speaker})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.content), 6)
                self.assertEqual(response.headers["x-sample-rate"], "24000")
            self.assertEqual(client.post("/tts", json={"text": "Hi", "speaker": "unknown"}).status_code, 400)
            self.assertEqual(client.post("/tts", json={"text": "Hi", "voice_mode": "clone"}).status_code, 400)
            with patch.object(service, "API_KEY", "test-secret"):
                self.assertEqual(client.post("/tts", json={"text": "Hi"}).status_code, 401)
                self.assertEqual(client.post("/tts", json={"text": "Hi"},
                                 headers={"Authorization": "Bearer test-secret"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
