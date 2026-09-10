"""Provider contract tests without downloading speech models."""
import ast
import asyncio
import pathlib
import re
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKENDS = ("app/main.py", "bot-template/app/main_bot.py",
            "techsupport-voice-bot/app/main_tech.py")


def functions_from(path, names):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if
                (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names)
                or ("_normalize_for_speech" in names and isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == "_CURRENCY_RE" for t in n.targets))]
    return compile(ast.Module(body=selected, type_ignores=[]), path, "exec")


class ProviderTests(unittest.TestCase):
    def test_exclusive_tts_selects_shared_owner_before_generation(self):
        import threading
        from functools import wraps
        for path in BACKENDS:
            calls = []
            ns = {"wraps": wraps, "_TTS_MODEL_LOCK": threading.RLock(),
                  "_shared_tts": types.SimpleNamespace(select=lambda e: calls.append(e))}
            exec(functions_from(path, {"_release_inactive_tts", "_exclusive_tts"}), ns)
            for engine in ("kokoro", "chatterbox", "cloud"):
                calls.clear()
                ns["_exclusive_tts"](engine, lambda: calls.append("speak"))()
                self.assertEqual(calls, [engine, "speak"])

    def test_local_chatterbox_dispatch_and_zero_expressiveness(self):
        for path in BACKENDS:
            calls = []
            def speak(engine, text, **kwargs):
                calls.append((engine, text, kwargs))
                return b"audio"
            cfg = {"tts_mode": "local", "tts_local_engine": "chatterbox",
                   "tts_cloud": {"chatterbox": {"exaggeration": 0, "cfg_weight": 0}}}
            ns = {"asyncio": asyncio, "load_provider_config": lambda: cfg,
                  "_shared_tts": types.SimpleNamespace(synthesize=speak),
                  "synthesize": lambda text: b"kokoro",
                  "_TTS_CLOUD_ENGINES": {"veena": lambda text, settings: b"veena"}}
            exec(functions_from(path, {"synthesize_active", "synthesize_chatterbox"}), ns)
            self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"audio")
            self.assertEqual(calls, [("chatterbox", "hello", {"exaggeration": 0, "cfg_weight": 0})])
            cfg["tts_local_engine"] = "kokoro"
            self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"kokoro")
            cfg.update(tts_mode="cloud", tts_cloud_engine="veena")
            self.assertEqual(asyncio.run(ns["synthesize_active"]("hello")), b"veena")
            cfg["tts_cloud_engine"] = "unknown"
            with self.assertRaises(ValueError):
                asyncio.run(ns["synthesize_active"]("hello"))

    def test_removed_qwen_config_migrates_to_kokoro(self):
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
                exec(functions_from(path, {"load_provider_config", "save_provider_config"}), ns)
                result = ns["load_provider_config"]()
                self.assertEqual((result["tts_mode"], result["tts_local_engine"]), ("local", "kokoro"))
                self.assertEqual(result["tts_cloud"]["qwen3"]["speaker"], "Aiden")
                result["tts_cloud_engine"] = "elevenlabs"
                ns["save_provider_config"](result)
                restored = ns["load_provider_config"]()
                self.assertEqual(restored["tts_local_engine"], "kokoro")
                result["tts_local_engine"] = "chatterbox"
                result["tts_cloud"]["chatterbox"] = {"exaggeration": 0, "cfg_weight": 0.3}
                ns["save_provider_config"](result)
                expressive = ns["load_provider_config"]()
                self.assertEqual(expressive["tts_local_engine"], "chatterbox")
                self.assertEqual(expressive["tts_cloud"]["chatterbox"]["exaggeration"], 0)
                self.assertEqual(restored["tts_cloud"]["qwen3"]["speaker"], "Aiden")



class ServiceTests(unittest.TestCase):
    def test_gpu_required(self):
        import qwen3_tts_server as service
        with patch.object(service.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an NVIDIA GPU"):
                service._get_model(service.PRESET_MODEL_ID)

    def test_unload_releases_model_and_prompt(self):
        import qwen3_tts_server as service
        with patch.object(service, "_model", object()), patch.object(service, "_loaded_model_id", "test"), patch.object(service, "API_KEY", ""):
            service._prompt_cache["test"] = object()
            service.unload(authorization=None)
            self.assertIsNone(service._model)
            self.assertIsNone(service._loaded_model_id)
            self.assertFalse(service._prompt_cache)

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
