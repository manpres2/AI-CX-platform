"""Contracts for a single model owner, independent of downloaded weights."""
import gc
import sys
import types
import unittest
import weakref
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import torch
import local_tts_server as service


class LocalTtsTests(unittest.TestCase):
    def tearDown(self):
        service._unload()

    def test_releases_old_model_before_new_constructor(self):
        class Model:
            pass
        old = Model()
        old_ref = weakref.ref(old)
        service._model = old
        service._loaded = ("kokoro", "old", "a")
        del old
        def create(*args, **kwargs):
            self.assertIsNone(old_ref())
            self.assertEqual(kwargs["device"], "cuda")
            return Model()
        module = types.SimpleNamespace(KPipeline=create)
        with patch.dict(sys.modules, {"kokoro": module}), patch.object(torch.cuda, "is_available", return_value=True):
            service._load(service.Speech(engine="kokoro", text="Hello", repo_id="new"))
            self.assertEqual(service.health()["loaded_models"], 1)
            self.assertEqual(service.health()["engine"], "kokoro")

    def test_gpu_required_and_previous_model_released_on_failure(self):
        service._model = object()
        service._loaded = ("kokoro", "old", "a")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                service._load(service.Speech(text="hello"))
        self.assertIsNone(service._model)
        self.assertIsNone(service._loaded)

    def test_select_unloads_without_loading_next_model(self):
        service._model = object()
        service._loaded = ("kokoro", "old", "a")
        client = TestClient(service.app)
        with patch.object(service, "_load") as load:
            self.assertEqual(client.post("/select", json={"engine": "kokoro"}).status_code, 200)
            self.assertIsNotNone(service._model)
            self.assertEqual(client.post("/select", json={"engine": "chatterbox"}).status_code, 200)
            self.assertIsNone(service._model)
            load.assert_not_called()

    def test_busy_rejects_synthesis_and_switch_without_touching_model(self):
        client = TestClient(service.app)
        service._model = object()
        original = service._model
        service._lock.acquire()
        try:
            self.assertEqual(client.post("/tts", json={"text": "hi"}).status_code, 409)
            self.assertEqual(client.post("/select", json={"engine": "cloud"}).status_code, 409)
            self.assertIs(service._model, original)
        finally:
            service._lock.release()

    def test_http_audio_expressiveness_and_validation(self):
        calls = []
        class Model:
            sr = 24000
            def generate(self, text, **kwargs):
                calls.append(kwargs)
                return torch.tensor([[0, 0.5, -0.5]])
        client = TestClient(service.app)
        with patch.object(service, "_load", return_value=Model()):
            response = client.post("/tts", json={"text": "hello", "exaggeration": 0, "cfg_weight": 0})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-sample-rate"], "24000")
            self.assertEqual(np.frombuffer(response.content, dtype="<i2").tolist(), [0, 16383, -16383])
            self.assertEqual(calls, [{"exaggeration": 0, "cfg_weight": 0}])
        for body in ({"text": ""}, {"text": "hi", "engine": "qwen3"},
                     {"text": "hi", "exaggeration": 2}, {"text": "hi", "cfg_weight": -1}):
            self.assertEqual(client.post("/tts", json=body).status_code, 422)

    def test_generation_failure_releases_owner(self):
        class Model:
            def generate(self, *args, **kwargs):
                raise RuntimeError("GPU failure")
        service._model = Model()
        service._loaded = ("chatterbox",)
        response = TestClient(service.app).post("/tts", json={"text": "Hello"})
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(service._model)
        self.assertEqual(service.health()["loaded_models"], 0)


if __name__ == "__main__":
    unittest.main()
