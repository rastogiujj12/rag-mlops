from pathlib import Path
from types import SimpleNamespace

import pytest

from src.indexing import model_loader


class DummySentenceTransformer:
    def __init__(self, model_name, device=None, cache_folder=None, local_files_only=None):
        self.model_name = model_name
        self.device = device
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only


def test_model_loader_uses_cache_dir_and_offline_flag(monkeypatch, tmp_path):
    calls = []

    def fake_sentence_transformer(*args, **kwargs):
        calls.append((args, kwargs))
        return DummySentenceTransformer(*args, **kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )

    model = model_loader.load_sentence_transformer(
        "intfloat/e5-base-v2",
        cache_dir=tmp_path / "st-cache",
        device="cpu",
        offline=True,
    )

    assert model.model_name == "intfloat/e5-base-v2"
    assert model.cache_folder == str(tmp_path / "st-cache")
    assert model.local_files_only is True
    assert Path(model.cache_folder).exists()
    assert calls


def test_model_loader_gives_clear_offline_error(monkeypatch, tmp_path):
    def fake_sentence_transformer(*args, **kwargs):
        raise OSError("not cached")

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )

    with pytest.raises(RuntimeError, match="offline mode"):
        model_loader.load_sentence_transformer(
            "intfloat/e5-base-v2",
            cache_dir=tmp_path / "missing-cache",
            offline=True,
        )
