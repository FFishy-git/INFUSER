from __future__ import annotations

from unittest.mock import patch

from verl_inf_evolve.utils import env_utils


def test_parse_dotenv_line_supports_quotes_and_export():
    assert env_utils._parse_dotenv_line('export FOO="bar baz" # comment') == (
        "FOO",
        "bar baz",
    )
    assert env_utils._parse_dotenv_line("X=1") == ("X", "1")
    assert env_utils._parse_dotenv_line("# comment") is None


def test_loads_default_dotenv_from_cwd(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("FOO=bar\nBAR='baz qux'\n", encoding="utf-8")

    with patch.dict(env_utils.os.environ, {}, clear=False):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VERL_INF_EVOLVE_DOTENV_PATH", raising=False)
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAR", raising=False)
        monkeypatch.setattr(env_utils, "_DOTENV_LOADED", False)

        loaded = env_utils.load_startup_env()

        assert loaded == [str(dotenv)]
        assert env_utils.os.environ["FOO"] == "bar"
        assert env_utils.os.environ["BAR"] == "baz qux"


def test_does_not_override_existing_env(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("FOO=from_file\n", encoding="utf-8")

    with patch.dict(env_utils.os.environ, {}, clear=False):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VERL_INF_EVOLVE_DOTENV_PATH", raising=False)
        monkeypatch.setenv("FOO", "already_set")
        monkeypatch.setattr(env_utils, "_DOTENV_LOADED", False)

        env_utils.load_startup_env()

        assert env_utils.os.environ["FOO"] == "already_set"


def test_loads_custom_dotenv_path_from_env(monkeypatch, tmp_path):
    dotenv = tmp_path / "custom.env"
    dotenv.write_text("FOO=bar\n", encoding="utf-8")

    with patch.dict(env_utils.os.environ, {}, clear=False):
        monkeypatch.setenv("VERL_INF_EVOLVE_DOTENV_PATH", str(dotenv))
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.setattr(env_utils, "_DOTENV_LOADED", False)

        loaded = env_utils.load_startup_env()

        assert loaded == [str(dotenv)]
        assert env_utils.os.environ["FOO"] == "bar"


def test_sanitize_cuda_visible_device_env_unsets_rocm_vars_for_nvidia(monkeypatch):
    with patch.dict(env_utils.os.environ, {}, clear=False):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")

        removed = env_utils.sanitize_cuda_visible_device_env()

        assert removed == {
            "ROCR_VISIBLE_DEVICES": "0,1",
            "HIP_VISIBLE_DEVICES": "0,1",
        }
        assert "ROCR_VISIBLE_DEVICES" not in env_utils.os.environ
        assert "HIP_VISIBLE_DEVICES" not in env_utils.os.environ


def test_sanitize_cuda_visible_device_env_keeps_rocm_vars_without_cuda(monkeypatch):
    with patch.dict(env_utils.os.environ, {}, clear=False):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")

        removed = env_utils.sanitize_cuda_visible_device_env()

        assert removed == {}
        assert env_utils.os.environ["ROCR_VISIBLE_DEVICES"] == "0"
        assert env_utils.os.environ["HIP_VISIBLE_DEVICES"] == "0"
