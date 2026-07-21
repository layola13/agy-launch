#!/usr/bin/env python3
# Copyright (c) 2026 agy-launch contributors
# 仅供学习与研究使用；其他用途后果自负。
# For learning and research only; any other use is at your own risk.
"""agy-launch: local Gemini Code Assist shim that forwards to an OpenAI-compatible API.

Starts a tiny local proxy, points agy at it via CLOUD_CODE_URL, translates:
  Google streamGenerateContent  <->  OpenAI /v1/chat/completions

Configuration is loaded from environment variables and/or .env files.
No secrets or private endpoints are hard-coded (safe for public repos).
See .env.example and README.md.
"""

from __future__ import annotations

from copy import deepcopy
from email.utils import parsedate_to_datetime
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))

# Required config keys (must come from env / .env — never hard-coded).
_REQUIRED_KEYS = (
    "AGY_LAUNCH_BASE_URL",
    "AGY_LAUNCH_MODEL",
)
_MANAGED_ENV_PREFIXES = ("AGY_LAUNCH_",)
_MANAGED_ENV_KEYS = {"AGY_BIN"}

# Filled by load_config() before the proxy serves traffic.
TARGET_BASE_URL: str = ""
TARGET_MODEL: str = ""
TARGET_API_KEY: str = ""
API_KEYS: list[str] = []
FROZEN_KEYS: dict[str, float] = {}
FROZEN_LOCK = threading.Lock()
REASONING_EFFORT: str = ""
MAX_COMPLETION_TOKENS: int = 0
MAX_TOKENS: int = 0
AGY_CLI_MODEL: str = "gemini-3.5-flash-low"
AGY_BIN: str = "agy"
DEBUG_DIR: str = ""
MODEL_DISPLAY_NAME: str = ""
MODEL_PROVIDER: str = ""
UPSTREAM_USER_AGENT: str = "curl/8.5.0"
UPSTREAM_MIN_INTERVAL_SECONDS: float = 0.25
UPSTREAM_RETRIES: int = 2
UPSTREAM_429_FREEZE_SECONDS: float = 60.0
UPSTREAM_5XX_FREEZE_SECONDS: float = 30.0
UPSTREAM_MAX_RETRY_AFTER_SECONDS: float = 300.0
UPSTREAM_BACKOFF_INITIAL_SECONDS: float = 1.0
UPSTREAM_BACKOFF_MAX_SECONDS: float = 30.0
UPSTREAM_NEXT_REQUEST_AT: float = 0.0
UPSTREAM_COOLDOWN_UNTIL: float = 0.0
UPSTREAM_THROTTLE_LOCK = threading.Lock()
_LOADED_ENV_FILES: list[str] = []


def _bounded_float(value: str | None, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _bounded_int(value: str | None, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value) if value is not None and value.strip() else default
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt is None:
            return None
        return max(0.0, dt.timestamp() - (time.time() if now is None else now))
    except Exception:
        return None


def _clamp_retry_delay(delay: float | None, fallback: float) -> float:
    selected = fallback if delay is None else delay
    return min(max(0.0, selected), UPSTREAM_MAX_RETRY_AFTER_SECONDS)


def _retry_backoff_seconds(attempt: int) -> float:
    base = min(UPSTREAM_BACKOFF_MAX_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return base + random.uniform(0.0, min(1.0, base * 0.25))


def _apply_upstream_cooldown(seconds: float, reason: str) -> None:
    if seconds <= 0:
        return
    capped = min(seconds, UPSTREAM_MAX_RETRY_AFTER_SECONDS)
    with UPSTREAM_THROTTLE_LOCK:
        global UPSTREAM_COOLDOWN_UNTIL
        until = time.time() + capped
        if until > UPSTREAM_COOLDOWN_UNTIL:
            UPSTREAM_COOLDOWN_UNTIL = until
    if os.environ.get("AGY_LAUNCH_VERBOSE"):
        print(f"[agy-launch] upstream cooldown {capped:.2f}s ({reason})", file=sys.stderr)


def _wait_for_upstream_slot() -> None:
    global UPSTREAM_NEXT_REQUEST_AT
    while True:
        with UPSTREAM_THROTTLE_LOCK:
            now = time.time()
            wait_until = max(UPSTREAM_COOLDOWN_UNTIL, UPSTREAM_NEXT_REQUEST_AT)
            wait_for = wait_until - now
            if wait_for <= 0:
                UPSTREAM_NEXT_REQUEST_AT = now + UPSTREAM_MIN_INTERVAL_SECONDS
                return
        time.sleep(min(wait_for, 5.0))


def get_active_key() -> str:
    with FROZEN_LOCK:
        now = time.time()
        for key in API_KEYS:
            if FROZEN_KEYS.get(key, 0) <= now:
                return key

        candidates = []
        for key in API_KEYS:
            expire = FROZEN_KEYS.get(key, 0)
            if expire - now < 43200:  # less than 12 hours
                candidates.append((expire, key))

        if candidates:
            candidates.sort()
            expire, key = candidates[0]
            FROZEN_KEYS[key] = 0
            return key

        return API_KEYS[0]


def mark_key_failed(key: str, status_code: int, retry_after: str | None = None) -> float:
    with FROZEN_LOCK:
        now = time.time()
        if status_code == 429:
            freeze_for = _clamp_retry_delay(
                _parse_retry_after(retry_after, now=now),
                UPSTREAM_429_FREEZE_SECONDS,
            )
            FROZEN_KEYS[key] = now + freeze_for
            print(f"[agy-launch] key {key[:10]}... frozen temporarily (429 rate limit) for {freeze_for:.0f}s", file=sys.stderr)
            return freeze_for
        elif status_code in (401, 402):
            FROZEN_KEYS[key] = now + 86400
            print(f"[agy-launch] key {key[:10]}... frozen permanently (401/402 auth error)", file=sys.stderr)
            return 86400.0
        elif 500 <= status_code <= 599:
            freeze_for = UPSTREAM_5XX_FREEZE_SECONDS
            FROZEN_KEYS[key] = now + freeze_for
            print(f"[agy-launch] key {key[:10]}... frozen temporarily ({status_code} server error) for {freeze_for:.0f}s", file=sys.stderr)
            return freeze_for
    return 0.0


def _parse_dotenv(path: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file (no external deps)."""
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                # Strip matching quotes
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                out[key] = val
    except OSError:
        pass
    return out


def _candidate_env_paths() -> list[str]:
    """Ordered list of .env paths to try (later files do not override earlier keys
    once applied with override=False — we apply highest priority first)."""
    paths: list[str] = []
    explicit = os.environ.get("AGY_LAUNCH_ENV")
    if explicit:
        paths.append(os.path.expanduser(explicit))

    # 1) Launcher/package directory config (the shipped agy-launch .env)
    paths.append(os.path.join(_HERE, ".env"))

    # 2) Current working directory (project-local)
    cwd = os.getcwd()
    paths.append(os.path.join(cwd, ".env"))
    paths.append(os.path.join(cwd, ".agy-launch.env"))

    # 3) Walk parents (monorepo / nested cwd)
    parent = os.path.dirname(cwd)
    for _ in range(6):
        if not parent or parent == os.path.dirname(parent):
            break
        paths.append(os.path.join(parent, ".env"))
        paths.append(os.path.join(parent, ".agy-launch.env"))
        parent = os.path.dirname(parent)

    # 4) User config (install script default)
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    paths.append(os.path.join(xdg, "agy-launch", ".env"))
    paths.append(os.path.expanduser("~/.agy-launch.env"))

    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(ap)
    return uniq


def _is_managed_env_key(key: str) -> bool:
    return key in _MANAGED_ENV_KEYS or key.startswith(_MANAGED_ENV_PREFIXES)


def load_dotenv_files(*, override_existing: bool = False) -> list[str]:
    """Load KEY=VALUE from discovered .env files into os.environ.

    By default, discovered .env files override inherited shell values for
    agy-launch-managed keys (AGY_LAUNCH_* and AGY_BIN), preventing stale
    exported credentials from winning over the launcher config. Unrelated
    environment variables keep the traditional shell-over-dotenv behavior.
    When multiple files define the same key, the first file in priority order wins
    (AGY_LAUNCH_ENV override before launcher .env before cwd/user config).
    """
    loaded: list[str] = []
    claimed: set[str] = set()
    if not override_existing:
        claimed |= {k for k in os.environ.keys() if not _is_managed_env_key(k)}

    for path in _candidate_env_paths():
        if not os.path.isfile(path):
            continue
        data = _parse_dotenv(path)
        if not data:
            continue
        any_applied = False
        for k, v in data.items():
            if k in claimed:
                continue
            os.environ[k] = v
            claimed.add(k)
            any_applied = True
        if any_applied or data:
            loaded.append(path)
    return loaded


def load_config() -> None:
    """Populate module-level TARGET_* from env + .env. Exit if required missing."""
    global TARGET_BASE_URL, TARGET_MODEL, TARGET_API_KEY, API_KEYS
    global AGY_CLI_MODEL, AGY_BIN, DEBUG_DIR, MODEL_DISPLAY_NAME, MODEL_PROVIDER
    global UPSTREAM_USER_AGENT, _LOADED_ENV_FILES
    global REASONING_EFFORT, MAX_COMPLETION_TOKENS, MAX_TOKENS
    global UPSTREAM_MIN_INTERVAL_SECONDS, UPSTREAM_RETRIES
    global UPSTREAM_429_FREEZE_SECONDS, UPSTREAM_5XX_FREEZE_SECONDS
    global UPSTREAM_MAX_RETRY_AFTER_SECONDS, UPSTREAM_BACKOFF_INITIAL_SECONDS
    global UPSTREAM_BACKOFF_MAX_SECONDS

    _LOADED_ENV_FILES = load_dotenv_files()

    missing = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or "").strip()]
    keys_str = os.environ.get("AGY_LAUNCH_API_KEYS") or os.environ.get("AGY_LAUNCH_API_KEY") or ""
    API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not API_KEYS:
        missing.append("AGY_LAUNCH_API_KEY (or AGY_LAUNCH_API_KEYS)")

    if missing:
        example = os.path.join(_HERE, ".env.example")
        xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        user_env = os.path.join(xdg, "agy-launch", ".env")
        print("error: missing required configuration:", ", ".join(missing), file=sys.stderr)
        print(file=sys.stderr)
        print("Set them in a .env file or environment variables.", file=sys.stderr)
        print("  project:  ./.env   (copy from .env.example)", file=sys.stderr)
        print(f"  user:     {user_env}", file=sys.stderr)
        print(f"  template: {example}", file=sys.stderr)
        print(file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  cp .env.example .env   # then edit", file=sys.stderr)
        print("  # or:  ./install.sh", file=sys.stderr)
        if _LOADED_ENV_FILES:
            print("loaded env files (still missing keys):", file=sys.stderr)
            for p in _LOADED_ENV_FILES:
                print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    TARGET_BASE_URL = os.environ["AGY_LAUNCH_BASE_URL"].strip().rstrip("/")
    TARGET_MODEL = os.environ["AGY_LAUNCH_MODEL"].strip()
    TARGET_API_KEY = API_KEYS[0]
    AGY_CLI_MODEL = (os.environ.get("AGY_LAUNCH_CLI_MODEL") or "gemini-3.5-flash-low").strip()
    AGY_BIN = (os.environ.get("AGY_BIN") or "agy").strip()
    MODEL_DISPLAY_NAME = (os.environ.get("AGY_LAUNCH_MODEL_DISPLAY_NAME") or TARGET_MODEL).strip()
    MODEL_PROVIDER = (os.environ.get("AGY_LAUNCH_MODEL_PROVIDER") or "").strip().lower()
    UPSTREAM_USER_AGENT = (os.environ.get("AGY_LAUNCH_USER_AGENT") or "curl/8.5.0").strip()
    UPSTREAM_MIN_INTERVAL_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_UPSTREAM_MIN_INTERVAL_SECONDS"), 0.25)
    UPSTREAM_RETRIES = _bounded_int(os.environ.get("AGY_LAUNCH_UPSTREAM_RETRIES"), 2)
    UPSTREAM_429_FREEZE_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_429_FREEZE_SECONDS"), 60.0)
    UPSTREAM_5XX_FREEZE_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_5XX_FREEZE_SECONDS"), 30.0)
    UPSTREAM_MAX_RETRY_AFTER_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_MAX_RETRY_AFTER_SECONDS"), 300.0)
    UPSTREAM_BACKOFF_INITIAL_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_BACKOFF_INITIAL_SECONDS"), 1.0)
    UPSTREAM_BACKOFF_MAX_SECONDS = _bounded_float(os.environ.get("AGY_LAUNCH_BACKOFF_MAX_SECONDS"), 30.0)
    REASONING_EFFORT = (os.environ.get("AGY_LAUNCH_REASONING_EFFORT") or "").strip().lower()
    try:
        MAX_COMPLETION_TOKENS = int(os.environ.get("AGY_LAUNCH_MAX_COMPLETION_TOKENS") or "0")
    except ValueError:
        print("warning: AGY_LAUNCH_MAX_COMPLETION_TOKENS must be an integer", file=sys.stderr)
        MAX_COMPLETION_TOKENS = 0
    try:
        MAX_TOKENS = int(os.environ.get("AGY_LAUNCH_MAX_TOKENS") or "0")
    except ValueError:
        print("warning: AGY_LAUNCH_MAX_TOKENS must be an integer", file=sys.stderr)
        MAX_TOKENS = 0
    DEBUG_DIR = (
        os.environ.get("AGY_LAUNCH_DEBUG_DIR")
        or os.path.join(tempfile.gettempdir(), "agy-launch")
    ).strip()


def _load_json_fixture(name: str, fallback: dict[str, Any]) -> dict[str, Any]:
    search = [_HERE, os.getcwd()]
    if DEBUG_DIR:
        search.append(DEBUG_DIR)
    for base in search:
        path = os.path.join(base, name)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return fallback


def _bootstrap_fixtures() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load Code Assist bootstrap JSON after config (paths may depend on env)."""
    load_resp = _load_json_fixture(
        "load_code_assist_resp.json",
        {
            "currentTier": {
                "id": "free-tier",
                "name": "Antigravity",
                "description": "Custom OpenAI-compatible backend via agy-launch",
                "privacyNotice": {"showNotice": False},
            },
            "allowedTiers": [
                {
                    "id": "free-tier",
                    "name": "Antigravity",
                    "description": "Custom OpenAI-compatible backend via agy-launch",
                    "privacyNotice": {"showNotice": False},
                    "isDefault": True,
                }
            ],
            "cloudaicompanionProject": "agy-launch-local",
            "gcpManaged": False,
        },
    )
    models_resp = _load_json_fixture(
        "models_resp.json",
        {
            "models": {
                "gemini-3.5-flash-low": {
                    "displayName": "Gemini 3.5 Flash (Medium)",
                    "supportsImages": True,
                    "supportsThinking": True,
                    "thinkingBudget": 4000,
                    "recommended": True,
                    "maxTokens": 1048576,
                    "maxOutputTokens": 65536,
                    "tokenizerType": "LLAMA_WITH_SPECIAL",
                    "quotaInfo": {"remainingFraction": 1.0},
                    "model": "MODEL_PLACEHOLDER_M20",
                    "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                    "modelProvider": "MODEL_PROVIDER_GOOGLE",
                }
            },
            "defaultAgentModelId": "gemini-3.5-flash-low",
            "agentModelSorts": [
                {
                    "displayName": "Recommended",
                    "groups": [{"modelIds": ["gemini-3.5-flash-low"]}],
                }
            ],
            "commandModelIds": ["gemini-3.5-flash-low"],
            "tabModelIds": [],
            "imageGenerationModelIds": [],
            "mqueryModelIds": ["gemini-3.5-flash-low"],
            "webSearchModelIds": ["gemini-3.5-flash-low"],
            "commitMessageModelIds": ["gemini-3.5-flash-low"],
            "experimentIds": [],
        },
    )
    return load_resp, _configure_models_resp(models_resp)


def _provider_metadata_overrides() -> dict[str, str]:
    mapping = {
        "google": {
            "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
            "modelProvider": "MODEL_PROVIDER_GOOGLE",
        },
        "openai": {
            "apiProvider": "API_PROVIDER_OPENAI_VERTEX",
            "modelProvider": "MODEL_PROVIDER_OPENAI",
        },
        "anthropic": {
            "apiProvider": "API_PROVIDER_ANTHROPIC_VERTEX",
            "modelProvider": "MODEL_PROVIDER_ANTHROPIC",
        },
    }
    return mapping.get(MODEL_PROVIDER, {})


def _merge_model_ids(values: Any, preferred: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for candidate in [preferred, *(values or [])]:
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _configure_models_resp(raw: dict[str, Any]) -> dict[str, Any]:
    models_resp = deepcopy(raw) if isinstance(raw, dict) else {}
    models = models_resp.get("models")
    if not isinstance(models, dict):
        models = {}
        models_resp["models"] = models

    template: dict[str, Any] | None = None
    if isinstance(models.get(AGY_CLI_MODEL), dict):
        template = models[AGY_CLI_MODEL]
    else:
        default_id = models_resp.get("defaultAgentModelId")
        if isinstance(default_id, str) and isinstance(models.get(default_id), dict):
            template = models[default_id]
        else:
            for value in models.values():
                if isinstance(value, dict):
                    template = value
                    break

    if template is None:
        template = {
            "supportsImages": True,
            "supportsThinking": True,
            "thinkingBudget": 4000,
            "recommended": True,
            "maxTokens": 1048576,
            "maxOutputTokens": 65536,
            "tokenizerType": "LLAMA_WITH_SPECIAL",
            "quotaInfo": {"remainingFraction": 1.0},
            "model": "MODEL_PLACEHOLDER_M20",
            "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
            "modelProvider": "MODEL_PROVIDER_GOOGLE",
        }

    entry = deepcopy(template)
    entry["displayName"] = MODEL_DISPLAY_NAME or TARGET_MODEL or AGY_CLI_MODEL
    entry["recommended"] = True
    entry.update(_provider_metadata_overrides())
    models[AGY_CLI_MODEL] = entry

    models_resp["defaultAgentModelId"] = AGY_CLI_MODEL
    for key in (
        "commandModelIds",
        "tabModelIds",
        "imageGenerationModelIds",
        "mqueryModelIds",
        "webSearchModelIds",
        "commitMessageModelIds",
    ):
        models_resp[key] = _merge_model_ids(models_resp.get(key), AGY_CLI_MODEL)

    agent_model_sorts = models_resp.get("agentModelSorts")
    if not isinstance(agent_model_sorts, list) or not agent_model_sorts:
        agent_model_sorts = [{"displayName": "Recommended", "groups": [{"modelIds": []}]}]
        models_resp["agentModelSorts"] = agent_model_sorts

    first_sort = agent_model_sorts[0] if isinstance(agent_model_sorts[0], dict) else {}
    if not isinstance(agent_model_sorts[0], dict):
        agent_model_sorts[0] = first_sort
    groups = first_sort.get("groups")
    if not isinstance(groups, list) or not groups:
        groups = [{"modelIds": []}]
        first_sort["groups"] = groups
    first_group = groups[0] if isinstance(groups[0], dict) else {}
    if not isinstance(groups[0], dict):
        groups[0] = first_group
    first_group["modelIds"] = _merge_model_ids(first_group.get("modelIds"), AGY_CLI_MODEL)

    return models_resp


# Placeholders replaced in main() after load_config().
LOAD_RESP: dict[str, Any] = {}
MODELS_RESP: dict[str, Any] = {}


def _debug_write(name: str, obj: Any) -> None:
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(obj, (bytes, bytearray)):
                f.write(obj.decode("utf-8", errors="replace"))
            elif isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def read_body(handler: BaseHTTPRequestHandler) -> bytes:
    te = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        body = bytearray()
        rfile = handler.rfile
        while True:
            line = rfile.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                chunk_size = int(line.split(b";")[0], 16)
            except ValueError:
                break
            if chunk_size == 0:
                # trailer
                while True:
                    trailer = rfile.readline()
                    if not trailer or trailer in (b"\r\n", b"\n"):
                        break
                break
            body.extend(rfile.read(chunk_size))
            rfile.read(2)  # CRLF
        return bytes(body)

    length = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(length) if length > 0 else b""


def unwrap_generate_request(raw: dict[str, Any]) -> dict[str, Any]:
    """agy wraps the GenerateContent payload under top-level `request`."""
    if isinstance(raw.get("request"), dict) and (
        "contents" in raw["request"] or "systemInstruction" in raw["request"]
    ):
        return raw["request"]
    return raw


def gemini_to_openai_messages(req: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    sys_inst = req.get("systemInstruction")
    if isinstance(sys_inst, dict):
        parts = sys_inst.get("parts") or []
        sys_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p)
        if sys_text.strip():
            messages.append({"role": "system", "content": sys_text})

    for msg in req.get("contents") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        if role == "model":
            role = "assistant"

        parts = msg.get("parts") or []
        content_text = ""
        tool_calls: list[dict[str, Any]] = []
        tool_responses: list[dict[str, Any]] = []

        for p in parts:
            if not isinstance(p, dict):
                continue
            if "text" in p and p["text"] is not None:
                content_text += str(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"] or {}
                name = fc.get("name") or "tool"
                args = fc.get("args", {})
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": fc.get("id") or name,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )
            elif "functionResponse" in p:
                fr = p["functionResponse"] or {}
                name = fr.get("name") or "tool"
                resp = fr.get("response", {})
                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": fr.get("id") or name,
                        "content": resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False),
                    }
                )

        openai_msg: dict[str, Any] = {"role": role, "content": content_text}
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
            if not content_text:
                openai_msg["content"] = None
        # Skip completely empty user turns (can happen with metadata-only parts)
        if openai_msg.get("content") or openai_msg.get("tool_calls"):
            messages.append(openai_msg)

        if tool_responses:
            messages.extend(tool_responses)

    return messages


def gemini_allowed_tool_names(req: dict[str, Any]) -> set[str]:
    cfg = req.get("toolConfig")
    if not isinstance(cfg, dict):
        return set()
    fn_cfg = cfg.get("functionCallingConfig")
    if not isinstance(fn_cfg, dict):
        return set()
    names = fn_cfg.get("allowedFunctionNames") or []
    return {str(name) for name in names if str(name).strip()}


def gemini_tools_to_openai(req: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    allowed_names = gemini_allowed_tool_names(req)
    for t in req.get("tools") or []:
        if not isinstance(t, dict):
            continue
        for fd in t.get("functionDeclarations") or []:
            if not isinstance(fd, dict):
                continue
            name = fd.get("name")
            if not name:
                continue
            if allowed_names and name not in allowed_names:
                continue
            params = fd.get("parameters") or {"type": "object", "properties": {}}
            # OpenAI expects JSON Schema; Gemini sometimes uses uppercase types.
            params = _normalize_schema(params)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fd.get("description") or "",
                        "parameters": params,
                    },
                }
            )
    return out


def _normalize_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            else:
                out[k] = _normalize_schema(v)
        return out
    if isinstance(node, list):
        return [_normalize_schema(x) for x in node]
    return node


def gemini_tool_choice_to_openai(req: dict[str, Any], tools: list[dict[str, Any]]) -> Any:
    if not tools:
        return None
    cfg = req.get("toolConfig")
    if not isinstance(cfg, dict):
        return None
    fn_cfg = cfg.get("functionCallingConfig")
    if not isinstance(fn_cfg, dict):
        return None

    mode = str(fn_cfg.get("mode") or "").upper()
    if mode == "NONE":
        return "none"
    if mode == "ANY":
        allowed_names = gemini_allowed_tool_names(req)
        if len(allowed_names) == 1:
            return {"type": "function", "function": {"name": next(iter(allowed_names))}}
        return "required"
    if mode in ("AUTO", "VALIDATED"):
        return "auto"
    return None


def build_openai_payload(raw: dict[str, Any]) -> dict[str, Any]:
    req = unwrap_generate_request(raw)
    messages = gemini_to_openai_messages(req)
    if not messages:
        # Hard failure surface: empty messages always 400/500 on most gateways
        messages = [{"role": "user", "content": "(empty request)"}]

    payload: dict[str, Any] = {
        "model": TARGET_MODEL,
        "messages": messages,
        "stream": True,
    }

    tools = gemini_tools_to_openai(req)
    if tools:
        payload["tools"] = tools
        tool_choice = gemini_tool_choice_to_openai(req, tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    gen = req.get("generationConfig") or {}
    if isinstance(gen, dict):
        max_out = gen.get("maxOutputTokens")
        if isinstance(max_out, int) and max_out > 0:
            payload["max_tokens"] = min(max_out, 16384)
        temp = gen.get("temperature")
        if isinstance(temp, (int, float)):
            payload["temperature"] = temp

    if "reasoning_effort" not in payload and REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT
    if "max_completion_tokens" not in payload and MAX_COMPLETION_TOKENS > 0:
        payload["max_completion_tokens"] = MAX_COMPLETION_TOKENS
    if "max_tokens" not in payload and MAX_TOKENS > 0:
        payload["max_tokens"] = MAX_TOKENS

    return payload


class TranslationProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet by default; set AGY_LAUNCH_VERBOSE=1 to log.
        if os.environ.get("AGY_LAUNCH_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        # Health / discovery
        if self.path in ("/", "/healthz"):
            self._send_json({"ok": True, "model": TARGET_MODEL, "base": TARGET_BASE_URL})
            return
        self._send_json({})

    def do_POST(self) -> None:
        body = read_body(self)
        path = self.path or ""

        if "loadCodeAssist" in path:
            self._send_json(LOAD_RESP)
            return
        if "fetchAvailableModels" in path:
            self._send_json(MODELS_RESP)
            return
        if "retrieveUserQuotaSummary" in path:
            self._send_json({"quotaSummary": {"remainingCredits": 999999}})
            return
        if "listExperiments" in path:
            self._send_json({"experiments": []})
            return
        if "fetchUserInfo" in path:
            self._send_json({"userEmail": "agy-launch@local", "userName": "agy-launch"})
            return
        if "streamGenerateContent" in path or "generateContent" in path:
            self._handle_generate(body, stream="streamGenerateContent" in path or True)
            return

        # Unknown RPC — empty success so agy does not crash on optional calls.
        self._send_json({})

    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _handle_generate(self, raw_body: bytes, stream: bool = True) -> None:
        try:
            raw = json.loads(raw_body.decode("utf-8") or "{}")
        except Exception as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, status=400)
            return

        _debug_write("incoming_request.json", raw)
        openai_payload = build_openai_payload(raw)
        _debug_write("outgoing_openai_request.json", openai_payload)

        api_url = f"{TARGET_BASE_URL}/chat/completions"

        response = None
        last_error = None
        max_attempts = max(len(API_KEYS), 1) + max(0, UPSTREAM_RETRIES)
        for attempt in range(max_attempts):
            active_key = get_active_key()
            req = urllib.request.Request(
                api_url,
                data=json.dumps(openai_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {active_key}",
                    "Accept": "text/event-stream" if stream else "application/json",
                    "User-Agent": UPSTREAM_USER_AGENT,
                },
                method="POST",
            )
            try:
                _wait_for_upstream_slot()
                response = urllib.request.urlopen(req, timeout=600)
                break
            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                frozen_for = mark_key_failed(active_key, exc.code, retry_after)
                last_error = exc
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if exc.code == 429:
                    cooldown = _clamp_retry_delay(
                        _parse_retry_after(retry_after),
                        min(frozen_for or UPSTREAM_429_FREEZE_SECONDS, _retry_backoff_seconds(attempt)),
                    )
                    _apply_upstream_cooldown(cooldown, "429 rate limit")
                elif 500 <= exc.code <= 599:
                    _apply_upstream_cooldown(_retry_backoff_seconds(attempt), f"{exc.code} upstream error")
                if retryable and attempt < max_attempts - 1:
                    continue
                if exc.code in (401, 402) and attempt < min(len(API_KEYS), max_attempts) - 1:
                    continue
                break
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    _apply_upstream_cooldown(_retry_backoff_seconds(attempt), "transport error")
                    continue
                break

        if not response:
            err_msg = str(last_error)
            status_code = 502
            if isinstance(last_error, urllib.error.HTTPError):
                status_code = last_error.code
                try:
                    err_msg = last_error.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            print(f"[agy-launch] Forwarding request failed: {err_msg}", file=sys.stderr)
            self._send_json({"error": f"Upstream error: {err_msg}"}, status=status_code)
            return

        try:
            with response:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                accumulated_tool_calls: dict[int, dict[str, str]] = {}
                for raw_line in response:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if not line.startswith(b"data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str.decode("utf-8"))
                    except Exception:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or choice.get("message") or {}

                    # Normal text
                    content = delta.get("content")
                    if content:
                        self._send_gemini_text(content)

                    # Some gateways put reasoning separately
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning and os.environ.get("AGY_LAUNCH_FORWARD_REASONING"):
                        self._send_gemini_text(reasoning)

                    # Streaming tool calls
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index") or 0)
                        slot = accumulated_tool_calls.setdefault(idx, {"name": "", "args_str": "", "id": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args_str"] += fn["arguments"]

                # Emit completed tool calls (Gemini style, after text stream)
                for idx in sorted(accumulated_tool_calls):
                    tc = accumulated_tool_calls[idx]
                    try:
                        args = json.loads(tc["args_str"] or "{}")
                    except Exception:
                        args = {"_raw": tc["args_str"]}
                    name = tc["name"] or "tool"
                    function_call: dict[str, Any] = {
                        "name": name,
                        "args": args,
                    }
                    if tc["id"]:
                        function_call["id"] = tc["id"]
                    self._send_chunk(
                        {
                            "response": {
                                "candidates": [
                                    {
                                        "content": {
                                            "role": "model",
                                            "parts": [
                                                {
                                                    "functionCall": function_call
                                                }
                                            ],
                                        }
                                    }
                                ]
                            }
                        }
                    )

                self._send_chunk(
                    {
                        "response": {
                            "candidates": [
                                {
                                    "content": {"role": "model", "parts": []},
                                    "finishReason": "STOP",
                                }
                            ]
                        }
                    }
                )
                self.wfile.write(b"0\r\n\r\n")
                try:
                    self.wfile.flush()
                except Exception:
                    pass

        except urllib.error.HTTPError as api_err:
            err_body = ""
            try:
                err_body = api_err.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            _debug_write(
                "failed_request.json",
                {
                    "url": api_url,
                    "payload": openai_payload,
                    "error": str(api_err),
                    "response_body": err_body,
                },
            )
            # If headers not yet sent, fall back to JSON error
            try:
                self._send_gemini_error(f"Upstream API Error: {api_err}\n{err_body}")
            except Exception:
                pass
        except Exception as api_err:
            _debug_write(
                "failed_request.json",
                {
                    "url": api_url,
                    "payload": openai_payload,
                    "error": str(api_err),
                },
            )
            try:
                self._send_gemini_error(f"Upstream API Error: {api_err}")
            except Exception:
                pass

    def _send_gemini_text(self, text: str) -> None:
        self._send_chunk(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": text}],
                            }
                        }
                    ]
                }
            }
        )

    def _send_gemini_error(self, msg: str) -> None:
        # Best-effort SSE error so the TUI shows something.
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
        except Exception:
            pass
        self._send_chunk(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": msg}],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            }
        )
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def _send_chunk(self, obj: Any) -> None:
        data = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(f"{len(data):X}\r\n".encode("utf-8") + data + b"\r\n")
        self.wfile.flush()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _normalize_cli_args(args: list[str]) -> list[str]:
    """Translate agy-launch convenience flags to agy flags."""
    normalized: list[str] = []
    saw_dangerous = False
    passthrough = False
    for arg in args:
        if passthrough:
            normalized.append(arg)
            continue
        if arg == "--":
            normalized.append(arg)
            passthrough = True
            continue
        if arg == "--yolo":
            if not saw_dangerous:
                normalized.append("--dangerously-skip-permissions")
                saw_dangerous = True
            continue
        if arg == "--dangerously-skip-permissions":
            if not saw_dangerous:
                normalized.append(arg)
                saw_dangerous = True
            continue
        normalized.append(arg)
    return normalized


def main() -> None:
    global LOAD_RESP, MODELS_RESP

    load_config()
    LOAD_RESP, MODELS_RESP = _bootstrap_fixtures()

    port = int(os.environ.get("AGY_LAUNCH_PORT") or 0) or find_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), TranslationProxy)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    env = os.environ.copy()
    env["CLOUD_CODE_URL"] = f"http://127.0.0.1:{port}"
    # Prevent accidental double-proxy if someone nests launches
    env.pop("AGY_LAUNCH_NESTED", None)

    args = _normalize_cli_args(sys.argv[1:])
    # Default to a known agy model id; upstream still uses TARGET_MODEL.
    if "--model" not in args:
        args = ["--model", AGY_CLI_MODEL, *args]

    if os.environ.get("AGY_LAUNCH_VERBOSE"):
        env_note = ", ".join(_LOADED_ENV_FILES) if _LOADED_ENV_FILES else "(none)"
        print(
            f"[agy-launch] proxy=http://127.0.0.1:{port} "
            f"upstream={TARGET_BASE_URL}/chat/completions "
            f"upstream_model={TARGET_MODEL} cli_model={AGY_CLI_MODEL}\n"
            f"[agy-launch] env files: {env_note}",
            file=sys.stderr,
        )

    try:
        res = subprocess.run([AGY_BIN, *args], env=env)
        sys.exit(res.returncode)
    except FileNotFoundError:
        print(f"error: cannot find agy binary ({AGY_BIN})", file=sys.stderr)
        sys.exit(127)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
