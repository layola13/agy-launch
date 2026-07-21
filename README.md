# agy-launch

Launcher for [agy](https://antigravity.google) (Antigravity CLI) that routes model traffic through a local translation proxy to a **standard OpenAI Chat Completions** endpoint.

**No secrets or private endpoints are hard-coded.** Configure everything via `.env` or environment variables so this repo is safe to publish.

## Copyright / 免责声明

Copyright (c) 2026 agy-launch contributors

**仅供学习与研究使用；其他用途后果自负。**

This project is for learning and research only. Any other use is at your own risk.
See [COPYRIGHT](COPYRIGHT).

## What it does

1. Loads config from env / `.env` (required: base URL, model, API key).
2. Starts a local HTTP proxy on a free port.
3. Sets `CLOUD_CODE_URL` so `agy` talks to the proxy instead of Google Code Assist.
4. Translates `streamGenerateContent` → `POST {base}/chat/completions`.
5. Streams the OpenAI SSE response back in Gemini-compatible SSE chunks.

## Quick start

```bash
# 1. Install (wrapper in ~/.local/bin + user config template)
./install.sh



# 2. Edit user config (or create a project ./.env)
$EDITOR ~/.config/agy-launch/.env

# 3. Ensure PATH and run
export PATH="$HOME/.local/bin:$PATH"
agy-launch
agy-launch --print "hello" --print-timeout 2m
```
`./install.sh` also auto-installs the upstream CLI when it is missing (use `--skip-cli` to opt out).

### Project-local `.env` (recommended for per-repo settings)

```bash
cp .env.example .env
# edit .env — this file is gitignored
agy-launch
```

The launcher/package `.env` next to `main.py` is loaded first for agy-launch settings, so it wins over cwd, parent, and user config files.
For agy-launch-managed keys (`AGY_LAUNCH_*` and `AGY_BIN`), `.env` values also override stale shell exports from an existing session.

## Configuration

### Required

| Variable | Meaning |
|----------|---------|
| `AGY_LAUNCH_BASE_URL` | OpenAI-compatible base, e.g. `https://gateway.example/v1` |
| `AGY_LAUNCH_MODEL` | Model name sent to `/chat/completions` |
| `AGY_LAUNCH_API_KEY` | Bearer token |

### Optional

| Variable | Meaning |
|----------|---------|
| `AGY_LAUNCH_CLI_MODEL` | Passed to `agy --model` and promoted as the default advertised agy model id (default `gemini-3.5-flash-low`; must be a known agy id) |
| `AGY_LAUNCH_MODEL_DISPLAY_NAME` | Optional display name shown for that agy model id; defaults to `AGY_LAUNCH_MODEL` |
| `AGY_LAUNCH_MODEL_PROVIDER` | Optional provider metadata for the advertised model: `google`, `openai`, or `anthropic` |
| `AGY_LAUNCH_USER_AGENT` | Optional upstream HTTP `User-Agent`; defaults to `curl/8.5.0` because some gateways block Python urllib’s default signature |
| `AGY_LAUNCH_UPSTREAM_MIN_INTERVAL_SECONDS` | Minimum spacing between upstream requests across proxy threads (default `0.25`) |
| `AGY_LAUNCH_UPSTREAM_RETRIES` | Extra retry attempts for retryable upstream failures after key rotation (default `2`) |
| `AGY_LAUNCH_429_FREEZE_SECONDS` | Default per-key freeze after 429 when `Retry-After` is absent (default `60`) |
| `AGY_LAUNCH_MAX_RETRY_AFTER_SECONDS` | Cap for `Retry-After` / global cooldown waits (default `300`) |
| `AGY_LAUNCH_BACKOFF_INITIAL_SECONDS` | Initial retry backoff for transient upstream failures (default `1`) |
| `AGY_LAUNCH_BACKOFF_MAX_SECONDS` | Maximum retry backoff for transient upstream failures (default `30`) |
| `AGY_BIN` | Path to `agy` (default `agy`) |
| `AGY_LAUNCH_ENV` | Force a specific env file path |
| `AGY_LAUNCH_VERBOSE=1` | Log proxy requests |
| `AGY_LAUNCH_PORT` | Fixed proxy port |
| `AGY_LAUNCH_DEBUG_DIR` | Directory for debug JSON dumps (default `$TMPDIR/agy-launch`) |

### Where `.env` is loaded (priority high → low)

1. `AGY_LAUNCH_ENV` if set
2. Package directory `.env` (repo-local / launcher-local, next to `main.py`)
3. `./.env` or `./.agy-launch.env` (cwd)
4. Parent directories (up to 6 levels)
5. `~/.config/agy-launch/.env`
6. `~/.agy-launch.env`

Copy the template from [`.env.example`](.env.example).

### Two “model” names

| Layer | Variable | Purpose |
|-------|----------|---------|
| Upstream | `AGY_LAUNCH_MODEL` | Real model string on the OpenAI API |
| CLI | `AGY_LAUNCH_CLI_MODEL` / `--model` | agy allowlist id (e.g. `gemini-3.5-flash-low`) |

agy rejects unknown `--model` values even if the proxy advertises them. Prefer known ids from `agy models`; the proxy still sends `AGY_LAUNCH_MODEL` upstream.

If your real upstream model is not Gemini, keep using a known agy CLI id in `AGY_LAUNCH_CLI_MODEL`, then map it to your gateway model with:

```env
AGY_LAUNCH_MODEL=your-real-upstream-model
AGY_LAUNCH_CLI_MODEL=gemini-3.5-flash-low
AGY_LAUNCH_MODEL_DISPLAY_NAME=your-real-upstream-model
AGY_LAUNCH_MODEL_PROVIDER=openai
```

That keeps agy happy with a known allowlist id while the proxy sends your actual model upstream and advertises a less misleading label/provider in `fetchAvailableModels`.

## install.sh

```bash
./install.sh              # install wrapper + create user .env from example if missing
./install.sh --force-env  # overwrite user .env from .env.example
./install.sh --no-env     # only install binary
./install.sh --link       # symlink to the repo launcher instead of generating a wrapper script
./install.sh --bin-dir ~/bin --config-dir ~/.config/agy-launch
```

If `agy-launch` fails with `python3: can't open file '/root/.local/bin/main.py'`,
you have an old copied/symlinked launcher that resolved its own directory as
`~/.local/bin`. Reinstall with `./install.sh` or use the fixed repo launcher,
which resolves symlinks before locating `main.py`.

## Usage (same flags as agy)

```bash
agy-launch
agy-launch --continue
agy-launch --print "hi" --print-timeout 2m
agy-launch --yolo
agy-launch --model gemini-3.5-flash-low
AGY_LAUNCH_VERBOSE=1 agy-launch
```

`--yolo` is a shortcut for `--dangerously-skip-permissions` and is forwarded to `agy` as that flag.

Direct run without install:

```bash
python3 main.py --print "hi"
```

## Layout

```
agy-launch/
  main.py                 # launcher + proxy
  install.sh              # install to ~/.local/bin + user .env
  .env.example            # public template (safe to commit)
  .gitignore              # ignores .env and secrets
  models_resp.json        # fetchAvailableModels catalog
  load_code_assist_resp.json
  README.md
```

## Debug

With `AGY_LAUNCH_VERBOSE=1`, stderr shows the proxy URL and which env files were loaded.
Optional dumps under `AGY_LAUNCH_DEBUG_DIR`:

- `incoming_request.json`
- `outgoing_openai_request.json`
- `failed_request.json`

## Security notes for public repos

- Do **not** commit `.env` (gitignored).
- Rotate any key that was ever committed historically.
- Prefer user config `~/.config/agy-launch/.env` with mode `600` (install.sh sets this when it creates the file).
