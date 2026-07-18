# agy-launch

Launcher for [agy](https://antigravity.google) (Antigravity CLI) that routes model traffic through a local translation proxy to a **standard OpenAI Chat Completions** endpoint.

**No secrets or private endpoints are hard-coded.** Configure everything via `.env` or environment variables so this repo is safe to publish.

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

### Project-local `.env` (recommended for per-repo settings)

```bash
cp .env.example .env
# edit .env — this file is gitignored
agy-launch
```

When both project `.env` and `~/.config/agy-launch/.env` exist, **project wins** for keys defined there.  
Shell `export` always wins over any `.env` file.

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
| `AGY_LAUNCH_CLI_MODEL` | Passed to `agy --model` (default `gemini-3.5-flash-low`; must be a known agy id) |
| `AGY_BIN` | Path to `agy` (default `agy`) |
| `AGY_LAUNCH_ENV` | Force a specific env file path |
| `AGY_LAUNCH_VERBOSE=1` | Log proxy requests |
| `AGY_LAUNCH_PORT` | Fixed proxy port |
| `AGY_LAUNCH_DEBUG_DIR` | Directory for debug JSON dumps (default `$TMPDIR/agy-launch`) |

### Where `.env` is loaded (priority high → low)

1. `AGY_LAUNCH_ENV` if set  
2. `./.env` or `./.agy-launch.env` (cwd)  
3. Parent directories (up to 6 levels)  
4. `~/.config/agy-launch/.env`  
5. `~/.agy-launch.env`  
6. Package directory `.env` (usually only for local dev)

Copy the template from [`.env.example`](.env.example).

### Two “model” names

| Layer | Variable | Purpose |
|-------|----------|---------|
| Upstream | `AGY_LAUNCH_MODEL` | Real model string on the OpenAI API |
| CLI | `AGY_LAUNCH_CLI_MODEL` / `--model` | agy allowlist id (e.g. `gemini-3.5-flash-low`) |

agy rejects unknown `--model` values even if the proxy advertises them. Prefer known ids from `agy models`; the proxy still sends `AGY_LAUNCH_MODEL` upstream.

## install.sh

```bash
./install.sh              # install wrapper + create user .env from example if missing
./install.sh --force-env  # overwrite user .env from .env.example
./install.sh --no-env     # only install binary
./install.sh --link       # symlink to main.py instead of a wrapper script
./install.sh --bin-dir ~/bin --config-dir ~/.config/agy-launch
```

## Usage (same flags as agy)

```bash
agy-launch
agy-launch --continue
agy-launch --print "hi" --print-timeout 2m
agy-launch --model gemini-3.5-flash-low
AGY_LAUNCH_VERBOSE=1 agy-launch
```

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
