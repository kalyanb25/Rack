# rack

Your own AI model platform. Build custom models, fine-tune them on your data, test them, deploy them, and use them from a web app, your phone, or your own code. It all runs on hardware you control, with open-weight models and open-source software.

```
rack up                 # start the model server + web UI
rack build dc-ops       # make a custom model (v1)
rack train dc-ops       # fine-tune it on your data on a free cloud GPU (v2)
rack eval dc-ops v1 v2  # compare versions on your test cases
rack deploy dc-ops v2   # ship the winner; "dc-ops" in every app now means v2
```

## What "free" means here

No subscription, no per-token bills, no rate limits, and nobody can switch it off: the models are downloaded to your machine and the software is open source. What's left:

- **Hardware and power.** You need a computer to run it. Leaving one on 24/7 costs electricity: watts × 730 hours ÷ 1000 × your price per kWh. A 15 W mini PC at 17¢/kWh is about $2/month, and an 80 W desktop with a GPU idling is about $10/month.
- **Free cloud tiers have limits.** Kaggle's free GPU (2× T4) comes with roughly 30 GPU-hours a week (it floats; `kaggle quota` shows yours), and sessions are capped. Oracle's Always Free ARM VM was cut to 2 cores / 12 GB in mid-2026, runs CPU-only, needs a card to sign up, and idle instances can be reclaimed. See [deploy/always-free-cloud.md](deploy/always-free-cloud.md).
- **Model licenses.** The default base, Qwen3, is Apache 2.0, which is about as no-strings as it gets. Other models come with their own terms, so check before you build on one.
- **Scale.** You're customizing existing open models (1B–30B parameters). Training a GPT-class model from scratch costs millions and no free tier will do it.

## Quickstart

Needs Python 3.10+ and Docker ([Desktop](https://docs.docker.com/get-docker/) on Mac/Windows, Engine on Linux).

```bash
git clone https://github.com/kalyanb25/Rack.git rack
cd rack
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

rack doctor      # what this machine can run, and what to fix
rack up          # starts Ollama + Open WebUI in Docker
rack build dc-ops
```

Open http://localhost:3000, create your account (the first one becomes admin), and pick **dc-ops** from the model list.

On a Mac, install the [Ollama app](https://ollama.com/download) first. Docker on macOS can't use the Apple GPU, and `rack up` detects the app and uses it instead.

## How it works

```
  edit system.md, add examples to data/train.jsonl
                 │
     rack build ─┴─ rack train            new immutable version: name:v1, v2, ...
                 │
     rack eval  (runs automatically)      score on evals/cases.jsonl
                 │
     rack deploy name vN                  "name" now points at vN (rollback any time)
                 │
     use it: web UI · phone · API · rack chat --save  →  collect better examples → repeat
```

Each model is a folder under `models/`:

```
models/dc-ops/
  model.toml        base model, parameters, training + eval settings
  system.md         the model's built-in instructions
  examples.jsonl    optional few-shot conversations baked into the model
  data/train.jsonl  your fine-tuning data
  evals/cases.jsonl test prompts with pass/fail checks
  registry.json     every version, its scores, and what's deployed (written by rack)
```

`rack new my-model` makes a fresh one. Commit the folder to git and you have the full history of your model.

### Two ways to customize

**`rack build`** puts your system prompt, parameters and examples on top of the base model. It takes seconds and needs no GPU. Start here: a good system prompt gets you surprisingly far.

**`rack train`** is a real fine-tune (LoRA, via [Unsloth](https://unsloth.ai)) on `data/train.jsonl`. It's worth it once you have 100+ high-quality examples of exactly the answers you want: a voice, a format, domain knowledge, refusals. The result is exported to GGUF and becomes a new version like any other.

### Evals

Every new version is scored automatically against `evals/cases.jsonl`. One JSON object per line:

```json
{"id": "pdu-safety", "prompt": "Walk me through swapping a PDU.", "contains_any": ["MOP", "change"], "regex": "lockout|LOTO"}
```

Checks: `contains`, `contains_any`, `not_contains`, `regex`, `max_words`, `min_words`, `equals`, `json`, and `judge` (a rubric graded by another model you set in `[eval] judge`). A case passes when all its checks pass. `rack eval dc-ops v1 v2 v3` prints them side by side. New versions don't replace the deployed one until you say so.

## Fine-tuning for free on Kaggle

One-time setup:

1. Make a [Kaggle](https://www.kaggle.com) account and verify your phone number in Settings (required for GPUs and internet access in notebooks).
2. Settings → API → create a token. Save it to `~/.kaggle/access_token`, or set `KAGGLE_API_TOKEN`.
3. `pip install kaggle`
4. Put your Kaggle username in `rack.toml` under `[kaggle]`.

Then:

```bash
rack data check dc-ops     # validates the data and eval cases
rack train dc-ops
```

rack uploads your data as a **private** Kaggle dataset, starts a private GPU job on the free T4 machine, waits (expect roughly 20–40 minutes for a 4B model), downloads the GGUF, imports it as a new version, and evaluates it. You can Ctrl+C while it runs and pick it up later with `rack train --resume .rack/jobs/<job>`. Each run uses part of your weekly quota.

**Straight from GitHub, nothing to install:** add `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` as repository secrets (Settings → Secrets and variables → Actions), then Actions → **train** → **Run workflow**. GitHub sends the job to Kaggle, scores the result against the prompt-only version, and gives you the GGUF as a downloadable artifact. Import it with `rack import dc-ops <file>.gguf`. Training data has to be committed for this, so keep anything confidential (work procedures, customer data) out of a public repo.

**Own NVIDIA GPU (8 GB+):** install Unsloth in a venv (`pip install unsloth`, or follow [their guide](https://unsloth.ai/docs) on Windows/WSL), set `[local_train] python` in `rack.toml` to that venv's Python, then `rack train dc-ops --backend local`.

**Anywhere else** (Colab, a rented GPU): `rack train dc-ops --dry-run` packages the job folder. Run `python train_unsloth.py --job data/job.json --out output` inside it, copy `output/` back, then `rack train --resume <job folder>`.

Training data is chat-format JSONL. ShareGPT, prompt/response and Alpaca formats are converted automatically:

```json
{"messages": [{"role": "user", "content": "What does N+1 mean?"}, {"role": "assistant", "content": "You have what you need (N) plus one spare..."}]}
```

The system prompt from `system.md` is added to every example automatically, so the model learns in the same context it's served in.

## Using your models

- **Web UI:** Open WebUI at http://localhost:3000. Chats, files, model switching. It works on a phone too.
- **Terminal:** `rack chat dc-ops`. Add `--save` to keep conversations in `data/inbox.jsonl`, then move the good ones into `train.jsonl` for the next fine-tune.
- **Your code:** Ollama speaks the OpenAI API, so any OpenAI SDK works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")
reply = client.chat.completions.create(
    model="dc-ops",
    messages=[{"role": "user", "content": "What does N+1 mean?"}],
)
print(reply.choices[0].message.content)
```

### From your phone or anywhere

Use [Tailscale](https://tailscale.com) (its Personal plan is free for non-commercial use). Install it on the machine running rack and on your phone/laptop, sign in to the same account on both, then on the server:

```bash
sudo tailscale serve --bg 3000                 # web UI at https://<machine>.<tailnet>.ts.net
sudo tailscale serve --bg --https=8443 11434   # API at https://<machine>.<tailnet>.ts.net:8443
```

Nothing is exposed to the public internet this way. Open the web UI address on your phone and add it to your home screen.

### For apps outside your network

Ollama has no authentication, so never put port 11434 on the internet. Use `rack up --gateway` instead. It starts an API-key gateway on port 8000, generates a key into `.env`, and apps call it with `Authorization: Bearer <RACK_API_KEY>`. Make it public with Tailscale Funnel (`sudo tailscale funnel --bg --https=10000 8000` gives you `https://<machine>.<tailnet>.ts.net:10000`) or a Cloudflare Tunnel. Tailscale's free plan is for personal use, so for a commercial app use a Cloudflare Tunnel or a paid plan.

## Running it 24/7

| Where | Cost | Notes |
|---|---|---|
| Your laptop/desktop | power only | Fine for building and testing. Not 24/7 if it sleeps. |
| An old PC or mini PC at home | power only (~$2–10/month) | The sweet spot. Use [deploy/setup-ubuntu.sh](deploy/setup-ubuntu.sh) on Ubuntu. |
| Oracle Cloud Always Free | $0 | Always on, but CPU-only with 2 cores, so small models and slower replies. [Guide](deploy/always-free-cloud.md). |

All three run the same stack. Containers restart on their own after a reboot as long as Docker starts at boot. To manage a remote box from your laptop, point `server.ollama_url` in `rack.toml` at it (for example the Tailscale HTTPS address above). rack then uploads new versions straight to it.

## Commands

| Command | What it does |
|---|---|
| `rack doctor` | Hardware check + which models fit + training options |
| `rack up` / `rack down` | Start/stop the stack (`--gateway`, `--gpu`, `--no-gpu`, `--native`) |
| `rack status` | Server health and every model's versions, deploys and scores |
| `rack new <name>` | New model project (`--base`, `--hf`) |
| `rack build <name>` | New version from system prompt + settings |
| `rack train <name>` | New version from a fine-tune (`--backend kaggle\|local`, `--dry-run`, `--resume DIR`) |
| `rack import <name> <file.gguf>` | New version from any GGUF file (e.g. downloaded from Hugging Face) |
| `rack eval <name> [v1 v2 ...]` | Score versions, side by side (`--all` for every version) |
| `rack deploy <name> [version]` | Point `<name>` at a version (default: latest) |
| `rack rollback <name>` | Back to the previously deployed version |
| `rack versions <name>` | Version history with scores and training stats |
| `rack chat <name> [version]` | Terminal chat (`--save` collects training data) |
| `rack data check <name>` | Validate training data and eval cases |
| `rack pull <model>` / `rack models` | Download a library model / list everything on the server |
| `rack rm <name> <version>` | Delete a version (refuses if it's deployed) |

`build`, `train` and `import` also take `--notes "what changed"`, `--deploy` and `--no-eval`.

## Choosing a base model

`rack doctor` recommends one for your hardware. Rough guide for the default Q4 builds:

| Memory free for the model | Ollama tag | Notes |
|---|---|---|
| ~2.5 GB | `qwen3:1.7b` | Tiny, runs almost anywhere |
| ~4 GB | `qwen3:4b-instruct-2507-q4_K_M` | Default. Good all-rounder, fine-tunes on the free T4 |
| ~7 GB | `qwen3:8b` | Smarter; thinks before answering |
| ~11 GB | `qwen3:14b` | Strong general model |
| ~16 GB | `gpt-oss:20b` | OpenAI's open-weight reasoning model |

For `rack train`, keep `[base] hf` the same family as `[base] ollama` (the default pair is the same Qwen3-4B-Instruct model). Unsloth also supports Llama 3.x, Gemma, Phi-4, Mistral and more. Browse [ollama.com/library](https://ollama.com/library) and [huggingface.co/unsloth](https://huggingface.co/unsloth).

## Troubleshooting

- **`rack up` fails with a GPU error:** install the NVIDIA Container Toolkit ([Ollama's Docker guide](https://docs.ollama.com/docker)), or run `rack up --no-gpu`.
- **Can't reach Ollama:** `rack status` shows the URL it's using. Check `server.ollama_url` in `rack.toml`, then `docker compose logs ollama`.
- **A fine-tuned version rambles or never stops:** that's almost always a chat-template mismatch. Training writes a `Modelfile` with the right template and rack imports it automatically. If you import a GGUF by hand, pass `--modelfile`.
- **Kaggle job runs out of memory:** lower `max_seq_length` or `batch_size` in `[train]`, or use a smaller base.
- **Kaggle says GPU or internet isn't allowed:** verify your phone number in Kaggle settings.
- **403 from the API through Tailscale while using the native Ollama app:** the app only accepts local hostnames by default. Set `OLLAMA_HOST=0.0.0.0` for the app ([Ollama FAQ](https://docs.ollama.com/faq)) or use the Docker Ollama (`rack up --docker-ollama`).
- **Kaggle's install recipe broke:** Unsloth occasionally changes its tested versions. Copy the new install lines from their Kaggle notebook into `[train] install = [...]` in `model.toml`.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite runs the whole loop (new, build, eval, deploy, rollback, import, chat, train on Kaggle and locally, plus failure paths) against a stand-in Ollama server built from Ollama's API docs and a fake Kaggle CLI. The GPU training script follows Unsloth's official Kaggle recipe for Qwen3-4B-Instruct, but it can only really be tested on a GPU, so your first `rack train` is its real test.

## Built on

[Ollama](https://ollama.com) (serving), [Open WebUI](https://openwebui.com) (web app), [Unsloth](https://unsloth.ai) (fine-tuning), [Kaggle](https://www.kaggle.com) (free GPUs), [Caddy](https://caddyserver.com) (gateway), [Tailscale](https://tailscale.com) (private access). The rack CLI itself uses only the Python standard library.

MIT licensed.
