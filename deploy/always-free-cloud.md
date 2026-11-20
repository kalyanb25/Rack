# Always-on for $0: Oracle Cloud Always Free

This gives you a server that never sleeps, running the same rack stack, reachable privately from your phone and laptop over Tailscale. Train on Kaggle, serve from here.

**What you get:** one Arm VM with 2 cores and 12 GB of RAM (Oracle cut the free allowance from 4 cores / 24 GB in mid-2026), 200 GB of block storage, no GPU.

**The strings:**
- Signing up needs a credit card for identity verification. Always Free resources aren't charged.
- CPU only. A 4B model replies at a few words per second. `qwen3:1.7b` is snappier, and 8B+ is too slow to enjoy.
- Popular regions often say "out of capacity" for Arm VMs. Your home region is permanent and Always Free compute has to live there, so pick it deliberately.
- Oracle can reclaim instances that look idle: over 7 days, CPU (95th percentile), network *and* memory all under 20%.
- Oracle already cut these limits once in 2026 and could do it again. Keep your model projects in git so you can move to any other box in minutes.

## 1. Create the VM

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/) and choose your home region.
2. In the console go to **Compute → Instances → Create instance**:
   - **Image:** Canonical Ubuntu 24.04 (the aarch64 build)
   - **Shape:** Ampere `VM.Standard.A1.Flex`, **2 OCPUs, 12 GB memory**
   - **Boot volume:** 100 GB is plenty (the free allowance is 200 GB in total)
   - **SSH key:** paste your public key (`~/.ssh/id_ed25519.pub`; make one with `ssh-keygen -t ed25519` if needed)
3. If you get "out of host capacity", try another availability domain or try again later. Upgrading the account to Pay As You Go helps with capacity and still doesn't charge for Always Free resources, but anything you create beyond the free limits is billed, so set a budget alert if you do that.

Leave the network security rules alone: only SSH open to the internet. Tailscale handles everything else without opening ports.

## 2. Install rack on it

Get the platform onto the VM:

```bash
ssh ubuntu@<public-ip>
git clone https://github.com/kalyanb25/Rack.git ~/rack
```

(If you make the repo private, run `sudo apt-get install -y gh && gh auth login` first and use
`gh repo clone kalyanb25/Rack ~/rack`. Or copy it from your laptop: `scp -r rack ubuntu@<public-ip>:~/`.)

On the VM:

```bash
bash ~/rack/deploy/setup-ubuntu.sh     # Docker, Tailscale, Python tools
exit                                    # log out and back in so docker works without sudo
ssh ubuntu@<public-ip>

sudo tailscale up                       # sign in with the same account as your phone/laptop
cd ~/rack
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
rack up
```

Keep the model loaded so replies start instantly. In `~/rack/.env` set:

```
OLLAMA_KEEP_ALIVE=-1
```

then run `rack up` again. With the 4B model loaded, memory use sits at roughly a third of the VM, above Oracle's 20% idle threshold.

## 3. Reach it from anywhere (privately)

```bash
sudo tailscale serve --bg 3000                 # web UI  → https://<vm-name>.<tailnet>.ts.net
sudo tailscale serve --bg --https=8443 11434   # API     → https://<vm-name>.<tailnet>.ts.net:8443
```

Your tailnet name is in the Tailscale admin console under **DNS** (turn on MagicDNS and HTTPS certificates there if they aren't on already). Open the web UI address on your phone, sign up (the first account is the admin), and add it to your home screen.

## 4. Drive it from your laptop

In the `rack.toml` on your laptop:

```toml
[server]
ollama_url = "https://<vm-name>.<tailnet>.ts.net:8443"
webui_url  = "https://<vm-name>.<tailnet>.ts.net"
```

Now `rack build`, `rack train`, `rack deploy` and `rack eval` on your laptop act on the VM. Fine-tunes come back from Kaggle to your laptop and get uploaded straight to the VM. `rack status` confirms it's talking to the right server.

## Keeping it healthy

- Containers restart on their own after reboots (Docker starts at boot, and the services use `restart: unless-stopped`).
- Update to the latest Ollama and Open WebUI: `cd ~/rack && docker compose pull && rack up`.
- Back up: your model projects live in git. Models on the server can always be rebuilt with `rack build`, or re-imported from `.rack/jobs/*/output/*.gguf` on your laptop.
