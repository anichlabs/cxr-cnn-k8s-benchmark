# DEPLOY.md

Operational reference for the live demo at
[cxr.anichlabs.com](https://cxr.anichlabs.com).

This document captures the production deployment of the
`cxr-cnn-k8s-benchmark` FastAPI service: what runs where, how to
reproduce it, how to operate it day to day, and what to do when
something breaks. It is written as the single source of truth for a
solo operator (Christopher Anich, AnichLabs OÜ) but should be enough
for any sufficiently senior collaborator to take over.

If you only need to redeploy after pushing code changes, jump to
[Day-to-day operations](#6-day-to-day-operations).

---

## 1. Overview

| Field | Value |
|---|---|
| Public URL | `https://cxr.anichlabs.com` |
| Server provider | Hetzner Cloud |
| Server type | CX33 (4 vCPU shared Intel, 8 GB RAM, 80 GB SSD) |
| Region | Nuremberg (NBG1) |
| OS | Ubuntu 24.04 LTS |
| Cost | ~6.49 EUR / month |
| Hosted apps on this server | CXR demo only (room for more) |
| Architecture | linux/amd64 (x86_64) |
| Container runtime | Podman 4.9, rootless |
| Orchestrator | `podman compose` (Compose v2 via the Docker plugin) |
| Reverse proxy | Caddy 2.11 (host-installed, system service) |
| TLS | Let's Encrypt, auto-renewed by Caddy |
| Process supervision | systemd, user-level (linger enabled) |
| Firewall | UFW, default-deny |
| Brute-force protection | fail2ban on SSH |
| Auto patches | unattended-upgrades (security only) |
| Timezone | UTC |

Operating principle: small VPS, boring tools, container-per-app, all
fronted by one Caddy. Designed to add more portfolio apps as
subdomains without rearchitecting anything.

---

## 2. Architecture

```text
Internet
   |
   v
[Hetzner CX33 VPS, public IPv4: 178.105.172.65]
   |
   +-- UFW: allow 22, 80, 443
   +-- Caddy 2.11 (host service, runs as caddy user)
   |     Listens on :80 (redirects to :443)
   |     Listens on :443 (HTTPS, HTTP/2, HTTP/3)
   |     Reverse-proxies cxr.anichlabs.com -> 127.0.0.1:8001
   |     Manages Let's Encrypt cert lifecycle
   |
   +-- Podman rootless (runs as chris)
         |
         +-- container cxr-cnn-k8s-benchmark-cxr-api-1
               image: localhost/cxr-api:cpu (~1.57 GB)
               port: 127.0.0.1:8001:8000 (bound to localhost only)
               command: uvicorn app.main:app --workers 1
               volumes: ./experiments/checkpoints:/app/checkpoints:ro
               healthcheck: HTTP GET /health every 30s
               restart: unless-stopped
```

The container is **bound to 127.0.0.1**, so the only path in is
through Caddy. The wider internet cannot reach :8001 directly.

---

## 3. Repository layout relevant to deployment

```text
cxr-cnn-k8s-benchmark/
├── podman-compose.yml                          # base compose (shared)
├── deployment/
│   ├── compose.dev.yml                         # dev override (public port, --reload)
│   ├── compose.prod.yml                        # prod override (localhost-only, healthcheck, restart)
│   └── api/
│       ├── Containerfile.cpu                   # CPU image recipe
│       ├── requirements-cpu.txt                # Python deps
│       ├── wheels-cpu/                         # offline-install wheels (~243 MB, gitignored)
│       ├── app/                                # FastAPI source
│       ├── config/                             # preprocessing_config.json
│       └── demo/                               # static front-end + samples
│           ├── index.html
│           └── samples/
│               ├── thumbs/                     # 90 x 96px WebP thumbnails
│               └── fulls/                      # 90 x 512px JPEG fulls
├── experiments/
│   └── checkpoints/                            # 8 .pt files (~240 MB, gitignored)
└── DEPLOY.md                                   # this file
```

Two folders are intentionally gitignored and must be transferred
out-of-band to a fresh server: `experiments/checkpoints/` and
`deployment/api/wheels-cpu/`. Both are committed in spirit (paths and
documentation are tracked) but not as binary blobs.

---

## 4. Stack rationale

| Layer | Choice | Why |
|---|---|---|
| Server | Hetzner CX33 | Cheapest x86 VPS that comfortably hosts PyTorch CPU inference plus a handful of small apps. EU jurisdiction. |
| OS | Ubuntu 24.04 LTS | Stable, well-documented, default Podman 4.9 in repos. |
| Container runtime | Podman, rootless | No daemon, smaller blast radius if the user is compromised. `podman kube generate` produces real K8s YAML, easing a future migration. |
| Orchestrator | `podman compose` with Compose v2 plugin | Step on the path from local dev to potential K8s. Keeps the YAML portable. |
| Reverse proxy | Caddy on the host | Zero-config HTTPS. Caddy needs ports 80/443; running it on the host avoids the rootless container port-binding fiddle. |
| Process supervisor | systemd user units (lingering) | Standard, observable, plays nicely with rootless Podman. |
| TLS | Let's Encrypt via Caddy ACME | Free, automated, industry standard. |

The design avoids exotic choices on purpose. The portfolio story is
"I run a production-grade multi-tenant platform with boring tools",
not "I shipped the latest hot framework".

---

## 5. First-time server setup (from clean Ubuntu)

Run these once on a fresh Hetzner CX33 with Ubuntu 24.04 selected
and the SSH public key registered in the Hetzner project's Security
section before server creation.

### 5.1 Initial SSH

```bash
# from your laptop
ssh -i ~/.ssh/portfolio_ed25519 root@<server-ip>
hostnamectl set-hostname portfolio-01
exit
```

### 5.2 Create a sudo user and copy the key

```bash
ssh -i ~/.ssh/portfolio_ed25519 root@<server-ip>
adduser chris            # press Enter through the metadata, set a strong password
usermod -aG sudo chris
mkdir -p /home/chris/.ssh
cp /root/.ssh/authorized_keys /home/chris/.ssh/authorized_keys
chown -R chris:chris /home/chris/.ssh
chmod 700 /home/chris/.ssh
chmod 600 /home/chris/.ssh/authorized_keys
exit
```

Test that `chris` can log in and use sudo before going further.

### 5.3 Lock down SSH

As `chris`:

```bash
sudo tee /etc/ssh/sshd_config.d/99-hardening.conf > /dev/null <<'CFG'
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
CFG
sudo sshd -t                       # validate
sudo systemctl reload ssh
```

From a third terminal verify that root login is rejected:

```bash
ssh -i ~/.ssh/portfolio_ed25519 root@<server-ip>
# expected: Permission denied (publickey).
```

### 5.4 Firewall

```bash
sudo apt update && sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'
sudo ufw enable
```

### 5.5 Brute-force protection

```bash
sudo apt install -y fail2ban
sudo tee /etc/fail2ban/jail.local > /dev/null <<'CFG'
[DEFAULT]
bantime = 1h
findtime = 10m
maxretry = 5
backend = systemd

[sshd]
enabled = true
CFG
sudo systemctl restart fail2ban
```

### 5.6 Automatic security updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades   # answer Yes
sudo timedatectl set-timezone Etc/UTC
```

### 5.7 Install Podman and the Compose plugin

```bash
sudo apt install -y podman
# Start Podman's user socket
systemctl --user enable --now podman.socket
# Install the Compose v2 binary into the standard plugin path
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# Tell Compose where Podman's socket is
echo 'export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock' >> ~/.bashrc
export DOCKER_HOST=unix:///run/user/1000/podman/podman.sock
```

### 5.8 Install Caddy

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

Write the Caddyfile:

```bash
sudo tee /etc/caddy/Caddyfile > /dev/null <<'CFG'
{
    email chris@anichlabs.com
}

cxr.anichlabs.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8001
    log {
        output file /var/log/caddy/cxr.access.log
        format json
    }
}
CFG
sudo mkdir -p /var/log/caddy
sudo chown caddy:caddy /var/log/caddy
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

DNS must already resolve `cxr.anichlabs.com` to the server's public
IPv4 and IPv6 addresses for Caddy to obtain a certificate from
Let's Encrypt. Set the A and AAAA records in Hetzner DNS (or your DNS
provider) before reloading Caddy.

### 5.9 Enable user-service persistence (linger)

Without this, rootless Podman containers stop when the user logs out
or the server reboots. **This step is mandatory.**

```bash
sudo loginctl enable-linger chris
loginctl show-user chris | grep Linger     # expected: Linger=yes
```

### 5.10 Bring the application onto the server

Generate a deploy key for GitHub (read-only, scoped to this repo).
Add the public key to the repo's Settings > Deploy keys.

```bash
ssh-keygen -t ed25519 -C "github-deploy-cxr@portfolio-01" -f ~/.ssh/github_cxr_deploy -N ""
cat >> ~/.ssh/config <<'CFG'

Host github.com-cxr
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_cxr_deploy
    IdentitiesOnly yes
CFG
ssh -T git@github.com-cxr    # expect: 'You've successfully authenticated'
mkdir -p ~/apps && cd ~/apps
git clone git@github.com-cxr:anichlabs/cxr-cnn-k8s-benchmark.git
```

Transfer the gitignored assets from your laptop:

```bash
# from your laptop, in the repo root
rsync -avP experiments/checkpoints/ portfolio-01:~/apps/cxr-cnn-k8s-benchmark/experiments/checkpoints/
rsync -avP deployment/api/wheels-cpu/ portfolio-01:~/apps/cxr-cnn-k8s-benchmark/deployment/api/wheels-cpu/
```

### 5.11 Build and start the container

Back on the server:

```bash
cd ~/apps/cxr-cnn-k8s-benchmark
podman compose -f podman-compose.yml -f deployment/compose.prod.yml build
podman compose -f podman-compose.yml -f deployment/compose.prod.yml up -d
sleep 70 && podman ps      # expected: Up (healthy)
```

### 5.12 Wire the container into systemd so it survives reboots

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/cxr-api.service <<'CFG'
[Unit]
Description=CXR FastAPI service via podman compose
After=podman.socket network-online.target
Wants=podman.socket network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h/apps/cxr-cnn-k8s-benchmark
Environment=DOCKER_HOST=unix:///run/user/%U/podman/podman.sock
ExecStart=/usr/bin/podman compose -f podman-compose.yml -f deployment/compose.prod.yml up -d
ExecStop=/usr/bin/podman compose -f podman-compose.yml -f deployment/compose.prod.yml down

[Install]
WantedBy=default.target
CFG
systemctl --user daemon-reload
systemctl --user enable --now cxr-api.service
```

Verify by rebooting and confirming the container comes back without
manual intervention.

---

## 6. Day-to-day operations

### 6.1 Deploy a code change

From your laptop, push the change to `origin/main`. Then on the
server:

```bash
cd ~/apps/cxr-cnn-k8s-benchmark
git pull
# Rebuild if Containerfile, requirements, or app source changed
podman compose -f podman-compose.yml -f deployment/compose.prod.yml build
# Always reapply compose so config changes take effect
podman compose -f podman-compose.yml -f deployment/compose.prod.yml up -d
podman ps   # confirm (healthy)
```

Most demo-only changes (HTML, samples) trigger a layer-cached rebuild
of ~30 seconds. Changes that touch dependencies or the base layer
take longer.

### 6.2 Tail logs

```bash
podman logs -f cxr-cnn-k8s-benchmark-cxr-api-1
```

For the reverse proxy:

```bash
sudo tail -f /var/log/caddy/cxr.access.log    # JSON access log
sudo journalctl -u caddy -f                   # service-level events
```

### 6.3 Restart cleanly

```bash
systemctl --user restart cxr-api.service
```

### 6.4 Stop the demo temporarily (without disabling auto-start)

```bash
podman compose -f podman-compose.yml -f deployment/compose.prod.yml down
```

It will come back at the next reboot via the systemd unit.

### 6.5 Free disk by pruning old images

After several rebuilds, dangling layers accumulate:

```bash
podman image prune -f
```

---

## 7. Configuration files at a glance

| Path | Purpose |
|---|---|
| `/etc/ssh/sshd_config.d/99-hardening.conf` | Disables root SSH + password auth |
| `/etc/fail2ban/jail.local` | Bans SSH brute-force attempts |
| `/etc/caddy/Caddyfile` | TLS termination and reverse proxy |
| `/var/log/caddy/cxr.access.log` | Caddy access log, JSON |
| `~/.ssh/config` | `portfolio-01` host alias for SSH and `github.com-cxr` for deploy key |
| `~/.config/systemd/user/cxr-api.service` | Auto-start unit for the container |
| `~/apps/cxr-cnn-k8s-benchmark/podman-compose.yml` | Base compose (shared) |
| `~/apps/cxr-cnn-k8s-benchmark/deployment/compose.prod.yml` | Production overrides (this is the file the systemd unit references) |
| `~/apps/cxr-cnn-k8s-benchmark/deployment/compose.dev.yml` | Local-dev overrides, not used on the server |

---

## 8. DNS, TLS, and ports

| Record | Value |
|---|---|
| `cxr.anichlabs.com` A | `178.105.172.65` |
| `cxr.anichlabs.com` AAAA | `2a01:4f8:1c18:109d::1` |
| TTL | 3600 |
| DNS provider | Hetzner DNS |

Ports open on the firewall: 22, 80, 443. Nothing else.

Caddy obtains and renews the TLS certificate automatically. The
renewal window is 30 days before expiry; Let's Encrypt certs are
valid for 90 days. No human action is required for renewals.

---

## 9. Security posture

| Control | State |
|---|---|
| SSH root login | Disabled |
| SSH password auth | Disabled |
| SSH key | ed25519, dedicated key per server (`portfolio_ed25519`) |
| Sudo user | `chris` (password-protected sudo) |
| Firewall | UFW, default-deny inbound |
| Brute-force | fail2ban (1h ban after 5 failures in 10 min) |
| Automatic OS patches | unattended-upgrades (security pocket only) |
| API exposure | Bound to `127.0.0.1` only; only Caddy reaches it |
| HTTPS | Enforced, HTTP automatically redirects |
| HSTS | Default (Caddy sets `Strict-Transport-Security` for the apex hostname) |
| Container user | Non-root `appuser` inside the image |
| Container runtime | Rootless Podman |
| Secrets in git | None (audited at deploy time) |

Threats *not* mitigated:

- Application-layer abuse such as upload spam or model-inference DoS.
  If the demo gets unwanted traffic, we add a Caddy rate-limit
  directive.
- Supply-chain compromise of pip wheels (mitigated only by the
  one-time offline install).
- Anything inside the model itself (the disclaimer is the legal
  control here, not a technical one).

---

## 10. Backups and disaster recovery

### What is precious

| Item | Where | Recovery |
|---|---|---|
| Code | GitHub (`origin/main`) | `git clone` |
| Model checkpoints | Local laptop and the server | rsync from laptop |
| Sample images | Tracked in git | `git pull` |
| Caddy data (cert + ACME state) | `/var/lib/caddy/` | Caddy will simply re-issue on a new host |
| Server state | Hetzner snapshot (optional, paid) | Snapshot restore in Hetzner Console |

### What is not precious

Logs, container images, build caches. All reproducible.

### Full disaster recovery procedure

If the server is unrecoverable:

1. Create a new CX33 in Hetzner with the same SSH key.
2. Update DNS A and AAAA records to the new IP.
3. Run steps 5.1 through 5.12 above. End-to-end this takes about an
   hour, mostly waiting for the image rebuild.
4. Verify `https://cxr.anichlabs.com/health` returns `200`.

Optional defence in depth: enable Hetzner daily backups (20% surcharge,
about 1.30 EUR/month). With backups, point-in-time recovery is one
click in the console.

---

## 11. Troubleshooting log

Issues encountered during the initial deployment and what fixed them.
Recorded so the same hour is not lost twice.

### 11.1 `podman compose` complained about a missing compose binary

> `Error: looking up compose provider failed`

Cause: Ubuntu's Podman package does not bundle a compose plugin.
Fix: install Compose v2 into `/usr/local/lib/docker/cli-plugins/`
(see step 5.7).

### 11.2 `podman compose ... build` failed with a missing socket

> `failed to connect to the docker API at unix:///run/user/1000/podman/podman.sock`

Cause: Podman's user socket was not running.
Fix: `systemctl --user enable --now podman.socket` and export
`DOCKER_HOST=unix:///run/user/1000/podman/podman.sock`.

### 11.3 Compose merged `ports` lists from base and override, producing a port conflict

> `bind: address already in use`

Cause: Compose v2 *appends* list-typed fields across overrides
instead of replacing them. The base file declared a public-facing
port; the prod override declared a localhost-only port; the runtime
got both.
Fix: refactor the base compose to declare only shared config. Put
the dev-only port in `compose.dev.yml` and the prod-only port in
`compose.prod.yml`. Each environment then defines its own port from
scratch, no merging conflict.

### 11.4 Container started but Caddy returned 502

> Domain unreachable; `/health` returned 502.

Most likely causes, ranked: container exited, container crash
looping, container reachable but stuck.

Diagnostic:

```bash
podman ps -a
podman logs --tail 60 cxr-cnn-k8s-benchmark-cxr-api-1
free -h
```

### 11.5 Container died after the user logged out

Symptom: container `Exited (0)` not long after SSH disconnect.

Cause: rootless Podman lives inside the user's systemd manager,
which is torn down when the user has no active sessions and
`Linger=no`.

Fix: `sudo loginctl enable-linger chris`. This is mandatory for any
rootless server deployment.

### 11.6 Healthcheck always reported `unhealthy` although the API was 200 OK

Cause: the inline-bracketed YAML form
`test: ["CMD","python","-c","import urllib.request, sys; sys.exit..."]`
was re-tokenised by Compose v2 at commas, splitting the Python
one-liner into multiple arguments. `python -c` received only the
bare word `import`, hence the SyntaxError.

Fix: rewrite as a multi-line YAML list under `CMD-SHELL`. Drop the
explicit `sys.exit` chain and rely on `urllib.request.urlopen`
raising on non-2xx, which makes Python exit non-zero automatically.
See the current `compose.prod.yml` for the working form.

### 11.7 Caddy reload failed with permission denied on the log file

Cause: the log file path existed but was owned by root with mode
`0600`, while Caddy runs as the `caddy` system user.

Fix:

```bash
sudo chown caddy:caddy /var/log/caddy
sudo chown caddy:caddy /var/log/caddy/cxr.access.log
```

---

## 12. Known limitations

- Single VPS, no high availability. Acceptable for a demo. If the
  server is unavailable, the demo is unavailable.
- 8 GB RAM is comfortable for the CXR API alone; with several more
  portfolio apps on the same box, a resize to CX43 may become
  necessary. Hetzner resize is online and takes ~2 minutes.
- No application-layer rate limiting yet. Add a `rate_limit`
  directive in the Caddyfile if abuse becomes a problem.
- No off-server log shipping. Logs live on the box. For a demo this
  is fine. For anything sensitive add a small Loki or vector setup.
- The healthcheck verifies HTTP reachability but not model
  correctness. Drift in checkpoint files would be invisible until a
  user hits a broken prediction.

---

## 13. Compliance reminder

The deployed service is a **research demonstration**. It is not a
medical device under EU MDR or any other regulation, has not been
clinically validated, and must not be used for diagnosis, treatment,
or any clinical decision support. The disclaimer in the public UI
makes this explicit. Do not remove it.
