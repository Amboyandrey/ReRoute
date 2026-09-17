# Phase 6 — Kubernetes deployment

## What's actually running

A local k3s cluster deploys the **gateway** and **router** (both CPU-only)
via the Helm chart in `infra/helm/reroute`, with `kube-prometheus-stack`
(Prometheus + Grafana) installed alongside and the dashboard from
`infra/grafana/reroute-dashboard.json` auto-imported. That part works
cleanly and is the real, demoable piece of this phase.

The **cheap tier stays outside the cluster**, as a plain vLLM process on
the host GPU (`infra/vllm/`), with the in-cluster gateway pointed at it.
`vllm.enabled: true` in the Helm chart (a GPU pod inside Kubernetes) is
implemented but **not running** — see below.

## Why the in-cluster GPU pod is off

Getting a GPU pod scheduled and running inside k3s on this machine surfaced
a real chain of issues, each fixed in the repo along the way:

1. **`nvidia-container-toolkit` wasn't installed** — Docker's `--gpus`
   flag silently had nothing to attach to. Fixed by installing it and
   configuring both the Docker and containerd runtimes
   (`infra/k3s/bootstrap.sh`).
2. **Configuring the runtime for an already-running k3s and restarting it
   corrupted flannel** — the node got stuck `NotReady` with
   `cni plugin not initialized` and never recovered on its own. A clean
   `k3s-uninstall.sh` + reinstall (with the runtime already configured
   *before* k3s's first boot) fixed it immediately. `bootstrap.sh` installs
   k3s only once, after the runtime is configured, specifically to avoid
   this.
3. **The NVIDIA device plugin's own pod needs `runtimeClassName: nvidia`**
   — it's not the cluster's default runtime, so the plugin couldn't load
   `libnvml` until patched. Baked into `bootstrap.sh`.
4. **No `startupProbe`/`readinessProbe` on the vLLM Deployment** meant
   Kubernetes reported the pod `Ready` the instant the container process
   started, long before the model had finished loading — masking real
   problems as "already working." Fixed in
   `infra/helm/reroute/templates/vllm-deployment.yaml`.
5. **Default `RollingUpdate` strategy deadlocks on a single-GPU node** —
   it tries to schedule a second pod before killing the first, which can
   never succeed with exactly one `nvidia.com/gpu`. Changed to `Recreate`.
6. **The `vllm/vllm-openai:latest` image crashed on the first real
   request** with `torch.AcceleratorError: CUDA error: an illegal
   instruction was encountered` during CUDA graph replay — on this exact
   GPU (RTX 4060 Laptop) + driver, even though the identical vLLM version
   ran fine bare-metal outside Docker. Mitigated with `--enforce-eager`
   (skips CUDA graph capture) in `values.yaml`, untested whether that
   fully resolves it.
7. **Running this GPU pod caused a hard system crash** (the whole laptop
   rebooted, not just a process crashing) after a training run, a
   bare-metal vLLM instance, and a containerized vLLM instance had all
   stressed the same 8GB laptop GPU in succession — including one stretch
   where two vLLM processes were competing for the same GPU
   simultaneously. That's a genuine hardware/thermal signal, not just a
   software bug, and is why this phase stops here rather than continuing
   to debug the CUDA crash.

None of this is a dead end — it's a fairly normal list of "things that
bite you the first time you put a consumer GPU inside a container
orchestrator," and every fix above is real and stays in the repo. It's
just not something to keep iterating on with a laptop's only GPU, which
also drives the display and has no thermal headroom to spare.

## Deploying this for real (on a VM, not this laptop)

The path documented here is exactly what a cloud GPU VM removes the risk
from: a dedicated GPU with no display contention, proper cooling, and (on
most providers) an image or driver stack already validated for
containerized CUDA workloads.

1. **Provision a GPU VM.** Nebius (this project's original target — see
   `docs/PLAN.md`) or any provider with an L4/L40/A10-class GPU. One GPU is
   enough; more VRAM than the 4060's 8GB gives real headroom.
2. **Install Docker + k3s + nvidia-container-toolkit** using
   `infra/k3s/bootstrap.sh` as-is — it already encodes the correct order
   (runtime configured before k3s's first boot) that this laptop run
   didn't get right the first time.
3. **Build and push real images** instead of the local
   `docker save | k3s ctr images import` workaround used for this laptop
   run: tag `reroute-gateway`/`reroute-router` for a real registry (GHCR,
   Docker Hub, or the provider's own), push, and switch
   `image.pullPolicy` to `IfNotPresent` (already the default) or `Always`.
4. **Deploy**: `helm upgrade --install reroute infra/helm/reroute --set
   vllm.enabled=true`. Watch `kubectl logs deployment/reroute-vllm -c
   vllm` through model load before trusting the readiness probe — Phase 6
   already fixed the probe timing, so `1/1 Running` should now mean what
   it says.
5. **If the CUDA illegal-instruction crash reappears** on real hardware
   (it may well have been specific to this laptop's driver/GPU
   combination and not recur on a data-center GPU): try dropping
   `--enforce-eager`, pin `vllm.image.tag` to a specific released version
   instead of `latest`, or check the vLLM GitHub issues for that image tag
   against your specific GPU architecture.
6. **Point real provider keys** at the `reroute-provider-keys` Secret
   (`ANTHROPIC_API_KEY`, `NEBIUS_API_KEY`) to light up the `mid` and
   `strong` tiers, which have only ever run against empty placeholders in
   this project so far.

## Current cluster state (this laptop, at close of this session)

- k3s running, single node
- `reroute-gateway`, `reroute-router`: `Running`, CPU-only
- `monitoring` namespace: Prometheus, Grafana, Alertmanager all `Running`
- `reroute-vllm`: scaled to 0 replicas (no GPU pod running)
- Cheap tier served by the bare-metal `infra/vllm/` process on the host,
  which the gateway's `cheap` tier config still points at
