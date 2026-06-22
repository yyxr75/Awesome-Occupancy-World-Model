# Awesome Occupancy World Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated, **direction-indexed** reading list for **occupancy (occ) world models** in autonomous driving.

Most existing lists are chronological dumps. This one is organized by **what problem each paper attacks**, so you can answer *"if I want to work on X, what should I read?"* directly. The taxonomy follows three battles every conditional generative field fights — **representation**, **mechanism**, and **long-horizon & measurement** — unfolded into six concrete directions plus the foundational backbone and adjacent fields.

> **Scope.** Core = generative/forecasting world models in the 3D **occupancy** space. We also track the **video** and **LiDAR/point-cloud** world-model lines, because occ borrows machinery from both.

---

## Contents

- [The leaderboard (and how to read it honestly)](#the-leaderboard-and-how-to-read-it-honestly)
- [Foundational occ world models](#foundational-occ-world-models)
  - [A. Token-AR / GPT-style](#a-token-ar--gpt-style-generative)
  - [B. Diffusion](#b-diffusion-based-generative)
  - [C. Next-scale (VAR)](#c-next-scale-var--efficient-tokenization-generative)
  - [D. Vision-centric + planning](#d-vision-centric-forecasting--planning)
  - [E. Self-supervised / pretraining](#e-self-supervised--pretraining-world-models)
- [Direction 1 — Representation](#direction-1--representation)
  - [1a. Grid latent: VQ / VAE / tri-plane / next-scale](#1a-grid-latent-vq--vae--tri-plane--next-scale)
  - [1b. Gaussian / object-centric](#1b-gaussian--object-centric)
- [Direction 2 — Mechanism (generation paradigm)](#direction-2--mechanism-generation-paradigm)
- [Direction 3 — Long-horizon & anti-drift](#direction-3--long-horizon--anti-drift)
- [Direction 4 — Measurement & weak-sensor conditioning](#direction-4--measurement--weak-sensor-conditioning)
- [Direction 5 — Control & motion source](#direction-5--control--motion-source)
- [Direction 6 — Evaluation, benchmarks & datasets](#direction-6--evaluation-benchmarks--datasets)
- [Related: video driving world models](#related-video-driving-world-models)
- [Related: LiDAR / point-cloud world models](#related-lidar--point-cloud-world-models)
- [Surveys & other lists](#surveys--other-lists)
- [Two caveats before you compare numbers](#two-caveats-before-you-compare-numbers)

---

## The leaderboard (and how to read it honestly)

4D occ forecasting on **Occ3D-nuScenes** (GT-occ input, 2s history → 3s future), as reported in the GenieDrive paper under one protocol:

| Method | Mech. | 1s | 2s | 3s | **Avg mIoU** | FPS | Params |
|---|---|---|---|---|---|---|---|
| DOME '24 | clip-diffusion | 35.1 | 25.9 | 20.3 | 27.1 | 6.5 | 397M |
| COME '25 | clip-diff + control | 42.8 | 33.0 | 27.0 | 34.2 | 0.3 | 692M |
| I²-World '25 | token-AR (encoder-decoder) | 47.6 | 38.6 | 33.0 | 39.7 | 37 | 22.7M |
| **GenieDrive '26** | VAE + AR + E2E | 50.5 | 41.5 | 35.8 | **42.6** | 41 | 3.5M |

⚠️ The aggregate mIoU is **static-dominated**: background voxels swamp the moving agents. A model can top this table while failing the dynamic objects that matter most. See [Two caveats](#two-caveats-before-you-compare-numbers).

---

## Foundational occ world models

The canonical generative/forecasting backbone, grouped by paradigm. This is the master list; the Direction sections below re-slice these by research axis.

### A. Token-AR / GPT-style generative

- **OccWorld** — Learning a 3D Occupancy World Model for Autonomous Driving. *ECCV'24*. [[paper]](https://arxiv.org/abs/2311.16038) · [[code]](https://github.com/wzzheng/OccWorld) — GPT-style token-AR; jointly forecasts scene evolution + ego trajectory. The seed of the field.
- **OccLLaMA** — An Occupancy-Language-Action Generative World Model. *arXiv'24*. [[paper]](https://arxiv.org/abs/2409.03272) — folds occupancy + language + action into one autoregressive vocabulary.
- **Occ-LLM** — Enhancing Autonomous Driving with Occupancy-Based LLMs. *CVPR'25*. [[paper]](https://arxiv.org/abs/2502.06419) — couples an LLM with occupancy for forecasting + reasoning.
- **RenderWorld** — World Model with Self-Supervised 3D Label. *arXiv'24*. [[paper]](https://arxiv.org/abs/2409.11356) — separate air / non-air grid tokenization for fine-grained 3D modeling.

### B. Diffusion-based generative

- **OccSora** — 4D Occupancy Generation Models as World Simulators. *arXiv'24*. [[paper]](https://arxiv.org/abs/2405.20337) · [[code]](https://github.com/wzzheng/OccSora) — 4D scene tokenizer + DiT; predicts occ evolution given arbitrary trajectories.
- **DOME** — Taming Diffusion into a High-Fidelity Controllable Occupancy World Model. *arXiv'24*. [[paper]](https://arxiv.org/abs/2410.10429) — continuous Occ-VAE + spatio-temporal diffusion transformer; the diffusion baseline most follow-ups build on.
- **COME** — Adding Scene-Centric Forecasting Control to Occupancy World Model. *NeurIPS'25*. [[paper]](https://arxiv.org/abs/2506.13260) — factors ego-motion out via scene-centric coordinates, injects a scene-centric forecast as control into a frozen DOME. Opened the "explicit-injection" line.
- **UniScene** — Unified Occupancy-centric Driving Scene Generation. *CVPR'25*. [[paper]](https://arxiv.org/abs/2412.05435) — one occ-centric hierarchy generating occ + multi-view video + LiDAR; occ-DiT first, then Gaussian-rendered guidance.
- **GenieDrive** — Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation. *CVPR'26*. [[paper]](https://arxiv.org/abs/2512.12751) — tri-plane VAE + Mutual Control Attention + end-to-end training; current top of the Occ board, then renders occ → multi-view video.

### C. Next-scale (VAR) / efficient-tokenization generative

- **I²-World** — Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting. *ICCV'25*. [[paper]](https://arxiv.org/abs/2507.09144) — decouples intra-scene (multi-scale residual VQ) and inter-scene (temporal) tokenizers; encoder-decoder, real-time.
- **OccVAR** — Scalable 4D Occupancy Prediction via Next-Scale Prediction. *ICLR'25*. [[paper]](https://openreview.net/forum?id=X2HnTFsFm8) — coarse-to-fine temporal next-scale prediction; ego movement prepended for controllable generation.
- **OccTENS** — 3D Occupancy World Model via Temporal Next-Scale Prediction. *arXiv'25*. [[paper]](https://arxiv.org/abs/2509.03887) — decomposes into spatial scale-by-scale + temporal scene-by-scene; pose-controllable long-term generation.
- **T³Former** — Delta-Triplane Transformers as Occupancy World Models. *arXiv'25*. [[paper]](https://arxiv.org/abs/2503.07338) — pretrained triplanes + multi-scale transformers; predicts *triplane changes* for forecasting + planning, real-time.
- **AFMWM** — high-ratio Swin-VAE compressor + efficient latent forecasting. *2025*. — efficiency-oriented latent occ world model.

### D. Vision-centric forecasting + planning

- **Drive-OccWorld** — Vision-Centric 4D Occupancy Forecasting and Planning via World Models. *AAAI'25 (Oral)*. [[paper]](https://arxiv.org/abs/2408.14197) · [[code]](https://github.com/yuyang-cloud/Drive-OccWorld) — action-conditioned occ + flow forecasting wired directly into an end-to-end planner.
- **DFIT-OccWorld** — Efficient Occupancy World Model via Decoupled Dynamic Flow + Image-Assisted Training. *arXiv'24*. [[paper]](https://arxiv.org/abs/2412.13772) — decoupled dynamic flow; non-autoregressive, warp static + predict dynamic.
- **OccProphet** — Pushing the Efficiency Frontier of Camera-Only 4D Occupancy Forecasting. *ICLR'25*. [[paper]](https://arxiv.org/abs/2502.15180) — Observer–Forecaster–Refiner framework; lightweight, camera-only.
- **PreWorld** — Semi-Supervised Vision-Centric 3D Occupancy World Model. *ICLR'25*. [[paper]](https://arxiv.org/abs/2502.07309) — 2D-render supervision (RGB / density / semantic) + state-conditioned forecasting, end-to-end from images.
- **IR-WM** — Vision-Centric 4D Occupancy Forecasting and Planning via Implicit Residual World Models. *arXiv'25*. [[paper]](https://arxiv.org/abs/2510.16729) — predicts residual world change rather than the full next state.
- **NeMo** — Neural Volumetric World Models for Autonomous Driving. *ECCV'24*. — volumetric WM with image-prediction + motion-flow modules for planning.

### E. Self-supervised / pretraining world models

*Occupancy forecasting as a pretext task — learn from raw LiDAR/video without dense occ labels.*

- **UnO** — Unsupervised Occupancy Fields for Perception and Forecasting. *CVPR'24*. [[paper]](https://arxiv.org/abs/2406.08691) — continuous 4D occupancy field, self-supervised from LiDAR (pos/neg sampling); strong transfer + high mover recall.
- **ViDAR** — Visual Point Cloud Forecasting enables Scalable Autonomous Driving. *CVPR'24*. [[paper]](https://arxiv.org/abs/2312.17655) — history encoder + latent rendering + future decoder; pretrains visual encoders by forecasting point clouds.
- **4D-Occ-Forecasting** — Self-supervised via future point clouds + differentiable depth rendering. *CVPR'23*. [[paper]](https://arxiv.org/abs/2310.11239) — future point clouds as a proxy for future occ.
- **UniWorld** — Autonomous Driving Pre-training via World Models. *arXiv'23*. [[paper]](https://arxiv.org/abs/2308.07234) — multi-frame point-cloud fusion for 4D occ-reconstruction pretraining.
- **DriveWorld** — 4D Pre-trained Scene Understanding via World Models. *CVPR'24*. — memory-augmented spatio-temporal pretraining for downstream planning.
- **Occupancy World Model for Robots** — *arXiv'25*. [[paper]](https://arxiv.org/abs/2505.05512) — carries the occ-WM recipe beyond driving into robotics.

---

## Direction 1 — Representation

*What do we generate in?* Two axes: **grid vs. object-centric**, and **explicit vs. compressed latent**. The whole leaderboard above lives in one cell (compressed grid latent); the Gaussian line is a genuinely different axis.

### 1a. Grid latent: VQ / VAE / tri-plane / next-scale

- **OccWorld** — discrete VQ scene tokens. [[paper]](https://arxiv.org/abs/2311.16038) · [[code]](https://github.com/wzzheng/OccWorld)
- **DOME** — continuous VAE latent (keeps detail, bulkier). [[paper]](https://arxiv.org/abs/2410.10429)
- **I²-World** — multi-scale residual VQ + next-scale (VAR) flavor. [[paper]](https://arxiv.org/abs/2507.09144)
- **GenieDrive** — **tri-plane** VAE; latent ≈ 58% of prior size. [[paper]](https://arxiv.org/abs/2512.12751)
- **UniScene** — temporal-aware continuous occ-VAE (vs. VQ tokenizers). [[paper]](https://arxiv.org/abs/2412.05435)
- *Open:* a **forecast-optimal** (not recon-optimal) latent; a latent that explicitly splits static/dynamic; whether the latent is even the ceiling (test with a latent oracle).

### 1b. Gaussian / object-centric

A sparse, continuous, object-centric alternative — movers fall out of the representation for free.

- **GaussianFormer** — Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction. *ECCV'24*. [[paper]](https://arxiv.org/abs/2405.17429) · [[code]](https://github.com/huang-yh/GaussianFormer) — first object-centric Gaussian occ representation; 75–82% less memory than voxel grids.
- **GaussianFormer-2** — Probabilistic Gaussian Superposition for Efficient 3D Occupancy. *arXiv'24*. [[paper]](https://arxiv.org/abs/2412.04384) — probabilistic Gaussian mixture for geometry + semantics.
- **GaussianFormer3D** — Multi-Modal Gaussian-Based Occupancy with 3D Deformable Attention. *arXiv'25*. [[paper]](https://arxiv.org/abs/2505.10685) — LiDAR-camera fusion in the Gaussian representation.
- **GaussianWorld** — Gaussian World Model for Streaming 3D Occupancy Prediction. *CVPR'25*. [[paper]](https://arxiv.org/abs/2412.10373) — recasts occ prediction as 4D forecasting in Gaussian space; **decomposes evolution into static ego-align + dynamic local motion + new-region completion** (streaming / perception-oriented).
- **GEM** — Gaussian Evolution Model for Occupancy Forecasting and Motion Planning. *arXiv'26*. [[paper]](https://arxiv.org/abs/2605.17682) — **non-autoregressive**: query continuous-time 4D Gaussians at any timestamp, then splat. No step-by-step rollout ⇒ no rollout-drift accumulation. *(Name clash: not the video WM "GEM".)*
- *Open:* the **sparse-latent** corner (sparse + compressed) is barely touched; Gaussian world models are vision-centric and not directly comparable to the GT-occ-token board.

---

## Direction 2 — Mechanism (generation paradigm)

*How do we generate?* Video's arc — GAN → token-AR → clip-diffusion → AR-diffusion → **AR-diffusion + anti-drift**. Occ is roughly 1.5 generations behind and has only just entered AR-diffusion.

- **Token-AR (GPT-style):** OccWorld [[paper]](https://arxiv.org/abs/2311.16038), DrivingWorld [[paper]](https://arxiv.org/abs/2412.19505), I²-World [[paper]](https://arxiv.org/abs/2507.09144).
- **Clip-diffusion:** OccSora [[paper]](https://arxiv.org/abs/2405.20337), DOME [[paper]](https://arxiv.org/abs/2410.10429), COME [[paper]](https://arxiv.org/abs/2506.13260), UniScene [[paper]](https://arxiv.org/abs/2412.05435).
- **Discrete diffusion (tokens):** Copilot4D [[paper]](https://arxiv.org/abs/2311.01017) — recasts MaskGIT as discrete diffusion (point-cloud world model; the mechanism is portable to occ tokens).
- **Non-AR / continuous-time:** GEM [[paper]](https://arxiv.org/abs/2605.17682) — bypass rollout entirely.
- *Open:* port a **named anti-drift mechanism** (below) into occ forecasting — nobody has yet.

---

## Direction 3 — Long-horizon & anti-drift

The biggest open battle. The leaderboard's hard 1s→3s decay is this problem. The honest test is **strict autoregressive rollout** (feed the model its own predictions back), not teacher-forced numbers.

**The anti-drift toolbox from video — ready to port:**

- **Diffusion Forcing** — Next-token Prediction Meets Full-Sequence Diffusion. *NeurIPS'24*. [[paper]](https://arxiv.org/abs/2407.01392) — independent per-token noise levels; rolls out past the training horizon where baselines diverge.
- **Self Forcing** — Bridging the Train-Test Gap in Autoregressive Video Diffusion. *arXiv'25*. [[paper]](https://arxiv.org/abs/2506.08009) · [[project]](https://self-forcing.github.io/) — train on the model's **own** rollouts (KV-cached) to kill exposure bias.
- **Rolling Forcing** — Autoregressive Long Video Diffusion in Real Time. *ICLR'26*. [[paper]](https://arxiv.org/abs/2509.25161) · [[code]](https://github.com/TencentARC/RollingForcing) — rolling-window denoising + attention-sink anchor; multi-minute, minimal drift.
- **Causal Forcing** — AR Diffusion Distillation for Real-Time Interactive Video. *arXiv'26*. [[paper]](https://arxiv.org/abs/2602.02214) — distillation done right for streaming.
- **BAgger** — Backwards Aggregation for Mitigating Drift in AR Video Diffusion. *arXiv'25*. [[paper]](https://arxiv.org/abs/2512.12080) — drift-mitigation via backward aggregation.

**Long-horizon in occ / LiDAR specifically:**

- **OccSim** — Multi-kilometer Simulation with Long-horizon Occupancy World Models. *arXiv'26*. [[paper]](https://arxiv.org/abs/2603.28887) — pushes occ rollout to km-scale.
- **Orbis** — Overcoming Challenges of Long-Horizon Prediction in Driving World Models. *arXiv'25*. — explicitly targets long-horizon driving prediction.
- *Open:* a **dynamic × horizon** diagnostic under strict AR rollout; non-AR (GEM) vs. anti-drift as two distinct routes.

---

## Direction 4 — Measurement & weak-sensor conditioning

*How do we stay faithful to what was actually sensed?* The column occ mostly ignores. Most occ world models are generation-only (no real-sensor correction loop). The self-supervised forecasting line is the closest existing thread.

- **4D-Occ-Forecasting** — uses future point clouds as a proxy for future occ via differentiable depth rendering (self-supervised). [[paper]](https://arxiv.org/abs/2310.11239)
- **ViDAR** — Visual Point Cloud Forecasting enables Scalable Autonomous Driving. *CVPR'24*. — latent rendering for self-supervised 4D occ pretraining.
- **GaussianFormer3D** — LiDAR-camera fusion conditioning in Gaussian occ. [[paper]](https://arxiv.org/abs/2505.10685)
- **Copilot4D** — unsupervised LiDAR world model via discrete diffusion. *ICLR'24*. [[paper]](https://arxiv.org/abs/2311.01017)
- **Foundational LiDAR World Models via Latent Flow Matching** — *arXiv'25*. [[paper]](https://arxiv.org/abs/2506.23434) — efficient latent flow matching for LiDAR world modeling.
- *Open:* injecting a **measured mover signal** (e.g. radar Doppler) as a condition; pose-complementary injection (coarse sensor injection adds nothing once ego-pose is given); a measurement-anchoring / inverse-problem (DPS-style) view of occ.
- *Known trap:* a sensor adapter can silently **proxy for missing ego-motion** — always ablate against a pose-on prior.

---

## Direction 5 — Control & motion source

*Who is told to move, and how?* A surprisingly load-bearing axis. Every gain over COME so far is *more/better explicit injection* — none from anti-drift.

- **COME** — scene-centric vs. ego-centric; explicitly factors out ego trajectory. [[paper]](https://arxiv.org/abs/2506.13260)
- **GenieDrive** — Mutual Control Attention models control → occ evolution. [[paper]](https://arxiv.org/abs/2512.12751)
- **Drive-OccWorld** — occupancy-flow control + planning. [[paper]](https://arxiv.org/abs/2408.14197)
- **GaussianWorld** — three-factor evolution (movers explicitly separated). [[paper]](https://arxiv.org/abs/2412.10373)
- **GEM** — per-primitive (per-Gaussian) ego/object velocity split. [[paper]](https://arxiv.org/abs/2605.17682)
- **LatentWorld** — entity-centric latents + per-actor motion diffusion. *(motion source = GT tracks)*
- ⚠️ *The leakage line:* motion computed from **past/conditioning** frames is legitimate context; motion computed from **predicted** frames is leakage (telling the model where every car goes) — the same confound that inflates GT-ego-trajectory methods.

---

## Direction 6 — Evaluation, benchmarks & datasets

Often the highest-leverage move: change *what* we measure.

**Benchmarks / datasets**

- **Occ3D** — A Large-Scale 3D Occupancy Prediction Benchmark. *NeurIPS'23*. [[paper]](https://arxiv.org/abs/2304.14365) — the de-facto Occ3D-nuScenes protocol.
- **Cam4DOcc** — benchmark for camera-only 4D occ forecasting. *CVPR'24*.
- **Occ4Cast** — LiDAR-based 4D occ forecasting benchmark.

**Open evaluation directions**

- **Dynamic × horizon** — report mover-only accuracy as a function of rollout time (aggregate mIoU hides mover collapse).
- **GT-mover oracle** — inject perfect mover motion to upper-bound the recoverable headroom; the residual is the genuinely hard part (non-constant velocity, turns, newly-entering movers).
- **Physics-consistency metrics** — velocity continuity, no teleportation, kinematic feasibility. "Physics-aware" is currently a branding word, rarely measured.

---

## Related: video driving world models

The source of the anti-drift machinery and the multi-view rendering stage.

- **GAIA-1** — A Generative World Model for Autonomous Driving. *arXiv'23*. [[paper]](https://arxiv.org/abs/2309.17080)
- **GAIA-2** — Controllable Multi-View Generative World Model. *arXiv'25*. [[paper]](https://arxiv.org/abs/2503.20523)
- **DriveDreamer** — Real-world-driven World Models. *ECCV'24*. [[paper]](https://arxiv.org/abs/2309.09777)
- **Vista** — Generalizable Driving World Model, High Fidelity + Controllability. *NeurIPS'24*. [[paper]](https://arxiv.org/abs/2405.17398)
- **Drive-WM** — Multi-View World Model for end-to-end planning. *CVPR'24*. — masked multi-view prediction; image-reward trajectory selection.
- **DrivingWorld** — World Model via Video GPT. *arXiv'24*. [[paper]](https://arxiv.org/abs/2412.19505)
- **GenAD / WoVoGen / MagicDrive(-V2) / Epona / InfiniCube / Cosmos-Drive-Dreams** — controllable / multi-view / long-horizon street-view generation (various venues; see the lists below).
- **Wan** — Open and Advanced Large-Scale Video Generative Models. *Alibaba '25*. — the open DiT video suite many anti-drift methods build on.

---

## Related: LiDAR / point-cloud world models

- **Copilot4D** — discrete diffusion LiDAR world model. *ICLR'24*. [[paper]](https://arxiv.org/abs/2311.01017)
- **ViDAR** — visual point-cloud forecasting for scalable AD. *CVPR'24*.
- **DynamicCity** — Large-Scale LiDAR Generation from Dynamic Scenes. *ICLR'25 (Spotlight)*.
- **LidarDM** — Generative LiDAR Simulation in a Generated World. *ICRA'25*. [[paper]](https://arxiv.org/abs/2404.02903)
- **Foundational LiDAR World Models (Latent Flow Matching)** — *arXiv'25*. [[paper]](https://arxiv.org/abs/2506.23434)

---

## Surveys & other lists

- **Awesome-World-Model** (LMD0311) — broad world-model list. [[repo]](https://github.com/LMD0311/Awesome-World-Model)
- **World-Models-Autonomous-Driving-Survey** (HaoranZhuExplorer). [[repo]](https://github.com/HaoranZhuExplorer/World-Models-Autonomous-Driving-Survey)
- **3D-Occupancy-Perception** (HuaiyuanXu) — occ *perception* survey + list. [[repo]](https://github.com/HuaiyuanXu/3D-Occupancy-Perception)
- *A Survey of World Models for Autonomous Driving*, *NeurIPS'25*; *The Role of World Models in Shaping Autonomous Driving*, *arXiv'25* [[paper]](https://arxiv.org/abs/2502.10498).

---

## Two caveats before you compare numbers

These two pitfalls invalidate a lot of head-to-head claims — bake them into any comparison.

1. **The ego confound.** Diffusion leaders consume **GT ego-trajectory**; AR methods **predict** it. Comparing a model that's told where the ego goes against one that isn't is not apples-to-apples. Always state which regime each row is in.

2. **Regime mismatch (vision-centric vs. GT-occ-token).** Gaussian world models (GaussianWorld, GEM) take **RGB** in; the GenieDrive/COME/I²-World board takes **GT occupancy tokens** in. GEM ≈ 13.6 mIoU vs. COME 34.2 is **not the same race** — never put them in one column.

3. **Aggregate mIoU is static-dominated.** Background voxels dominate the count; mover accuracy at horizon is the thing that's actually unsolved. Prefer dynamic × horizon breakdowns (Direction 6).

---

*PRs welcome. Organized by research direction rather than date — if a paper fits multiple directions, it is listed under each.*
