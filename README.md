# Awesome Occupancy World Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated, **direction-indexed** reading list for **occupancy (occ) world models** in autonomous driving.

Most existing lists are chronological dumps. This one is organized by **what problem each paper attacks**, so you can answer *"if I want to work on X, what should I read?"* directly. The taxonomy follows three battles every conditional generative field fights — **representation**, **mechanism**, and **long-horizon & measurement** — unfolded into six concrete directions plus the foundational backbone and adjacent fields.

> **Scope.** Core = generative/forecasting world models in the 3D **occupancy** space. We also track the **video** and **LiDAR/point-cloud** world-model lines, because occ borrows machinery from both.

---
## Overview Workflow

<img width="2720" height="1712" alt="occ_world_model_design_choices_with_gaussian" src="https://github.com/user-attachments/assets/432caace-9510-400b-a9de-39fc49b93160" />



---

## Contents
 
- [The leaderboard (and how to read it honestly)](#the-leaderboard-and-how-to-read-it-honestly)
- [Foundational occ world models](#foundational-occ-world-models)
  - [A. Token-AR / GPT-style](#a-token-ar--gpt-style-generative) · [B. Diffusion](#b-diffusion-based-generative) · [C. Next-scale (VAR)](#c-next-scale-var--efficient-tokenization-generative) · [D. Vision-centric + planning](#d-vision-centric-forecasting--planning) · [E. Self-supervised / pretraining](#e-self-supervised--pretraining-world-models)
- [Language-guided occupancy](#language-guided-occupancy)
  - [A. Query — open-vocabulary perception](#a-query--open-vocabulary-perception)
  - [B. VLA world models](#b-vla-world-models)
  - [C. Text-conditioned generation & editing](#c-text-conditioned-generation--editing)
  - [The open frontier](#the-open-frontier-language-as-mover-level-control)
- [Related: video driving world models](#related-video-driving-world-models)
- [Related: LiDAR / point-cloud world models](#related-lidar--point-cloud-world-models)
- [Related: language-controlled scenario generation (mechanisms to borrow)](#related-language-controlled-scenario-generation-mechanisms-to-borrow)
- [Related: generative data engine — does generated data actually help downstream?](#related-generative-data-engine--does-generated-data-actually-help-downstream)
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
- **OccTENS** — 3D Occupancy World Model via Temporal Next-Scale Prediction. *RA-L'26*. [[paper]]([https://arxiv.org/abs/2509.03887](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11358403)) — decomposes into spatial scale-by-scale + temporal scene-by-scene; pose-controllable long-term generation.
- **T³Former** — Delta-Triplane Transformers as Occupancy World Models. *arXiv'25*. [[paper]](https://arxiv.org/abs/2503.07338) — pretrained triplanes + multi-scale transformers; predicts *triplane changes* for forecasting + planning, real-time.
- **Foundational LiDAR World Models (Swin-VAE + Latent Flow Matching)** — Liu et al. *nips'25*. [[paper]](https://arxiv.org/abs/2506.23434) — high-ratio Swin-Transformer VAE + latent flow matching; strong semantic-occ forecasting

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

<img width="2920" height="1536" alt="language_guided_occupancy_three_paradigms" src="https://github.com/user-attachments/assets/26630008-dca3-4659-8a71-32e1184a7998" />

 
## Language-guided occupancy
 
"Language + occupancy" is really **three different problems**, separated by *where language enters* the pipeline. Keep them apart — conflating them is the fastest way to a confused contribution.
 
### A. Query — open-vocabulary perception
 
*Language is a **test-time probe**: CLIP-aligned voxel features let you segment / retrieve occupancy by free-form text. Single-frame; no time, no control. The mature, crowded lane.*
 
- **OVO** — Open-Vocabulary Occupancy. *arXiv'23*. [[paper]](https://arxiv.org/abs/2305.16133) — distills a 2D open-vocab segmentor into the 3D occ network; simple, model-agnostic.
- **POP-3D** — Open-Vocabulary 3D Occupancy Prediction from Images. *NeurIPS'23*. [[paper]](https://arxiv.org/abs/2401.09413) · [[project]](https://vobecant.github.io/POP3D/) — 2D-3D encoder + occ head + 3D-language head; tri-modal (image / language / LiDAR) self-supervised; camera-only at inference.
- **VEON** — Vocabulary-Enhanced Occupancy Prediction. *ECCV'24*. [[paper]](https://arxiv.org/abs/2407.12294) — assembles + adapts MiDaS + CLIP (LoRA, side-adaptor); 46M params, zero manual labels.
- **LOC** — A General Language-Guided Framework for Open-Set 3D Occupancy Prediction. *arXiv'25*. [[paper]](https://arxiv.org/abs/2510.22141) — language-guided framework prioritizing segmentation quality, not just classification.
- *Also (render-based / CLIP-pseudo-label variants of the same idea):* OccNeRF [[paper]](https://arxiv.org/abs/2312.09243), LangOcc, DistillNeRF, OccCLIP, LOcc, CAL.
### B. VLA world models
 
*Language is **one token type in a unified vocabulary** alongside occ and action; an LLM co-generates future occ + text + action. Weakly steering — language is a co-modality, not a fine-grained controller of dynamics.*
 
- **OccLLaMA** — An Occupancy-Language-Action Generative World Model. *arXiv'24*. [[paper]](https://arxiv.org/abs/2409.03272) — VQVAE occ tokens + language tokens + ego-action tokens → one vocabulary; fine-tuned LLaMA does next-token / next-scene for forecasting + planning + VQA.
- **Occ-LLM** — Enhancing Autonomous Driving with Occupancy-Based LLMs. *ICRA'25*. [[paper]](https://arxiv.org/abs/2502.06419) — couples an LLM with occupancy for forecasting + reasoning.
- **SparseOccVLA** — Bridging Occupancy and Vision-Language Models via Sparse Queries. *arXiv'26*. [[paper]](https://arxiv.org/abs/2601.06474) — sparse-query bridge for unified 4D scene understanding + planning. ⚠️ recent neighbor — differentiate against it.
### C. Text-conditioned generation & editing
 
*Language is a **generation condition**: text → occ scene generation or semantic editing. Mostly scene appearance / layout ("rainy, dense"), not mover behavior over time.*
 
- **OccScene** — Semantic Occupancy-based Cross-task Mutual Learning for 3D Scene Generation. *arXiv'24 (TPAMI under review)*. [[paper]](https://arxiv.org/abs/2412.11183) — the only **pure text → occ** generator (joint occ + RGB diffusion, perception↔generation mutual learning, no GT layout at inference). Eval on SemanticKITTI / NuScenes-Occupancy / NYUv2; 3D-scene-gen FID 39.21 (vs SemCity 56.55) + a downstream perception mIoU boost. Controls scene *appearance*, not rare layouts.
- **X-Scene** — Large-Scale Driving Scene Generation with Multi-Granular Control. *NeurIPS'25*. [[paper]](https://arxiv.org/abs/2506.13558) · [[project]](https://x-scene.github.io/) — strongest **text → (RAG + GPT-4o → BEV layout) → triplane-VAE occ → image / video → 3DGS** cascade (layout = additive cond, box + text = cross-attn). Eval on Occ3D-nuScenes: occ-gen FID3D 258.8 (vs UniScene 529.6), Real+Gen downstream 3DOD mAP 39.9 (best synthetic augmenter). ⚠️ **the closest competitor for text→occ** — but long-tail only *claimed* (no corner-case experiment), downstream only *aggregate* (no rare slice), editing only qualitative, no per-agent behavior / collision constraint, closed-vocabulary + nuScenes-bounded RAG.
- **UniScene** — Unified Occupancy-centric Driving Scene Generation. *CVPR'25*. [[paper]](https://arxiv.org/abs/2412.05435) — occ-centric hierarchy; conditionable generation across occ / video / LiDAR. Control is **BEV layout**, not text (text only in its video stage).
- **HorizonWeaver** — Generalizable Multi-Level Semantic Editing for Driving Scenes. *arXiv'26*. [[paper]](https://arxiv.org/abs/2604.04887) — text-guided multi-level semantic scene editing.
- **Scaling Up Occupancy-centric Driving Scene Generation** — Dataset and Method. *arXiv'25*. [[paper]](https://arxiv.org/abs/2510.22973) — dataset + method for occ-centric controllable scene generation.
- **VLA-World** — Learning Vision-Language-Action World Models for Autonomous Driving. *arXiv'26*. [[paper]](https://arxiv.org/abs/2604.09059) — trajectory / direction-conditioned generation of the near-future scene. ⚠️ recent neighbor.

> **Evaluation protocol for text→occ** (to sit next to X-Scene / UniScene / DynamicCity / OccSora in one table): dataset = **Occ3D-nuScenes** (200×200×16). Report (1) VAE recon **IoU + mIoU**; (2) occ-gen **FID3D / F3D / KID / Precision–Recall** (DynamicCity protocol, both 11- & 17-class); (3) occ-gen **mIoU** (UniScene protocol; number to beat ≈ **20.5**); (4) optional downstream occ-pred / 3DOD / UniAD-planning. ⚠️ No shared head-to-head benchmark exists — every paper reports a different dataset×metric mix, so a clean protocol is itself a contribution.

### The open frontier: language as mover-level control
 
All three lanes above leave the **same cell empty**: natural language that steers the future *behavior of individual movers*, with the occupancy world model rolling out a physically consistent future — *"the left car cuts in"*, *"the pedestrian crosses"*. Open because:
 
- **A** never touches dynamics (query only).
- **B** treats language as a co-generated token, not a per-agent controller.
- **C** conditions scene *appearance*, not per-mover *behavior over time*.
This is the language interface to the **per-mover-control** problem — and it is distinct from the trajectory / instruction level of SparseOccVLA and VLA-World, which steer the *ego*, not each agent.

**A second empty cell — validated long-tail usefulness.** Even the works that *do* condition occ generation (UniScene, X-Scene) validate only on **aggregate** downstream metrics; none proves a gain on a **held-out rare slice**. A *prior-driven, language-controlled occupancy generator that targets safety-critical scenes and proves a downstream gain on a rare slice* does not exist — see [Related: generative data engine](#related-generative-data-engine--does-generated-data-actually-help-downstream).
 
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
 
## Related: language-controlled scenario generation (mechanisms to borrow)

*How language is actually wired into generation — at the trajectory / BEV / video level. Occ has adopted almost none of these; this is the menu to port to occupancy. Four mechanisms:*

**A. LLM emits a structured intermediate, a learned model renders it** *(dominant & the most portable to occ: text → layout/boxes → frozen occ generator; decouples semantics from the generative prior).*
- **LCTGen** — Language-Conditioned Traffic Generation. *CoRL'23*. [[paper]](https://arxiv.org/abs/2307.07947) — text → GPT-4 → integer scene-vector → transformer generator.
- **InteractTraj** — Language-Driven Interactive Traffic Trajectory Generation. *NeurIPS'24*. [[paper]](https://arxiv.org/abs/2405.15388) — text → interaction-code (follow / yield / cut-in) → trajectories.
- **DriveDreamer-2** — LLM-Enhanced World Models for Diverse Driving Video. *AAAI'25*. [[paper]](https://arxiv.org/abs/2403.06845) — text → LLM trajectories → HDMap → multi-view video; rare events + a **measured** downstream det gain.
- **Text2Street** — Controllable Text-to-Image for Street Views. *ICPR'24*. [[paper]](https://arxiv.org/abs/2402.04504) — text → road-topology + bbox layout → image.

**B. LLM emits a guidance loss as code; test-time-guide a diffusion sampler** *(no paired text data; naturally adversarial / long-tail).*
- **CTG++** — Language-Guided Traffic Simulation via Scene-Level Diffusion. *CoRL'23*. [[paper]](https://arxiv.org/abs/2306.06344) — GPT-4 writes a differentiable loss → guides trajectory diffusion.
- **LD-Scene** — LLM-Guided Diffusion for Adversarial Safety-Critical Scenarios. *arXiv'25*. [[paper]](https://arxiv.org/abs/2505.11247) — CTG++ + code-debugger + unit-test hardening.

**C. Direct conditioning (cross-attn / prompt embeddings)** *(cleanest end-to-end; needs paired text↔scene data).*
- **ProSim** — Promptable Closed-loop Traffic Simulation. *NeurIPS'24*. [[paper]](https://arxiv.org/abs/2409.05863) — per-agent multimodal/text prompts; closed-loop; ships **520k** text↔scenario pairs (the most portable paired asset).
- **SimGen** — Simulator-conditioned Driving Scene Generation. *NeurIPS'24*. [[paper]](https://arxiv.org/abs/2406.09386) — text + simulator layout → cascade diffusion image.

**D. LLM-as-agent driving a tool / edit pipeline** *(flexible editing / report-grounded; heavyweight; bounded by underlying assets).*
- **ChatSim** — Editable Scene Simulation via Collaborative LLM-Agents. *CVPR'24*. [[paper]](https://arxiv.org/abs/2402.05746) — role-specialized LLM agents place / edit assets in a NeRF.
- **CrashAgent** — Crash Scenario Generation via Multi-modal Reasoning. *arXiv'25*. [[paper]](https://arxiv.org/abs/2505.18341) — crash reports + diagrams → simulation-ready scenarios.
- **Seeking to Collide** — Online Safety-Critical Generation w/ Retrieval-Augmented LLMs. *ITSC'25*. [[paper]](https://arxiv.org/abs/2505.00972) — LLM intent inference + retrieval bank → adversarial agents.
- **AGENTS-LLM** — Agentic LLM Augmentation of Challenging Scenarios. *arXiv'25*. [[paper]](https://arxiv.org/abs/2507.13729) — NL-instructed edits that stress SOTA planners.

---

## Related: generative data engine — does generated data actually help downstream?

*The honest scoreboard for "world model as a long-tail data source." Separate **closes the loop** (trains a downstream model on generated data, reports real-task gains) from **realism-only** (FVD/FID, or a pretrained detector run on replayed frames). Almost all positive results are **in-distribution camera-video augmentation** (+1–4 mAP/NDS); long-tail is usually **claimed, rarely measured on a held-out rare slice**.*

**Closes the loop (real downstream numbers):**
- **Delphi** — Controllable Long Video Generation for E2E AD. *arXiv'24*. [[paper]](https://arxiv.org/abs/2406.01349) — **VLM failure-mining**: GPT-4 attributes UniAD failures → generate similar data; **972 samples (4%) → collision 0.34→0.27 (~25% rel.)**. Closest to a true long-tail engine.
- **SubjectDrive** — Scaling Generative Data via Subject Control. *ECCV'24*. [[paper]](https://arxiv.org/abs/2403.19438) — StreamPETR **+3.6 mAP / +3.3 NDS**; key finding: *diversity, not volume, scales*.
- **MagicDrive** — Street View Generation with 3D Geometry Control. *ICLR'24*. [[paper]](https://arxiv.org/abs/2310.02601) — BEVFusion **+2.52 mAP / +1.95 NDS**.
- **Panacea / Panacea+** — Panoramic & Controllable Video. *CVPR'24*. [[paper]](https://arxiv.org/abs/2311.16813) — StreamPETR **+2.6 mAP / +2.3 NDS** (+AMOTA / lane in Panacea+).
- **DrivingDiffusion** — Layout-Guided Multi-View Video. *ECCV'24*. [[paper]](https://arxiv.org/abs/2310.07771) — det **NDS 0.412 → 0.434**.
- **WoVoGen** — World Volume-aware Multi-camera Generation. *ECCV'24*. [[paper]](https://arxiv.org/abs/2312.02934) — BEVDet **+1.3 mAP** (narrow, cars/trucks/buses).
- **UniScene** — *CVPR'25*. [[paper]](https://arxiv.org/abs/2412.05435) — the **only occ generator that closes the loop**: occ **+8.5 mIoU**, det **+3.13 mAP / +2.61 NDS**. But occ is GT/layout-derived; long-tail only qualitative.

**Skeptic's anchor — cite it, beat it:**
- **Dream4Drive** — Rethinking the Driving World Model as a Synthetic Data Generator. *arXiv'25 (ICLR'26)*. [[paper]](https://arxiv.org/abs/2510.19195) — the augmentation benefit can become **negligible once you simply train longer on real data**; aggregate +mAP may be a training-budget artifact, not new information.

> **The verified gap (the thesis of this list's frontier).** A **prior-driven, language-controlled OCCUPANCY** generator that **targets safety-critical / long-tail** scenes and **proves a downstream gain on a held-out RARE slice** does not exist in the verified literature. Three rules for claiming it honestly: (1) report **rare-class / held-out-scenario** downstream metrics, not aggregate mAP/NDS; (2) include the **train-longer-on-real control** (the Dream4Drive test); (3) avoid the **X-Drive trap** — mixing real + synthetic modalities hides the synthetic contribution.

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
3. **Aggregate mIoU is static-dominated.** Background voxels dominate the count; mover accuracy at horizon is the thing that's actually unsolved. Prefer dynamic × horizon breakdowns.
---
 
*PRs welcome. Organized by research direction rather than date — if a paper fits multiple directions, it is listed under each.*
