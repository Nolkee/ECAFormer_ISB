# AAAI-27 Abstract Registration (due 2026-07-22 19:59 Beijing)

## FINAL CANDIDATE (user title + patched abstract, 2026-07-21)

Title: Trainable Short Schrödinger Bridges for Low-Light Image Enhancement

Abstract:
Low-light image enhancement (LLIE) is typically pursued either with one-step
regression or with multi-step diffusion that transports images from pure
noise. We instead formulate paired LLIE as a short Image Schrödinger Bridge
between an illumination-lifted endpoint—produced by a jointly trained
estimator—and the normal-light image. The nearby boundaries leave only a brief
transport to learn, realized with an eight-step deterministic sampler and
evaluated without GT-mean brightness alignment. Because the lifted endpoint is
learnable, training exhibits endpoint drift and a mid-run phase transition; we
stabilize optimization with a late self-calibrating endpoint anchor that
freezes an exponential moving average of the achieved endpoint regime after
recovery. With an ECAFormer-style denoiser, the model attains competitive
fidelity on standard paired benchmarks at a fraction of the sampling cost of
noise-initialized diffusion, and we give a controlled account—against the
model's own regression backbone under matched budgets—of what the short bridge
buys and what it does not. Our evaluation audit further reveals that the
common LOL-v1-to-LOL-v2-Real transfer protocol is compromised: 99 of 100 test
images appear in LOL-v1, inflating published cross-dataset claims. Together
these provide a practical, honestly evaluated recipe for training short,
learnable-boundary bridges for LLIE.

Patch rationale: (1) "competitive ... perceptual scores" removed — LPIPS claim
is not data-backed until fair24k lands (our 0.166 vs backbone 0.088 at
mismatched protocol); replaced by the two-way "what it buys and what it does
not" pivot, true in both endings. (2) Leakage sentence restored — the most
bulletproof verified finding; steers reviewer assignment toward
evaluation-aware reviewers and anchors the analysis branch. (3) LLIE expanded
at first use.

## User's original draft (2026-07-21, kept for reference)

Title: Trainable Short Schrödinger Bridges for Low-Light Image Enhancement

Abstract:
Low-light image enhancement is typically pursued either with one-step
regression or with multi-step diffusion that transports images from pure
noise. We instead formulate paired LLIE as a short Image Schrödinger Bridge
between an illumination-lifted endpoint—produced by a jointly trained
estimator—and the normal-light image. The nearby boundaries leave only a brief
transport to learn, realized with an eight-step deterministic sampler and
evaluated without GT-mean brightness alignment. Because the lifted endpoint is
learnable, training exhibits endpoint drift and a mid-run phase transition; we
stabilize optimization with a late self-calibrating endpoint anchor that
freezes an exponential moving average of the achieved endpoint regime after
recovery. With an ECAFormer-style denoiser, the model attains competitive
fidelity and perceptual scores on standard paired benchmarks while using far
fewer sampling steps than noise-initialized diffusion. We further analyze
endpoint choices and anchoring, providing a practical recipe for training
short, learnable-boundary bridges for LLIE.

Risk note on the original: "competitive ... perceptual scores" bets on the
pending fair24k outcome; if kept, delete "and perceptual scores".

## Earlier assistant draft (superseded, kept for reference)

Title: Illumination-Lifted Image Schrödinger Bridges for Low-Light
Enhancement: Stable Training and a Controlled Account of What Generative
Transport Buys

### Alternates
1. Bridging from Illumination: Training, Stabilizing, and Honestly Evaluating
   Image Schrödinger Bridges for Low-Light Enhancement
2. Short Bridges, Learned Endpoints: A Controlled Study of Schrödinger-Bridge
   Low-Light Enhancement

## Abstract (~210 words)

Low-light image enhancement is dominated by two families: discriminative
regressors, which restore in a single step but inherit the capture conditions
of their training set, and diffusion models, which transport images from pure
noise at the cost of tens to hundreds of sampling steps and evaluation
protocols that align output brightness to the reference. We study a third
construction: an image Schrödinger bridge whose degraded boundary is not the
raw input but a coarse illumination-lifted image produced by a jointly learned
estimator, leaving a short transport that runs in eight deterministic steps
with no brightness alignment at evaluation. Making the bridge boundary
learnable exposes a training pathology we analyze in depth: the endpoint
drifts, and optimization traverses a stochastic collapse-and-recovery phase
transition exceeding 5 dB; a self-calibrating anchor that detects the recovery
turn and freezes the achieved endpoint regime makes training reliable. Against
its own regression backbone under matched budgets and protocols — a control
the bridge-restoration literature rarely runs — we give a precise account of
what the short bridge buys and what it does not, in fidelity, perceptual
quality, and cross-domain transfer. Our audit further shows the standard
LOL-v1-to-LOL-v2-Real transfer protocol is compromised: 99 of 100 test images
appear in LOL-v1, inflating published cross-dataset claims.

## Subject areas / keywords

CV: Low-level & Physics-based Vision (primary); ML: Generative Models /
Diffusion; CV: Evaluation, Benchmarks & Reproducibility. Keywords: low-light
enhancement, Schrödinger bridge, diffusion bridge, training stability,
benchmark leakage, evaluation protocol.

## Branch swap at full-paper time (7/29)

- Method branch (fair24k favorable): lead contribution = illumination-lifted
  bridge + anchor; add the matched-budget win sentence ("at matched compute
  the bridge surpasses its backbone on all three metrics; the regressor needs
  ~5x compute to overtake") + NFE/latency table vs diffusion baselines.
- Analysis branch: lead = the controlled account ("the bridge buys neither
  in-distribution fidelity nor cross-domain robustness at matched backbone")
  + dynamics + leakage; consider TMLR instead if advisor prefers not to spend
  the AAAI slot on an analysis paper.
- In both branches: cross-domain result reported as measured (LOLv2-Synthetic,
  verified disjoint); seed-basin sensitivity disclosed in limitations;
  checkpoint-reproducibility trap (EMA-scored vs bare-saved) documented in the
  protocol section.
