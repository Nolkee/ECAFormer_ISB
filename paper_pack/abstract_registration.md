# AAAI-27 Abstract Registration (due 2026-07-22 19:59 Beijing)

Two-way wording: every claim below is already data-backed and survives BOTH
endings (method resurrection via fair24k, or analysis framing). Direction-
specific claims are deferred to the full paper (7/29), where the abstract may
be revised; the title is branch-agnostic and should not change.

## Title (recommended)

Illumination-Lifted Image Schrödinger Bridges for Low-Light Enhancement:
Stable Training and a Controlled Account of What Generative Transport Buys

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
