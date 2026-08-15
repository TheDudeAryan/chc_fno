# Autoregressive Cahn–Hilliard with a Fourier neural operator

Two files, matching the layout of the original scripts:

```
fno_train.py      training      — reads test.npz, writes fno_ch.pt
fno_predict.py    prediction    — rollout at l = 128, R², snapshots, kinetics
submit_slurm.sh   submission only
archive/          the previous multi-file pipeline and the original scripts
```

Every setting lives in a `CONFIG` block at the top of each file — no
command-line flags. `fno_predict.py` imports the solver and the model from
`fno_train.py`, so keep the two together.

```bash
python fno_train.py      # then
python fno_predict.py
```

or `sbatch submit_slurm.sh` for both.

The brief: train on `ch.py`'s dataset, then roll out autoregressively at
**l = 128** from **t = 100 to t = 2000**, scoring R² and stopping once
R² < 0.80.

---

## 1. Why the previous run was unsatisfactory

Measured on the old checkpoint (`fno_cahn_hilliard_best.pth`) through a
corrected evaluation harness, unseen seed 99999, identical protocol:

| rollout resolution | R² ≥ 0.80 survives until | steps |
|---|---|---|
| l = 64 (training resolution) | t ≈ 709 | 608 |
| **l = 128 (what the advisor asked for)** | **t ≈ 108** | **7** |

The model was not badly trained. It was being *evaluated at a resolution its
architecture cannot represent*, and that alone cost ~99 % of the rollout.

### 1a. l = 128 is a bigger box, not a finer mesh

`ch.py` hard-codes `dx = 1`, so `l = 128` is a domain of four times the area at
the same lattice spacing — not a refined discretisation of a fixed domain.
That distinction is invisible in the code and fatal for a stock FNO:

* A spectral convolution stores its multiplier `R` indexed by integer mode
  number `j`. On an `N`-point grid, mode `j` **is** the wavenumber
  `k = 2πj/(N·dx)`. Reusing weights trained at `N = 64` on `N = 128` applies
  `R` at half the physical wavenumber it was fitted for.
* Truncating at `j ≤ 24` keeps `k ≤ 2.36` at `N = 64` but only `k ≤ 1.18` at
  `N = 128` — a 2× more aggressive low-pass filter, which erases exactly the
  interfacial structure that dominates a coarsening field.

The old comment calls this *"zero-shot super-resolution"*. That property holds
when the domain is fixed and the mesh is refined; here the mesh spacing is
fixed and the domain grows, which is the opposite case.

### 1b. One-step training, 1900-step rollout

The model was fitted only on `(ψ_t → ψ_{t+1})` pairs with teacher forcing, so
it never saw its own output as an input. Every rollout step then feeds it a
slightly off-distribution field and the error compounds.

### 1c. The evaluation script could not run as written

`PhaseOrdering.chc` allocates the whole trajectory: at l = 128 and t = 2000
that is `(200100, 128, 128)` float64 = **26 GB**, so it raises `MemoryError`
before the first prediction.

### 1d. `L(t)` was measured with a divergent estimator

`L = 2π/⟨k⟩` with `⟨k⟩ = ∫kS dk / ∫S dk` omits the `k^(d-1)` radial measure,
and — more seriously — with the Porod tail `S ~ k^-(d+1)` the first-moment
integrand goes as `k^-1`, so `⟨k⟩` diverges logarithmically with the lattice
cutoff. The reported "domain size" tracked the cutoff, not the physics.

### 1e. Mass conservation was a penalty, not a constraint

Rolling the old checkpoint out without its inference-time patch,
`|⟨ψ⟩ − ⟨ψ₀⟩|` is 6.1e−04 after one step and 1.2e−02 by step 100 — 6 % of the
0.2 gap between two different physical compositions. Here the constraint is a
property of the model and the measured drift over a full rollout is 1.0e−07.

*(Attribution, to be fair to the old code: because that patch was applied, mass
drift is not what limited it — removing the patch changes l = 64 survival only
from t = 709 to t = 705. Same for its `torch.clamp(psi, -1, 1)`. Both are fixed
here because they are free to fix, not because they were the bottleneck.)*

---

## 2. What the new pipeline does differently

**Resolution-consistent spectral weights.** The multiplier is stored on a grid
of *physical* wavenumbers and resampled onto whatever lattice arrives, so
`R(k)` is always applied at the `k` it was trained at and `k_max` is held fixed
instead of shrinking. Because `irfft2` divides by `N²`, and at `dx = 1` that is
exactly the `(2π)^-d` measure of the k-integral, the real-space kernel is
independent of `N` — no rescaling needed.

*The test that pins this down.* Tile a 64×64 field 2×2 into a 128×128 one. The
tiled field is a genuine solution of the same PDE on the bigger box, so any
operator representing the physics must satisfy `F₁₂₈(tile(u)) = tile(F₆₄(u))`:

```
reference solver           : max|err| = 0.000e+00
FNO mode_scaling=index     : relative discrepancy = 7.04e-02
FNO mode_scaling=physical  : relative discrepancy = 1.07e-05   (float32 round-off)
```

Set `MODE_SCALING = "index"` to reproduce the stock-FNO behaviour for an
ablation.

**Rollout (pushforward) training.** Each batch is unrolled several steps with
the earlier ones detached, so the cost stays near one-step training while the
model is supervised on states it produced itself (Brandstetter et al.,
*Message Passing Neural PDE Solvers*, ICLR 2022). The horizon follows a staged
curriculum: `1` for the first `WARM_FRAC` of epochs, then equal-length stages
at `2, 4, …, MAX_UNROLL`. A continuous ramp was tried first and is a trap —
rounded to an integer it leaves ~19 of 60 epochs at K = 1 and reaches K = 8
only in the last three.

**Hard mass conservation.** The network predicts an *increment* and the
increment's mean is projected out, so `∫ψ` is conserved to machine precision at
every step. The penalty term is dropped because it would always be zero.

**Augmentation by exact symmetries only** — the 8 elements of D4 (the 5-point
stencil is D4-symmetric) and `ψ → -ψ` (the CH right-hand side is odd, and
`ch.py`'s off-values are symmetric about zero, so the ensemble is closed under
it). Translations are deliberately *omitted*: the model is spectral convs +
circular convs + pointwise ops, so it is exactly translation-equivariant and
translating the input teaches it nothing.

**Selection on the metric the advisor set** — how long the l = 128 rollout
holds R² ≥ 0.80, not one-step validation loss. That metric is noisy on a single
trajectory (adjacent epochs swing by hundreds of time units), so `VAL_SEEDS`
averages over three.

**Interface-weighted loss** — relative L2 plus the relative H1 seminorm. At
late times the field is flat at ±1 almost everywhere and all the information
is in the interfaces.

**Streaming reference solver** — keeps only the sampled frames, so the
l = 128 / t = 2000 ground truth costs 125 MB instead of 26 GB. Cached under
`gt_cache/`.

### 2a. The diagnostics were calibrated on ground truth first

Run on exact solver output before any model was judged by them:

| estimator | fitted exponent (t ≥ 200) | L(200) → L(2000) |
|---|---|---|
| **`L_zero`** — first zero of `C(r,t)` | **0.318** | 6.74 → 13.93 |
| `L_grad` — inverse interfacial density | 0.293 | 8.92 → 17.47 |
| `L_moment` — cutoff-limited ⟨k⟩ | 0.244 | 16.80 → 29.54 |
| `L_peak` — peak of `S(k,t)` | 0.425 | 18.52 → 41.26 |

`L_zero` recovers Lifshitz–Slyozov (1/3) to 5 %, so it is `LENGTH_KEY`.
`L_peak` is quantised (bins are `2π/N` apart; by t = 2000 the peak sits at bin
3 of 128) and `L_moment` stays cutoff-sensitive. Both are kept as cross-checks.

The Porod window is likewise narrow: bounded below by the peak (`k ~ 2π/L`) and
above by the interface width (`ξ_b = √2`, `k ~ 0.7`), leaving only
`k ∈ [0.3, 0.7]` on a 128² lattice. Measured slope there is **−2.26** against
the ideal −3; outside it a fit returns ≈ −6, which is the tanh interface
profile, not Porod. `S(k = 0, t) = 0` holds exactly (Eq. 1.184).

---

## 3. Reading `test.npz`

`ch.py` writes four float64 arrays totalling **4.90 GB**, but they are half
redundant: with `n_ahead = 100` and `gap = 10`, `y[s]` is literally `X[s + 10]`.
So `fno_train.py` reads only the `X` arrays and casts to float32 — 0.98 GB for
train, 0.24 GB for val — which is what makes the whole set fit in VRAM.

That identity is *verified*, not assumed: the label offset is detected from the
small `val` split at load time and checked against `STRIDE`. If `ch.py`'s
`n_ahead`/`gap` ever change, you get a clear error telling you which constants
to update, rather than silently wrong multi-step targets.

Storing sequences rather than `(X, y)` pairs is also what makes rollout
training possible at all — the pair format cannot express multi-step targets.

Verified against a real `ch.py`-generated `test.npz` (same call, `nens = 10`,
shortened to `end = 30` for speed): the loader detects stride 10, and an
earlier full-size check reproduced `ch.py` to `max|diff| = 3.0e-08`.

---

## 4. Configuration

Both files open with a `CONFIG` block. The training defaults target a 40 GB
A100:

```python
CHANNELS, MODES, LAYERS = 64, 24, 5
EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH = 200, 32, 1500
MAX_UNROLL, BPTT = 16, 2
AMP, TF32, GPU_DATA = "bf16", True, "auto"
VAL_SEEDS = (31337, 24601, 8675309)
```

Notes on the hardware settings:

* **`GPU_DATA = "auto"`** parks the whole training set in VRAM and samples and
  augments there — no DataLoader, no workers, no host-to-device copy per step.
  Locally this took the steady-state epoch from 11.1 s to 0.7 s. It falls back
  to host memory automatically if the set would not fit.
* **`AMP = "bf16"`, `TF32 = True`.** The complex spectral path is pinned to
  fp32: complex tensors have no autocast kernels, and a bf16 FFT would discard
  the dynamic range the spectrum needs. Mass conservation is unaffected
  (measured drift exactly 0 under autocast). There is no fp16 option — with the
  spectral path in fp32 a loss scaler buys nothing.
* **`COMPILE = True`** forces compilation for both lattice sizes up front and
  falls back to eager with the reason printed, rather than dying mid-training
  the way a naive `torch.compile` guard would.
* **`MAX_HOURS`** stops cleanly before a SLURM wall limit and writes
  `CHECKPOINT`; resubmitting continues. The LR schedule is a closed-form cosine
  (not `CosineAnnealingLR`, which updates recursively and inherits a stale rate
  on resume).

Each epoch prints ms/step, and the run prints peak VRAM at the end — enough to
retune `CHANNELS`/`BATCH_SIZE` after one short run.

**Seed hygiene.** `ch.py` draws training seeds from `[0, 10000)` and validation
seeds from `[10000, 20000)`; `VAL_SEEDS` spends 31337 / 24601 / 8675309 on
model selection. `EVAL_SEEDS = (99999, 54321, 77777)` is outside all of it, so
nothing reported has been seen during fitting *or* selection.

---

## 5. Results

60 epochs on one RTX 3050 Ti (~55 min), 9.07 M parameters, on the three unseen
test seeds at l = 128, t = 100 → 2000:

| seed | steps at R² ≥ 0.80 | first drop | mean R² | growth exponent (FNO / truth) |
|---|---|---|---|---|
| 99999 | **1900 — the full rollout** | — | 0.9216 | 0.296 / 0.318 |
| 54321 | 1147 | t = 1248 | 0.9050 | 0.310 / 0.306 |
| 77777 | **1900 — the full rollout** | — | 0.9737 | 0.282 / 0.299 |
| *old checkpoint, seed 99999* | *7* | *t = 108* | — | — |

Two of three seeds complete the entire requested rollout — a factor of ~270
against the previous checkpoint. Physics over the same rollouts:

* `L(t)` tracks the exact solver almost on top of it (exponent 0.28–0.31 vs a
  ground-truth 0.30–0.32 and Lifshitz–Slyozov's 1/3).
* `C(r,t)` collapses onto `g(r/L)` and `S(k,t)` onto `L^d f(kL)`, for the
  prediction as well as the truth.
* Free energy decays **monotonically** — required of a gradient flow, and not
  implied by any R² score.
* Maximum mass drift over 1900 steps: **9.15e−08**.

Two caveats stated plainly:

* Seed 54321 crossing at t = 1248 is not a divergence. With `EARLY_STOP =
  False` it continues to t = 2000 at mean R² 0.8334 — it decays through the
  line and hovers.
* The checkpoint was selected on a single validation rollout at the time, so
  selection luck was a live worry. Scoring the *final* epoch instead gives
  1900 / 1056 / 1900 on the same seeds — statistically the same model. (The
  three-seed `VAL_SEEDS` default now removes most of that concern.)

These numbers come from the laptop configuration (`CHANNELS=40, MODES=18,
LAYERS=4, MAX_UNROLL=8`), trained in full fp32.

### 5a. The A100 run did not improve on it

Scaling to the A100 defaults (~50 M parameters against 9 M, 200 epochs against
60, `MAX_UNROLL=16`, bf16) came out level-to-slightly-behind:

| seed | steps ≥ 0.80 (laptop → A100) | mean R² | final R² |
|---|---|---|---|
| 99999 | 1900 → 1900 | 0.9216 → 0.9216 | 0.8273 → **0.8572** |
| 54321 | 1147 → **1006** | 0.9050 → 0.9025 | 0.7999 → 0.7998 |
| 77777 | 1900 → 1900 | 0.9737 → 0.9706 | 0.9267 → **0.9030** |
| **total** | **4947 → 4806** | | |

The shape of the difference is consistent across seeds: the A100 model is
*worse at short horizons and flatter at long ones*. Error rate per 100 steps
over t ≤ 300 goes 0.0097 → 0.0133 (seed 99999), 0.0146 → 0.0209 (54321),
0.0022 → 0.0033 (77777) — uniformly worse — while past t = 1500 it is better
on two of three seeds. That is the pushforward trade-off pushed further than it
pays for itself: a longer unrolling horizon buys long-run stability at the cost
of the one-step map.

Since the accumulated error grows close to linearly in t at late times (fitted
`1 - R² ~ (t-100)^α` with α ≈ 0.9–1.3), survival time is set by **per-step
accuracy**, and degrading the one-step map costs more than the extra stability
returns.

*Precision is not the explanation.* bf16 was the other thing that changed. Its
effect at inference is negligible — rolling the same fp32-trained weights out
under bf16 autocast costs 0.0007 mean R² and zero steps — so this operator is
not precision-sensitive.

### 5b. What the training log showed

The A100 log (200 epochs, 5.7 h) identifies the cause and rules out the
alternatives:

| unroll stage | epochs | held-out one-step `val1` | mean rollout survival | best |
|---|---|---|---|---|
| K ≤ 4 | 1–120 | 0.00928 | 1389 | 1855 |
| K = 8 | 121–160 | **0.00630** | **1719** | **1952** |
| K = 16 | 161–200 | 0.00721 | 1570 | 1763 |

**The K = 16 stage made the model worse on both metrics.** The last 20 % of
training — 1.7 hours — was actively harmful, and the best checkpoint (epoch
145) came from the K = 8 stage before it.

**Overfitting is ruled out.** `val1` is held-out and fell steadily to 0.00415
by epoch 116, with no upward drift until the unroll ceiling rose. More
independent seeds is therefore *not* the lever; the 20-trajectory ensemble was
not the binding constraint.

Two structural faults, both now fixed:

* **The curriculum fought the learning-rate schedule.** Spreading stages over
  the whole run put the hardest objective last, and K = 16 began at epoch 161
  with lr = 2.0e-4, 10 % of peak — the model met its most demanding objective
  when it could no longer adapt. The curriculum now reaches `MAX_UNROLL` at
  epoch 85 with lr = 1.26e-3 and holds it for the remaining 115 epochs.
* **The horizon sampler starved the one-step map.** Sampling `k` uniformly from
  1…`k_max` gives one-step batches a share of `1/k_max`, so raising the ceiling
  silently halves the training signal for the thing that sets rollout length.
  Sampling over powers of two `{1, 2, …, k_max}` holds that share at
  `1/(log₂ k_max + 1)` — 0.25 rather than 0.125 at `k_max = 8`.

Two smaller ones: selection now uses a bounded mean R² over a *fixed* horizon
rather than the survival time, which swung by ~300 time units between adjacent
validations (best-of-40 on that is mostly a lottery); and a batch whose
detached rollout diverges is now dropped, after the log showed one epoch at
train = 1.02 against ~0.014 neighbours.

**The A100 was 10 % utilised** — 4.27 GB peak of 42.4 GB. There is room for a
much larger batch, more validation seeds, or several configurations at once.

---

## 6. A residual artefact, diagnosed

The kinetics figure shows the FNO carrying too much power at the very smallest
wavenumber. Tracking `S_FNO/S_truth` mode by mode localises it:

| rollout step | k = 0.049 | k = 0.098 | k = 0.147 | R² |
|---|---|---|---|---|
| 1   | 1.4  | 1.00 | 1.00 | 1.0000 |
| 50  | 23.9 | 0.88 | 0.91 | 0.9936 |
| 300 | 16.5 | 1.16 | 1.02 | 0.9175 |

Every other mode is accurate to ~10 %; the error is confined to `k = 2π/128`,
the fundamental of the *evaluation* box, which does not exist on the l = 64
training lattice (smallest non-zero `k` there is `2π/64 = 0.098`).

One consequence is solid: **a low-wavenumber spectral loss cannot fix this**,
because no loss evaluated at l = 64 can see `k = 0.049` at all.

The fix is `ZERO_DC = True`, which pins `R(k=0) = 0` — physically the right
value, since conserved dynamics does nothing at infinite wavelength, and the
value that makes the interpolation below the training fundamental well-posed.
Testing it needed care, because the obvious test gives the wrong answer:
zeroing those entries *post-hoc* on a trained model makes things worse
(16.7 → 29.7, R² 0.9563 → 0.9227), but that is confounded — it perturbs a
function fitted *with* those weights. A matched pair of training runs points
the other way:

| 20 epochs, matched config | k = 0.049 excess | R² @ 200 steps | survives to |
|---|---|---|---|
| plain | 18.66 | 0.9797 | t = 1261 |
| `ZERO_DC = True` | **10.20** | **0.9930** | **t = 1578** |

Better on all three axes, so it is the default. Caveat: one matched pair of
short runs at reduced size on a single test seed — suggestive, not settled.

The artefact is small in absolute terms (a box-scale modulation of amplitude
≈ 0.015 against an order parameter running from −1 to +1) and is not what ends
the rollout. It is reported because it violates the `f(p) ~ p⁴` small-`p` limit
the conservation law imposes — the kind of defect R² cannot see.

---

## 7. Extrapolation past t = 300 is *not* the bottleneck

`ch.py` generates `end = 300` while evaluation runs to `t = 2000`, so with
`L(t) ~ t^(1/3)` every state past t = 300 is extrapolation in domain size — 
training tops out near `L ≈ 7.4` lattice units, evaluation reaches `L ≈ 14`.
I previously called extending the dataset the largest remaining gain. **The
A100 results contradict that**, so the recommendation is withdrawn.

If the training horizon were binding, the error-accumulation rate would break
at t = 300. It does not. Ratio of the error rate outside the training window to
inside it, per seed:

| seed | laptop | A100 |
|---|---|---|
| 99999 | 0.89 | 0.44 |
| 54321 | 1.19 | 0.93 |
| 77777 | 1.58 | 1.34 |

Errors accumulate *no faster* past t = 300 than within it — on most seeds
slower. The fitted local exponent of `1 - R²` decreases smoothly straight
through the boundary, with no discontinuity there.

That makes physical sense: coarsening is statistically self-similar, so a
t = 1500 state looks locally like a t = 250 state with larger domains, and the
interface width is fixed at `ξ_b = √2` regardless. The operator only ever needs
local physics, which is the same in both regimes — and the resolution-consistent
spectral weights (§2) are what let it apply that physics at the right
wavenumbers on a bigger box.

So longer trajectories are not the lever. What the data points to instead is
**per-step accuracy**, and the open question is what limits it — see §8.

---

## 8. Next run and what to watch

The four fixes in §5b are in. The next A100 run is the test of them, and it
costs the same ~6 hours as the last one. Three numbers decide whether they
worked:

1. **`val1` at the end of training.** It bottomed at 0.00415 (epoch 116) and
   drifted back to ~0.0072 as the unroll ceiling rose. With the ceiling at 8
   and reached early, it should now settle at or below 0.0042 instead of
   rising. If it does not, the one-step map is at a genuine floor.
2. **`rollout R2`, the new selection score.** Bounded in [0, 1] and far
   smoother than survival, so a rising-then-flat curve means converged, while
   a still-rising one at epoch 200 means the run should simply be longer.
3. **Total steps ≥ 0.80 across the three test seeds**, against 4947 (laptop)
   and 4806 (first A100 run). Anything at or below 4947 means model scale is
   genuinely not the lever here and the honest conclusion is that a ~9 M
   parameter model already saturates this dataset.

If `val1` plateaus and test performance still does not move, the remaining
explanation is an **irreducible floor in the Δt = 1.0 one-step map** — the
operator would be at the limit of what a single 100-Euler-step jump can
represent. The levers then are a shorter model step or a multi-step input,
both departures from the brief and worth raising with the advisor rather than
deciding unilaterally.

Other open items:

* **Confirm `ZERO_DC` at full scale** (§6) — one matched pair of short runs
  favours it; repeat at the A100 configuration across more than one test seed.
  The 10 % GPU utilisation makes this nearly free to run alongside.
* **Result traceability** — checkpoints now carry a `train_config` snapshot and
  `fno_predict.py` writes it into the summary. Runs made before this change
  (including the A100 one above) cannot be traced back to their settings, and
  the summary says so explicitly.
