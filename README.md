# FNO for the Cahn–Hilliard equation — trained on l = 64 to END ≤ 1000, autoregressive rollout to t = 10000 on l = 128–512

Fourier-neural-operator surrogate for the dimensionless Cahn–Hilliard equation (the Cahn–Hilliard–Cook equation without its noise term; no solver here adds noise)

$$\partial_t\psi = \nabla^2\left(\psi^3 - \psi - \nabla^2\psi\right)$$

(Puri, *Kinetics of Phase Transitions*, Eq. 1.182 — what `ch.py` integrates).
This is one of four architectures in the group, all on the same protocol.

Protocol, per the project brief: train on the dataset `ch.py` produces (64×64
boxes, data up to t = END), then predict **unseen** trajectories on a
**128×128** lattice, starting at **t = 100** and rolling autoregressively in
real time, scored by **R²** plus snapshot comparison. The brief's horizon is
t = 2000. The group's shared protocol (run by `ftp.sh`) goes five times further,
to **t = 10000**, and also uses larger boxes at the
same lattice spacing. Since 2026-11-09 the runs use l = 128, 256 and 512, each
at all five compositions (progress §4t). The models are trained to END = 300, 500 or 1000, with
the Laplacian-output head or the plain head. The rollout runs the full
horizon; the **0.80** threshold is reported (when R² first crosses it) and is
not used to stop the rollout (`EARLY_STOP = False` in `fno_predict.py`).
Since 2026-10-22 a model that takes a step of any length reaches t = 10000 in
about 256 variable steps instead of 9,900 steps of dt = 1 (progress §4g).
Since 2026-10-23 the default model also takes the chemical potential of its
input as a third input channel (progress §4k).

## Results in short

Pooled end-of-rollout R² at t = 10000, all-seeds mean, of the default model (`train_t<END>_lap`):
- END = 1000: 0.9310 on the common 6 cases, 0.9314 on all 15 cases (750 seeds).
- END = 500: 0.9151 on the common 6 cases, 0.9169 on all 15 cases (750 seeds).
- Model size at END = 500, 24 modes, one training seed each: 0.889 to 0.933 on all 15 cases, with no trend in width (progress §2.4).

Details: progress.md §2.

## 1. Running

On the cluster (account `m25ph1005`, project folder `chc_fno`; partition
`dgx_fat`, one A100-SXM4-40GB; `module load python; source env/bin/activate`),
from the project root. The jobs are submitted with four scripts,
each taking its settings as `--export` variables, with no arguments.
Every `ft.sh`, `fp.sh` and `ftp.sh` line passes
`FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache` (the reference cache on
scratch; training and prediction must use the same).

| script | time limit | CPUs, memory | does | variables (default) |
|---|---|---|---|---|
| `ch_gpu.sh` | 30 min | 4, 16 GB | the dataset `Work/data/test_t<END>.npz` (ch.py's format), once per END | `END` (500; 1000 until 2026-11-11) |
| `ft.sh` | 2 h | 4, 20 GB | train one model (dt-conditioned; validated to t = 10000) | `END` (500; 1000 until 2026-11-11), `VARIANT` (lap), `CHANNELS` (38), `MODES` (24), `SEED` (0), `FNO_GT_CACHE` |
| `fp.sh` | 7 h 15 min | 4, 10 GB | predict and draw a trained model, then refresh the overview of `Results/` (`make_submission.py --overview`) | the same, to pick the model, and `T_END` (10000) |
| `ftp.sh` | 8 h | 4, 10 GB | train, then predict and refresh the overview, in one job; predicts only after a training that finished | as `fp.sh` |

After every job the script's `#SBATCH` lines are set to what the job used plus
a margin (standing rule, `CLAUDE.md`; `sacct`). `hpc_storage.sh` asks for 1 GB and 15 min.
**The CPU count of a GPU job cannot go below 4:** `dgx_fat` refuses a GPU
job with fewer ("Job violates accounting/QOS policy (job submit limit, user's
size and/or time limits)", found with `sbatch --test-only` on 2026-10-22), so
every GPU script keeps 4, although the jobs use about 1 to 1.3.
A training job needs 20 GB because of the dataset load: it reads
`train_X` of `test_t1000.npz` as float64 (6.6 GB) and converts it to float32
(3.3 GB), so the peak is near 12 GB. sacct's MaxRSS, sampled about every 30 s,
can miss such a short peak, so it is not a safe basis for a memory request.

**Submit from a login shell.** A job submitted from a shell that is not a
login shell (for example a tool shell) does not get lmod's `module` function
and dies at `module load python` (`module: command not found`). From a normal
terminal, or as `bash -lc 'sbatch ...'`, the scripts work unchanged.

The variables give the label: `VARIANT=base` drops `_lap`, a size other than
38 channels and 24 modes adds `c<C>m<M>`, and a seed other than 0 adds
`s<N>` (`fno_train.py --print-label` prints it, which is how `fp.sh` finds
the model). Logs are written next to the scripts (`ft_<job>.out`,
`fp_<job>.out`, `ftp_<job>.out`, `ch_gpu_<job>.out`).

The run grid below has one line for each run folder that exists in
`Results/`. The 8- and 12-mode sizes of END = 1000 were trained
(`Work/models/`) but have no results folder; their lines are in progress §4n.

```bash
# once per copy of the project: the folder layout (dry run first; no jobs queued).
python migrate_layout.py
python migrate_layout.py --apply

# storage, then the datasets: Work/data/test_t<END:04d>.npz
sbatch hpc_storage.sh --du
sbatch --export=ALL,END=1000 ch_gpu.sh   # a bare `sbatch ch_gpu.sh` makes END = 500
sbatch --export=ALL,END=300 ch_gpu.sh
sbatch --export=ALL,END=500 ch_gpu.sh

# the references (about 860 GB in 800 files for the 15 cases) are on scratch, for training and prediction alike:
# every ft / fp / ftp line passes the same FNO_GT_CACHE
# the group protocol at END = 1000, Laplacian head on and off, to t = 10000
sbatch --export=ALL,END=1000,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=1000,VARIANT=base,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
# END = 300 and 500 (Laplacian head on and off), to t = 10000
sbatch --export=ALL,END=300,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=300,VARIANT=base,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,VARIANT=base,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh

# the size sweep at END 1000, 24 modes: channels 16 / 24 / 32 / 45 / 64, one job per size;
# 38 / 24 is the default (above).  A job stopped by its limit (exit 75) is resubmitted with the same line.
sbatch --export=ALL,END=1000,CHANNELS=16,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=1000,CHANNELS=24,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=1000,CHANNELS=32,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=1000,CHANNELS=45,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=1000,CHANNELS=64,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
# the training-seed replicate of the default
sbatch --export=ALL,END=1000,SEED=1,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh

# the five 24-mode sizes at END = 500 (END = 500 is the default of the scripts)
sbatch --export=ALL,END=500,CHANNELS=16,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,CHANNELS=24,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,CHANNELS=32,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,CHANNELS=45,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh
sbatch --export=ALL,END=500,CHANNELS=64,MODES=24,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ftp.sh

# train and predict as separate jobs: submit fp.sh once the ft.sh job has ended
sbatch --export=ALL,END=500,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ft.sh
sbatch --export=ALL,END=500,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache fp.sh

# the no-hard-constraints ablation (slides 10-12): the soft mass penalty 0.1, and none
sbatch --time=03:00:00 --export=ALL,END=500,VARIANT=base,NOHARD=1,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ft.sh
sbatch --time=03:00:00 --export=ALL,END=500,VARIANT=base,NOHARD=1,MASS_WEIGHT=0,TAG=free,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ft.sh

# the model without the μ input channel (slide 29); --no-mu-channel needs a tag
sbatch --time=03:00:00 --export=ALL,END=500,NOMU=1,TAG=nomu,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache ft.sh
```

**Speed timing (one job, one GPU).** With `FNO_TIMING` set, `fp.sh` does not
predict.  It times the solver and the networks on one device, in one process,
and writes the table.  For each size l = 128, 256, 512 it solves 50 runs
(composition 0, the first 50 group seeds) for t = 0..200 with
`fno_gt.ground_truth_batch`, and times `fno_train.step` for each network
(3 warm-up calls, 20 timed, the batch a real rollout uses).  It scales to a
whole run: solver to the 990,000 steps of t = 100..10000, network to 320
calls.  It also solves t = 0..10000 at 128 once, to check the short-span
estimate.  `FNO_TIMING=1` times the `--model` checkpoint; otherwise give model
labels or `.pt` paths separated by `:` (not commas: `sbatch --export` splits
at them).  The result is `timing.txt` and `timing.json` in
`Work/analysis/timing_<date>/` (`FNO_TIMING_OUT` names another folder; an
existing folder is never overwritten, `_2`, `_3` are added).  Nothing else is
written; the solver frames go to a temporary folder in it, removed at the end
(up to 16 GB at the check).  Put `FNO_TIMING_OUT` on /scratch, beside the
reference cache, so the frames are written where a real cache fill writes
them.  It takes about 15-30 min of GPU time.

```bash
sbatch -w gpu1 --export=ALL,FNO_TIMING=train_t0500_lap:train_t0500_lap_c16m24,FNO_TIMING_OUT=/scratch/data/m25ph1005/timing_2026-11-12 fp.sh
```

**Mean drift (one job, one GPU).** With `FNO_DRIFT` set, `fp.sh` does not
predict either.  For each model it names it rolls out the critical quench
(ψ̄₀ = 0, l = 128) on the group protocol's 50 test seeds, all on one
schedule (the ψ̄₀ = 0 pick in `train_t0500_lap`'s rollout-cache `meta.json`,
else alpha 0.1 and dt_max 30), and records the mean ⟨ψ⟩ of the network and of
the solver at every step.  Models are labels or `.pt` paths separated by `:`;
`+nomean` after one switches off its final mean removal (refused for a model
that has none).  The result is `drift.txt` and `drift.json` in
`Work/analysis/drift_<date>/` (`FNO_DRIFT_OUT` names another folder; an
existing one is never overwritten): per model, the deviation
⟨ψ_pred⟩ − ⟨ψ_true⟩ of each seed at each time, its seed mean, the largest per
seed and the number of seeds that diverged.  Slides 10 to 12 use it.

```bash
sbatch --export=ALL,FNO_DRIFT=train_t0500_lap:train_t0500_lap+nomean:train_t0500_nohard:train_t0500_nohard_free,FNO_DRIFT_OUT=/scratch/data/m25ph1005/drift_2026-11-12,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache fp.sh
```

**Ablation (one job per model, one GPU).** With `FNO_ABLATE` set, `fp.sh`
does not predict either.  For each model it names it rolls out the full test
grid (15 cases, the 50 group test seeds, 750 runs) and scores R² as a
prediction does, on the schedules of the default's real run (each
composition's pick from `train_t0500_lap`'s rollout-cache `meta.json`, the
same for every model).  `+nofill` after a model switches its low-k fill off,
`+noclamp` its μ clamp (each refused where it does not apply).  The result is
`ablate.txt` and `ablate.json` in `Work/analysis/ablate_<date>/`
(`FNO_ABLATE_OUT` names another folder; never an existing one), rewritten
after every case: per model and case the all-seeds mean R² at t = 2000, 5000
and 10000 with the survivors-only mean and the diverged seeds, pooled over the
15 cases, the common 6 and each box size.  Nothing goes into the results tree
or the rollout cache.  Slide 29 uses it.

```bash
sbatch --export=ALL,FNO_ABLATE=train_t0500_lap+nofill,FNO_ABLATE_OUT=/scratch/data/m25ph1005/ablate_nofill,FNO_GT_CACHE=/scratch/data/m25ph1005/fno_gt_cache fp.sh
```

Environment on the cluster: `env/` works after `module load python` (Python
3.11.7, torch 2.13.0+cu126). pyright 1.1.414 is pip-installed in `env/`; its
bundled node needs libatomic, so it is run as
`LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/12.2.0/lib64:$LD_LIBRARY_PATH pyright <files>`.
Storage: `/iitjhome` has a 200 GB hard per-user quota (GPFS; read it with
`/usr/lpp/mmfs/bin/mmlsquota --block-size auto`, since `quota` and `df` do not
show it); `/scratch` has 1 TB per user, the user's area being
`/scratch/data/m25ph1005`.

A job stopped by the time limit is simply resubmitted with the same command.
A training stopped by the time limit exits with code 75, so `ftp.sh` ends
there and never predicts with a half-trained model. A
training that has already finished (`EPOCHS`; patience for `--fixed-step`)
trains nothing more, and a prediction reuses what it has written: the
references it solved, the schedules it tuned (in `meta.json`) and the per-seed
records of the cases it finished. Do not queue two jobs for the same label at
once: they would share one model folder.

Anything the scripts do not cover is a flag of the Python programs, run by
hand on a GPU node (for example `--tag NAME`, `--keep-folder`, `--fixed-step`,
`--seed-set fno`):

Training flags (`python fno_train.py --help`):

| flag | meaning |
|---|---|
| `--t-train T` | train on t = 0..T; a lean dataset is sliced to it (`ft.sh` and `ftp.sh` pass it) |
| `--dataset PATH` | `test.npz` from `ch.py`, or a lean `.npy` from `ch_gpu.py --lean` |
| `--variant lap\|base` | Laplacian head (default) or plain head |
| `--seed N` | training seed (default 0); another seed adds `s<N>` to the label |
| `--channels C`, `--modes M` | model width (default 38) and Fourier modes (default 24); another size adds `c<C>m<M>` |
| `--keep-folder` | leave the size token out of the folder's name, to continue a model of another size in the folder it has |
| `--fixed-step` | train the old kind of model (one step of 1.0 per call, the old loop, loss and values) instead of the variable-step default |
| `--ema` | validate and save a running average of the weights (decay 0.999); on by default for the variable-step model |
| `--select score\|survival` | what picks the best model: the mean rollout score (default for `--fixed-step`), or the time R² stays above 0.8 with accuracy as tie-break (default for the variable-step model) |
| `--tag NAME` | a name for the folder: `train_t<T>_NAME` |
| `--epochs N` | default 80 |
| `--no-resume` | start fresh even if a resume state exists; the old files are kept as `fno_ch_prev.pt` and `fno_ch_last_stale.pt` |
| `--accept-schedule` | resume a state made with another training schedule anyway; the differences are printed and recorded |
| `--allow-cpu`, or `FNO_ALLOW_CPU=1` | run on the CPU inside a Slurm job that has no GPU (refused otherwise) |
| `--print-label` | print the train label these flags give and exit (needs `--t-train`; `fp.sh` and `ftp.sh` use it to find the model) |
| `--smoke` | a tiny end-to-end run that writes only under `Work/smoke` |

Also `--val-t-start`, `--val-t-end` (the validation window; for the
variable-step model it is t = 100 to 10000 and must be a whole number of 4.0
reference spacings, and the scripts do not pass it) and `--monitor-t-end` (the
horizon of the fixed-step model's one-step monitor; the variable-step model
has `val1` instead).
`--ema`, `--select` and `MU_CHANNEL` do not change the folder's name: a run
trained another way in the same folder is refused (a mu run in a folder holding
a best model without the channel too), and `--tag NAME` gives it a folder of its
own.

Prediction flags (`python fno_predict.py --help`):

| flag | meaning |
|---|---|
| `--model PATH` | checkpoint to evaluate; results are named for its folder |
| `--views H,H,...` | one rollout to the longest, results for each horizon (`Results/<train label>_pred_t<H:05d>`) |
| `--start-time T`, `--end-time T` | rollout window (default 100 to 2000) |
| `--seeds N` | test seeds per case (default 50) |
| `--seed-set fno\|group` | `fno`: this project's seeds (default for a fixed-step model); `group`: the group protocol's 50 test seeds (default for a dt-conditioned model), to score the models of the group on the same trajectories |
| `--sizes L,L,...` | box sizes (default 128,256,512) |
| `--eval-mode all\|critical\|off_critical\|both\|none` | which compositions to run (default all) |
| `--no-supersede` | stop, instead of moving the old results to `Archive/superseded`, when the folder holds another checkpoint's results |
| `--no-diagnostics` | skip the tile-invariance and low-wavenumber checks |
| `--no-records` | keep no permanent per-seed records and no `meta.json` |
| `--gt-cache-dir DIR` | where the references are: for a dt-conditioned model the 4.0-spaced ones (default `FNO_GT_CACHE`, else `Work/gt_cache`) |
| `--gt-batch N` | seed engine: missing reference solutions solved N at a time (default 16; about 1 GB a seed at l = 256) |
| `--redo` | roll out again cases an earlier job finished |
| `--engine seed\|lockstep`, `--batch B` | fixed-step models only: the seed engine up to t = 2000, the lock-step engine beyond. A dt-conditioned model always takes the schedule engine, and `--engine schedule` is refused for a fixed-step one |
| `--no-tune` | schedule engine: alpha 0.2, dt_max 40 for every quench instead of tuning on the validation seeds |
| `--video` | lock-step engine: make the film (off there by default; the schedule engine has none) |
| `--no-video`, `--no-submission`, `--no-gt-cache` | skip films, skip the submission figures, solve references without storing them |
| `--smoke`, `--allow-cpu` | as for training |

`fp.sh` passes only `--model` and `--views`, which for a dt-conditioned model
means the schedule engine and the group protocol's 50 seeds. To use this project's own seeds
instead, run
`python fno_predict.py --model Work/models/train_t1000_lap/fno_ch.pt --views 10000 --seed-set fno`
(on a GPU); tune seed 90210 is also one of those test seeds and is then left
out of the tuning, with a warning.

Redrawing and checking, none of which needs a GPU:

```bash
python make_submission.py --records Work/rollout_cache/<run label> --out Results/<run label>
python make_submission.py --records Work/rollout_cache/<run label> --out Results/<run label> --partial
python make_submission.py                  # redraw every complete run from its records, then the overview
python make_submission.py --run <run label>   # one run, then the overview
python make_submission.py --overview       # only Results/README.md + comparison images (--index is the old name)
python make_diagram.py                     # Results/fno_architecture.png/.pdf, project_pipeline.png/.pdf
python tests/run_all.py                    # all 32 suites, 4 at a time, then pyright (about 13 min)
python tests/run_all.py --changed fno_gt.py make_submission.py   # only the suites that use these files; pyright on them
python tests/run_all.py --changed fno_gt.py --dry-run           # which suites that would run, and why
python tests/run_all.py --failed           # rerun what failed or crashed last time
python tests/run_all.py --sections         # where the slowest suites spend their time
python tests/run_all.py health             # only the suites whose name contains "health"
python tests/run_all.py --no-lint          # skip the pyright pass
```

`--partial` draws the complete cases of a run that is not finished. The
redraw has no film. `--records` also rebuilds the `extras/` folder beside the
records in the rollout cache (the records and `meta.json` are not changed). To
redraw without touching the cache, give it a scratch copy: `meta.json` and
`extras/` copied, `records/` linked with `cp -as`.


The test runner prints one line per suite; the log of each is
`tests/_scratch/logs/<suite>.log`, and `tests/_scratch/last_run.json` keeps the
last result of each suite. `tests/_paths.py` sets `FNO_FAST_FIGURES=1`, which saves figures at half the DPI with light PNG compression (`fno_paths.save_figure`); a real run does not set it and writes the usual files. `python fno_train.py --smoke`,
`python fno_predict.py --smoke` and `python make_submission.py --smoke` are the
quick end-to-end check; they write only under `Work/smoke`.

Copying results back to the laptop: the run folder, and its
`Work/rollout_cache/<run label>/` (the records, `meta.json` and `extras/`),
then `python make_submission.py --overview`.
## 2. Files, labels and the results tree

| file | role |
|---|---|
| `ch.py`, `ch_gpu.py` | the advisor's dataset generator (now `end = 500`, see §6) and its batched GPU version. `ch_gpu.py --lean FILE --end N` writes one float32 file of X only that serves any training horizon up to N |
| `fno_train.py` | **training**: CH solver, the FNO (dt-conditioned by default), the variable-step training method (chains of steps of 1 to 50 time units; with `--fixed-step`, the unroll curriculum of §4), divergence guards and rollback, validation to t = 10000 and checkpoint selection. Every setting is a constant at the top of the file. `--t-train T [--variant lap or base] [--seed N] [--channels C] [--modes M] [--keep-folder] [--fixed-step] [--ema] [--select score or survival] [--tag NAME] [--dataset PATH] [--epochs N] [--no-resume] [--accept-schedule] [--print-label] [--allow-cpu] [--smoke]`. The default `lap` variant predicts M and applies the 5-point Laplacian (§3.1). `set_precision_flags()` (the fp32 flags, shared with `fno_predict.py`) and the switchable low-k weight fill (`LOWK_FILL = True`; config key `lowk_fill`, `set_lowk_fill(model, on)`) and the chemical-potential input channel (`MU_CHANNEL = True`; config key `mu_channel`, absent = off; `FIXED_STEP` sets it False; part of the best-model check; `CahnHilliardFNO(mu_channel=...)` appends μ = ψ³ − ψ − ∇²ψ after the mean channel) |
| `fno_gt.py` | the reference cache (files named like `gt_l128_off+0.0_seed31337_t100-10000_ev4.npz`), the batched GPU solver that fills it, `make_schedule` (the times of a variable-step rollout) and the ten validation seeds |
| `fno_predict.py` | **prediction**: reference prewarm, rollouts (the schedule engine for a dt-conditioned model; for a fixed-step one the seed engine up to t = 2000 and the lock-step engine beyond), R², snapshots, kinetics, conservation, diagnostics, per-seed records, `meta.json`, then the drawing of the run. `--model PATH [--views H,H,..] [--start-time T] [--end-time T] [--seeds N] [--seed-set fno or group] [--sizes L,L,..] [--eval-mode ..] [--no-tune] [--video] [--no-records] [--no-supersede] [--no-diagnostics] [--gt-batch N] [--redo] [--smoke] [--allow-cpu]`. It runs in the training's precision (`setup_precision()`: fp32, TF32 off), has the constant `LOWK_FILL` (None: the checkpoint's own setting, the default; a checkpoint without the key being off; True or False: override) and stamps its numerics in `meta.json`, the manifest and `report.txt` |
| `make_submission.py` | **drawing**: the run folders from caches or from the permanent records, with no model. No argument: every complete run in `Results/` is redrawn from its records, then the overview. `--overview` (old name `--index`): only `Results/README.md`, `comparison.png` and `comparison_t<END>.png`. `--run LABEL`: one run. `--records DIR --out DIR [--partial]`: one run into a named folder. `--cache`, `--all` and `--model` are older options; `--smoke` rebuilds the newest smoke run |
| `fno_paths.py` | torch-free: the roots and the label grammar (`train_label`, `run_label`, `parse_label`, `compose_tag`) and the reference-cache folder (`ref_cache_dir`: `FNO_GT_CACHE`, else `Work/gt_cache`) |
| `fno_runlog.py` | tees each process's output into `Work/logs/` with a header naming host, python, torch, GPU, SLURM job |
| `migrate_layout.py` | moves the loose outputs of the old layout into `Results/`, `Work/`, `Archive/`. **Dry run by default**; `--apply` moves, never overwrites, never deletes, writes `Work/logs/migrate_layout_<time>.json` with how to undo each move. Run it with no jobs queued |
| `ch_gpu.sh` | Slurm job: the dataset `Work/data/test_t<END>.npz`; `sbatch ch_gpu.sh` (END = 500 since 2026-11-11) or `sbatch --export=ALL,END=1000 ch_gpu.sh` |
| `ft.sh` | Slurm job, 2 h, 20 GB: **train**. `sbatch --export=ALL,END=500[,VARIANT=base][,CHANNELS=C,MODES=M][,SEED=N][,FNO_GT_CACHE=DIR] ft.sh` |
| `fp.sh` | Slurm job, 7 h 15 min (4 CPUs, 10 GB; a first prediction that has to solve the references resumes on resubmission): **predict and draw**, the same variables plus `T_END` (10000); ends with `make_submission.py --overview` |
| `ftp.sh` | Slurm job, 12 h, 20 GB (right-sizing pending): **train, then predict**, the same variables; predicts only after a training that finished; ends with `make_submission.py --overview` |
| `hpc_storage.py`, `hpc_storage.sh` | used and available storage and quota on the cluster. It only reads. `sbatch hpc_storage.sh --du` adds the size of `Work`, `Results`, `Archive` |
| `make_diagram.py` | draws `fno_architecture` and `project_pipeline` (`.png` and `.pdf`) into the root of `Results/`, numbers read from the code; needs no GPU. The older `Results/diagrams/` folder is left as it was |
| `tests/run_all.py` | the 32 suites, four at a time, each with its own scratch folder and log (`tests/_scratch/logs/<suite>.log`), then pyright over the shipped files; `--changed FILE ...` runs only the suites that use those files, `--failed` reruns what failed . `FNO_TEST_MODEL` names a checkpoint for the suites that need a real one; the fixed-step suites build a small model of their own (`tests/_paths.py`) |

Every setting lives in a block of constants at the top of each Python file, so
a run is reproducible from those files, the flags and the checkpoint's
`train_config`. `fno_predict.py` imports the model and the solver from
`fno_train.py` and the reference cache from `fno_gt.py`, so keep the three
together.

### Runs, labels and where things live

A run is a model trained on data up to t = T and rolled out from a start time
to an end time. The **train label** is

    train_t<T:04d>[_lap][_nohard][_<tag>]

and the **run label** adds the rollout window:

    <train label>_pred_t<H:05d>              from t = 100 (the default start) to H
    <train label>_pred_t<S:05d>-<H:05d>      from another start S to H

The `_lap` token is the Laplacian-output head, `_nohard` the
no-hard-constraints ablation (`HARD_CONSTRAINTS = False` at the top of
`fno_train.py`; it takes the plain head by itself). The tag slot holds one
token, built in a fixed order

    c<C>m<M>[-s<N>][-<tag>]

where `c<C>m<M>` (channels and modes, both numbers) appears only when either
differs from the file's `CHANNELS` / `MODES`, `s<N>` only when the seed is not
0, and `<tag>` is the user's own `--tag`. A `--tag` that looks like a size or
seed token is refused. The size token is compared with the file's values, not
with fixed 45 and 24.

| label | what it is |
|---|---|
| `train_t0300` | trained to t = 300, plain head. The existing September model (64 channels) has this label |
| `train_t1000_lap` | trained to t = 1000, Laplacian output, default size and seed |
| `train_t1000_lap_s1` | the same with `--seed 1` |
| `train_t1000_lap_c24m8` | `--channels 24 --modes 8` |
| `train_t1000_lap_c24m8-s1-mine` | both, and `--tag mine` |
| `train_t1000_lap_pred_t10000` | that model rolled out from t = 100 to 10000 |
| `train_t1000_pred_t00200-04000` | the plain-head model, from t = 200 to 4000 |

The group's protocol uses `_pred_t10000`. `fno_predict.py --views` can write
results for several horizons (2000, 10000, 20000) from one rollout to the
longest.

```
Results/                        report material only; make_submission.py regenerates it
  README.md                     index of every run (generated; never edit by hand)
  comparison.png                critical-quench mean R²(t) of the runs, log t axis
  comparisons/l<size>/          comparison_<case>.png: comparison.png for each other case (per size since 2026-11-09)
  comparison_t<END>.png         the Laplacian runs of one END: a panel per mode count, pooled R² (two or more, one tagged)
  end500_vs_end1000.png         R²(10000) against l, a panel per composition, the END = 500 and 1000 Laplacian runs
  fno_architecture.png/.pdf     python make_diagram.py
  project_pipeline.png/.pdf       (the older diagrams/ folder is left as it was)
  <run label>/                  one folder per box size (since 2026-11-09):
    l128/, l256/, l512/         for each composition <case> (psi0m0.40 ... psi0p0.40): snapshots_<case>.png,
                                phase_ordering_kinetics_<case>.png, domain_growth_<case>.png,
                                psi_distribution_<case>.png, composition_<case>.png, free_energy_<case>.png;
                                composition_all.png, r2_time.png, figure_notes.txt
                                For the run: composition.csv, manifest.json, report.txt, training_curves.png,
                                figure_notes.txt (training_curves.png only; no sweep PDF since 2026-11-11),
                                r2_sweep_per_size/l<size>.png, r2_sweep_per_case/<composition>.png
                                (one image per box size / composition; each folder has its own figure_notes.txt)
Work/                           heavy and machine-side, never part of the report
  data/                         test_t1000.npz, test_t0300.npz, test_t0500.npz (ch_gpu.sh); older: test.npz, test_numpy.npz, ch_l64_T2000_f32.npy
  gt_cache/                     reference solutions, shared by every run: the 4.0-spaced files of
                                the variable-step runs (FNO_GT_CACHE names another
                                folder), and the older 1.0-spaced files of the fixed-step engines
  models/<train label>/         fno_ch.pt (best), fno_ch_last.pt (resume state),
                                fno_ch_history.json, fno_ch_config.json; with --no-resume
                                the old files stay as fno_ch_prev.pt / fno_ch_last_stale.pt
  rollout_cache/<run label>/    meta.json (with each quench's tuned schedule) and
                                records/<case>/seed<N>.npz, kept for good; the temporary per-case
                                cache is deleted as each case is written
    extras/                     what only the FNO writes, never in Results/: diagnostics.json, the
                                bookkeeping manifest, sweep_curves.npz, the timing report, and a
                                folder per case (psi0+0.00_l128, ...: summary json, mean_r2.csv,
                                CSVs, best_seed<N>/, conservation figure, the film when made)
  logs/                         one log per process (fno_runlog.py), migrate_layout_*.json
  slurm/                        Slurm output; must exist before sbatch
  smoke/                        everything a --smoke run writes
Archive/                        superseded runs and old code, moved there and never deleted:
                                superseded/<label>_<fingerprint>_<time>/ (a run another
                                checkpoint replaced), superseded/<label>_<fingerprint>-<old
                                numerics>_<time>/ (the same checkpoint, other numerics),
                                code_<date>_before_<change>/, Old/, duplicates/
```

**Superseded runs.** If a prediction would write into a results folder that
holds results of another checkpoint, the old results and records (and the
extras) are moved to `Archive/superseded/` first. `--no-supersede` stops the
run instead. Since 2026-10-22 the same happens for results or records of the
same checkpoint made with other numerics (the stamp `{"precision", "tf32",
"lowk_fill"}` in `meta.json`; a missing stamp counts as fp32, TF32 on, fill
off): they go to `Archive/superseded/<label>_<fingerprint>-<old numerics>_<time>/`,
the run is computed again and the schedules are tuned again. With the same
numerics a job resumes as before.
## 3. Model options

### 3.1 Laplacian output

`LAPLACIAN_OUTPUT = True` is the default. The head predicts a field M, and the
increment is its 5-point Laplacian, the stencil the solver uses. Mass is then
conserved exactly for any time step, and the lowest wavenumbers are suppressed
by k². `--variant base` gives the plain head (the increment's mean projected
out); the label gets `_lap` for the Laplacian head and nothing for the plain
one.

* A checkpoint with no `laplacian_output` key loads as the plain head, so the
  September models still load and predict. An old plain-head *training state*
  refuses to resume under the new default, and the refusal names
  `--variant base`.
* The no-hard-constraints ablation (`--no-hard-constraints`, ft.sh `NOHARD=1`)
  has no zero DC and no final mean removal, and takes the plain head by
  itself; its folder is `train_t<END>_nohard[_<tag>]`, and `--variant lap`
  with it is refused.  Its loss keeps a soft mass penalty, `MASS_WEIGHT = 0.1`
  times the squared drift of the mean.  `--mass-weight W` (ft.sh
  `MASS_WEIGHT=W`, with `TAG=` for a folder of its own) changes it and is
  refused without the ablation; `--mass-weight 0` gives a free output
  (`train_t0500_nohard_free`).
* The idea is not new (Fantasia et al. 2025; Lanzoni et al. 2024). The group
  protocol trains both heads at each END.

Measured: progress §4n.

### 3.2 Low-k weight fill

**What it fixes.** A spectral layer stores its weights by mode number on the
64-point training lattice: kx = −24 … 24 and ky = 0 … 24 for 24 modes, at the
physical wavenumbers k = 2πj/64. On a box of l points at the same spacing the
layer keeps m = round(24 · l / 64) modes (48 at l = 128, 96 at l = 256), at
k = 2πj/l, and resamples the weight grid bilinearly onto them, so that every
physical wavenumber gets the weight trained at that wavenumber
(`SpectralConv2d._resampled_weight`). With `ZERO_DC` the weight of the zero
wavenumber is forced to 0. The new lowest modes of a larger box lie between
that zero and the first trained mode, so the interpolation pulls them towards
0: at l = 128 the mode at k = 2π/128 gets half the trained weight, at l = 256
the modes j = 1, 2, 3 get a quarter, a half and three quarters of it. As the domains grow, more and
more of the field's power sits at these lowest wavenumbers. Without the fill the critical quench and the
larger boxes coarsened too slowly and the low-wavenumber gain was far from 1.

**What it does.** Before the resampling, the masked zero-wavenumber entry is
filled with the mean of its three trained neighbours (kx = ±1 at ky = 0, and
ky = 1 at kx = 0); after the resampling that entry is set to 0 again
(`SpectralConv2d._filled_weight`). The zero wavenumber stays at 0, and the new
low modes start from trained values instead of from 0. On the 64-point
training lattice nothing is resampled, so the training steps are unchanged; the
fill acts in the validation rollouts (critical quench, l = 128) that choose
the checkpoint and in every prediction on l = 128 to 256.

**Settings.** `fno_train.LOWK_FILL = True`, the training default. Each checkpoint stores `lowk_fill` in its config; one made before
the key existed has it off. `fno_predict.LOWK_FILL = None` follows the
checkpoint; True or False overrides it for one prediction. `set_lowk_fill`
changes it on a built model. The report's numerics stamp says "low-k fill on".

Measured: progress §4h and §4i.

### 3.3 μ input channel

**What it is.** The network gets a third input channel next to ψ and its mean:
the chemical potential of its own input, μ = ψ³ − ψ − ∇²ψ, computed with the
solver's 5-point periodic Laplacian (Δx = 1). The Cahn–Hilliard equation is
∂ψ/∂t = ∇²μ, and the Laplacian head already makes each step the 5-point
Laplacian of the field M that the network predicts (scaled with the step length
dt), so the exact step for a short dt is close to ∇²(dt · μ). With μ as an input
the network sees directly where the bulk of a domain is at equilibrium (μ flat,
so the true field there hardly moves). It adds one input channel to the lift: 38
parameters at 38 channels (17,799,303 → 17,799,341).

**Why it was added.** The model with the fill alone held the critical quench and
l = 256 but did poorly off-critical. Prediction-only checks of that checkpoint
(progress §4j) found that the model moved the bulk of the domains far more than
the truth late in a rollout, that the one-step error grew outside the training
window (most off-critical), and that smaller time steps did not help.

**Settings.** `fno_train.MU_CHANNEL = True`, the training default. Each checkpoint stores `mu_channel`; one made before the key
existed has no channel. `--fixed-step` turns it off, and so does `--no-mu-channel` (ft.sh `NOMU=1`;
it needs `--tag`, e.g. `train_t0500_lap_nomu` for slide 29). The model folder's name
does not change with it, so a training into a folder whose model has the other
setting is refused.

Measured: progress §4k.

### 3.4 μ clamp

**What it fixes.** The μ channel is cubic in ψ. In the model's own rollout a
single pixel can spike where a small droplet dissolves (|ψ| about 2 to 3 after
one step); the channel feeds that back as a much larger input, and the rollout
runs away past |ψ| = 5, which counts as diverged. This showed up in the
training-seed-1 model (§3.6): trajectories in the ψ₀ = 0 family diverged
suddenly, at t = 1248 to 5848. The seed-0 model had none.

**What it does.** The channel is computed from ψ clamped to [−1.2, 1.2]:
μ = m³ − m − ∇²m with m = clamp(ψ, −1.2, 1.2). Only the channel's input is
clamped; ψ itself, the residual update and the head are not. A true field stays
below |ψ| ≈ 1.05 (training inputs with their noise about 1.06), so the clamp
never acts on training data and does not change the training; in a rollout it
bounds the input of a spike, and the spike heals.

**Settings.** `fno_train.MU_CLIP = 1.2`, the training default; each checkpoint stores `mu_clip` (none in one made before the key existed).
`fno_predict.MU_CLIP = 1.2` sets it on the loaded model for a prediction, so the
two older default checkpoints get it too; None follows the checkpoint.
`set_mu_clip` changes it on a built model. It is in the report's numerics stamp
("mu clip 1.2"). A model without a μ channel ignores it.

Measured: progress §4m.

### 3.5 Model size

Defaults in `fno_train.py`: `CHANNELS = 38`, `MODES = 24`, `LAYERS = 5`,
`LOCAL_KERNEL = 3`, `MU_CHANNEL = True`: **17,799,341** parameters for the
default model (dt-conditioned, with the mu channel; 17,799,303
without the channel; 17,766,217 for the fixed-step model, `--fixed-step`), with `EPOCHS = 80`. The
mu channel is a third input (ψ, its mean, μ) and adds 38 parameters to the lift
at 38 channels. The September model had 64 channels
(50,394,241). Old 64-channel checkpoints keep their own configuration.

`--channels C` and `--modes M` change the size; `--modes` is refused above
`(lattice − 1) // 2` = 31 for the 64-point training boxes. Parameter counts of
the sweep grid for the dt-conditioned model (the head does not change the count; the dt-conditioning, a small MLP
and one FiLM layer per block, adds 33,086 at 38 channels over the fixed-step
model; the mu channel adds 38 at 38 channels, which does not change the values in M at two decimals):

| channels | 8 modes | 12 modes | 24 modes |
|---|---|---|---|
| 16 | 0.42M | 0.86M | 3.17M |
| 24 | 0.94M | 1.93M | 7.11M |
| 32 | 1.65M | 3.41M | 12.63M |
| **38** | 2.32M | 4.80M | **17.80M (default)** |
| 45 | 3.24M | 6.73M | 24.95M |
| 64 | 6.54M | 13.58M | 50.44M |

A model of another size from the file's goes into its own folder
(`train_t1000_lap_c24m8`). To go on training an existing model of another size
in the folder it already has, give its own flags and **`--keep-folder`**, which
leaves the size out of the folder's name: the old 64-channel state in
`train_t1000_lap` is continued with `--channels 64 --keep-folder`. The resume
refusal prints these exact flags, from the saved values. Without
`--keep-folder`, `--channels 64` on the 38-channel file means a new model in
`train_t1000_lap_c64m24`. Nothing is moved or deleted by this.

Measured: progress §4n (END = 1000) and §4t (END = 500).

### 3.6 Training seeds

**Two kinds of seed.** The *test seeds* are the initial conditions of the 50
test trajectories of each case. They are the same for every run, so every model
is scored on exactly the same problems. The *training seed* (`SEED`,
`--seed N`, label suffix `s<N>`, seed 0 adds nothing) sets everything random in
a training: the initial network weights, the training sampler's draws (which
trajectory, which frame and which step length dt go into each batch) and the
augmentation draws (the 8 square symmetries and ψ → −ψ). The data, the code
and the settings are the same; only the luck of the training changes. The
training config (`fno_ch_config.json`) records seed and size, the checkpoint
carries them in `train_config`, and the prediction's manifest has `train_setup`
(seed, channels, modes, head), which a redraw keeps. A checkpoint from before the
seed was recorded has seed `null`.

**Why more than one training seed.** One training run can be lucky or unlucky.
A difference between two methods or two model sizes, each trained once, can be
training noise. Repeating the same training with another seed measures that
noise, and it can show a failure that one run happens not to have.

**Seed 0** (`train_t1000_lap`) is the default run: every number quoted for the
default model is from it, and every END and size run is seed 0 too.
**Seed 1** (`train_t1000_lap_s1`) is the replicate with the same training and
seed 1. Before the clamp, 26 of its 400 trajectories diverged at the ψ₀ = 0
family, where seed 0 had none; this led to the μ clamp (§3.4).
**Seed 2** (`train_t1000_lap_s2`) has not been run. Two runs give one
difference, not a spread; a third is the smallest set that gives a spread with
some meaning.

Measured: progress §4l and §4m.

## 4. Method

The default model takes a step of any length: it is *dt-conditioned*, it
trains on chains of steps of random length, and it rolls out to t = 10000 in
about 256 variable steps instead of 9,900 steps of dt = 1. The formulas are in
§7; this section says what each part is and where it lives.

**Why it is built this way.** The spectral weights are stored on a grid of
physical wavenumbers and resampled onto whatever lattice arrives, so the
model trained at l = 64 applies each weight at the wavenumber it was trained at
on l = 128 to 512; a tile test (a 64×64 field tiled 2×2 must give the tiled
output) pins this down, and `fno_predict.py` repeats it, with a
low-wavenumber gain check, for every checkpoint it loads. Mass is conserved
exactly (the Laplacian head, §3.1); the only augmentations are exact
symmetries (the 8 elements of D4 and ψ → −ψ), since translations teach a
translation-equivariant model nothing.

Training is on chains of the model's own outputs, with the gradient through
every step, so that it sees states it produced itself. The checkpoint is
chosen on the metric the advisor set, how long the l = 128 rollout holds
R² ≥ 0.80, averaged over ten validation seeds because the metric is noisy on one
trajectory. The loss is the R² loss plus an increment and a gradient term, so
that the interfaces, where all the information is at late times, count.

* **Model (`fno_train.py`).** `forward(psi, dt)` takes the step length in time
  units. log dt is embedded (sine and cosine at 32 frequencies from 1 to 64,
  plus log dt) and passed through a small MLP; every block turns the result into
  a per-channel scale and shift (FiLM, zero at the start), and the projection is
  multiplied by a learned gain that starts at dt before the Laplacian head.
  Mass conservation, the zero DC mode, the residual form and the Laplacian
  output are as in §3. The third input channel is the chemical potential (§3.3).
  With 38 channels, 24 modes and 5 layers it has **17,799,341** parameters (§3.5);
  a fixed-step checkpoint still loads and gives the output it gave before
  (`tests/t_vdt.py`).
* **Training.** Each sample has its own step dt, drawn
  log-uniformly from 1 to 50 time units on a 0.1 grid (a chain has to fit in
  the trajectory); a chain starts at a random time from t = 35 on and uses its
  dt at every step. A chain is 1 step long at first, 2 steps from 8 % of the
  epochs and 4 steps from 19 %, with the gradient through every step; the
  batch is 128, 64 and 32. Each step's loss is the R² loss + 0.25 × an
  increment loss + 0.25 × a gradient loss. AdamW (betas 0.9 and 0.95, weight
  decay 1e-4), gradient clip 1.0, learning rate a cosine from 3e-3 to 1e-5,
  times 0.65 for each stage. An average of the weights (EMA, decay 0.999, lower
  at first) is what is validated and saved. D4 symmetries and ψ → −ψ,
  noise of amplitude 0.015 × U(0, 1) on each batch's chain input, and
  all 25 trajectories of the dataset train (`TRAIN_ON_VAL`). **80 epochs of
  200 steps**, no patience. The smoothness and amplification penalties and the
  batch spike guard of the fixed-step loop are not used (the epoch-level
  divergence rollback stays).
* **Validation and selection.** Every 2 epochs and at the last: 10 seeds (the
  group protocol's validation seeds 31337, 24601, 8675309, 20250, 42424, 90210,
  23571, 55555, 48151, 62831), critical quench, l = 128, rolled out from
  t = 100 to **t = 10000**, whatever END is, on a schedule (stride 0.2 t, at
  most 40, rounded to a multiple of 4) against the shared references (below). A
  seed that goes non-finite or beyond |ψ| > 5 is frozen and scored R² = −1 from
  there. The best model is the one with the longest *survival* (mean
  over the seeds of the first time R² < 0.80, interpolated, 10000 if never);
  an epoch within 1.0 time unit of the best takes over only with a higher
  accuracy up to twice END. A one-step check, `val1`, uses held-out reference
  pairs at steps of 4, 12 and 40. `--val-t-end` is not passed by the scripts:
  the validation window is always to t = 10000, beyond every END.
* **`--fixed-step` keeps the old kind.** It trains the model that takes one
  step of 1.0 per call, with the old loop (an unroll curriculum) and loss and
  the old values (the `FIXED_STEP` block at the top of `fno_train.py`: batch 32,
  1500 steps per epoch, no EMA, 5 validation seeds to t = 2000 every 5 epochs,
  patience 8, selection by score). **Labels do not change** (they carry no step
  token), so a model folder holds one kind. A run of the other kind in the
  folder is refused: with a resume state (`fno_ch_last.pt`) the resume check
  stops it ("belongs to another run", printing both sets of run settings, which
  differ in the validation window and in `dt_conditioned`); with only
  `fno_ch.pt` the best-model check stops it, names `dt_conditioned`, and says to
  move the old folder to `Archive/` or to give the new run a folder of its own
  with `--tag NAME`. `--no-resume` skips the resume check, but the best-model
  check then stops the run wherever `fno_ch.pt` is in the folder.
* **Prediction (`fno_predict.py`).** A dt-conditioned checkpoint is recognised
  from its config and takes the *schedule engine*: every seed starts from its
  reference at t = 100 and takes about **256 variable steps** to t = 10000
  (stride 0.2 t, at most 40, rounded to a multiple of the reference spacing 4).
  `(alpha, dt_max)` is tuned **per quench** on the
  10 validation seeds (alpha 0.1, 0.2, 0.35, 0.5 or 0.7 × dt_max 30, 40, 45 or
  50: 20 candidates, scored by the mean R² at T/4, T/2, 3T/4 and T); the pick
  is stored in `meta.json` and reused when a job is resubmitted, and the box
  sizes take the critical quench's pick (`--no-tune`: alpha 0.2, dt_max 40).
  Tune seeds that are also test seeds are dropped with a warning. **The test
  seeds are the group protocol's 50 by default** (`--seed-set group`;
  `--seed-set fno` is still there), so that the models of the group are scored
  on the same trajectories. The schedule
  visits every snapshot time (100, 500, 1000, 2000, 4000, 5000, 6000, 8000,
  10000, and every 2000 beyond for a longer end) and every view horizon. A seed
  that diverges (|ψ| > 5) is scored R² = −1 to the end of the schedule and
  stays in the means; once no seed is alive the model is not called again.
  The two checks of the loaded model use a step: the tile test dt = 10, the
  low-wavenumber gain dt = 48. There is no film. **Fixed-step
  checkpoints keep the seed and lock-step engines**; the lock-step engine
  films only with `--video` and stops when every seed has stopped or diverged.
  The prediction runs in the training's precision (fp32 with TF32 off) and
  takes the low-k weight fill from the checkpoint (§3.2) and the μ clamp (§3.4);
  results made with other numerics are moved to `Archive/superseded/`.
  A prediction job sets the precision, moves a results folder of another
  checkpoint (or the same one with other numerics) away, makes the two model
  checks once, and **prewarms the references**: the 50 test seeds of every
  case not yet finished and the 10 tune seeds of each quench still to be tuned
  are found in the cache or solved. Then, case by case, it takes the quench's
  `(alpha, dt_max)` from `meta.json` or tunes it, rolls the seeds out in chunks
  (10 seeds at l = 128) from t = 100 along the schedule, and writes the case.
  At the end come the run-level files and `make_submission.py --overview`.
* **Reference cache (`fno_gt.py`).** The files are named like
  `gt_l128_off+0.0_seed31337_t100-10000_ev4.npz`, one file per lattice,
  quench and seed, with the exact solution (float64 Euler, as `ch.py`) saved as
  float32 every 4.0 from t = 100 to 10000 (2,476 frames, 162 MB at l = 128),
  uncompressed so that it can be memory-mapped. The folder is `Work/gt_cache/`,
  or the environment variable `FNO_GT_CACHE` (the 15-case set is about 860 GB,
  so a cluster run points it at scratch). Training validation and prediction
  both read it and solve what is missing, together on the GPU (a CPU solve
  inside a Slurm job is refused). **The 15-case set is about 860 GB in 800
  files** (15 cases × 50 test seeds: 40.6, 162.3 and 649.1 GB at l = 128,
  256 and 512; plus the 10 tune seeds of each of the 5 quenches at l = 128,
  8.1 GB; measured, progress §4v), and **the first prediction solves the
  references it needs; that dominates the job** *(about 8.5 h of solver time
  on gpu1 for the 750 test trajectories, an estimate from the one-job timing,
  progress §4w; in the 8-case pilot it was 112.08e9 bytes in 450 files and
  about 54 min of the 75.1 min prediction)*; a job that runs out of time there
  is resubmitted and continues from the references already in the cache. The
  older 1.0-spaced float64 files of the fixed-step engines have other names
  and are left alone. Before it solves, `fno_predict` prints the references
  still missing (`[gt] need X GB for N files; Y GB free in <dir>`) and
  refuses when the disk cannot hold them with 2 GB to spare
  (`CACHE_MARGIN`). In a Slurm job without `FNO_GT_CACHE` (and without
  `--gt-cache-dir`) it warns that `/iitjhome` has a 200 GB quota that `df`
  does not show, and refuses above 100 GB (`HOME_CACHE_LIMIT`).
* **Results layout.** `Results/<run label>/` has one subfolder per box size
  (`l128/`, `l256/`, `l512/`) with the figures of each composition, the
  `r2_sweep_per_size/` and `r2_sweep_per_case/` folders, `report.txt`,
  `composition.csv`, `manifest.json` and `training_curves.png`; the tree is in
  §2. What only the FNO has (per-case summaries and CSVs, the best seed's
  folder, the conservation figure, `diagnostics.json`, `sweep_curves.npz`, the
  bookkeeping manifest, a film when one is made) is kept in
  `Work/rollout_cache/<run label>/extras/`. `make_submission.py`: no argument
  redraws every complete run from its records and then the overview;
  `--overview` (`--index` is the old name) only `README.md`, the comparison
  images and every run's `training_curves.png`; `--run LABEL` one run;
  `--records DIR --out DIR [--partial]` one run into a named folder.
* **Job scripts.** `ft.sh`, `fp.sh` and `ftp.sh` are as in §1. `fp.sh` and
  `ftp.sh` end with `python make_submission.py --overview`. `ftp.sh` predicts
  only after a training that finished (a training stopped by its time limit
  exits with code 75, which ends the job).
* **Speed.** The old lock-step engine took about 160 ms per step (65 ms of
  solver, 95 ms of model, l = 128, 50 seeds) for 9,900 steps: about 26 min per
  50-seed case in the step loop. On the laptop the schedule engine was about
  15 times faster than that, an order of magnitude on different hardware.
  **Measured on the cluster (the pilot, A100-SXM4-40GB, 2026-10-22):** training
  30.8 min for 80 epochs (95 ms per step at unroll 4); prediction 75.1 min for 8
  cases of 50 seeds, of which about 54 min was solving 440 references and 250.7 s
  was the 115,050 model steps; speedup over the solver 12.9 times for those
  trajectories (the solver time of the uncached trajectories over the FNO time).
* **Tests.** `FNO_TEST_MODEL` names the checkpoint that a suite needing a real
  one reads, then `Work/models/train_t0500/fno_ch.pt`; once that folder is
  moved to `Archive/`, point `FNO_TEST_MODEL` at a copy.

## 5. What to watch in a training run

The first two checks catch failure early rather than at the end. The
variable-step training prints `d/psi`, `DIVERGED` lines and `val1`, and on
validation epochs `rollout`, `in-range R2` and `survives t=` with the survival
of each seed.

1. **`d/psi` must never trend to zero.** Zero is the identity collapse: the
   model returns its input. If it starts falling, kill the job — the rest of
   the run is worthless. Its value is measured on training chains of random
   step length; the "low 0.0Xs" of the fixed step of 1.0 has not been measured
   for it.
2. **`DIVERGED` lines.** One or two rollbacks are the guard doing its job.
   Repeated ones mean the learning rate is still too high for the horizon;
   lower `LR` or `LR_STAGE_DECAY` rather than letting it burn recoveries.
3. **`val1` at the end.** The one-step error at steps of 4, 12 and 40; a
   plateau means the one-step map is at a floor.
4. **`survives t=`.** The survival of each validation seed (the first time its
   R² falls below 0.80) is what picks the checkpoint; watch that it does not
   fall from one validation epoch to the next.

**Result traceability.** Checkpoints carry a `train_config` snapshot and
`fno_predict.py` writes it into the summary. Runs made before that change
cannot be traced back to their settings, and the summary says so explicitly.

## 6. Setup notes

* **Test seeds are genuinely unseen.** `ch.py` draws training seeds from
  [0, 10000) and validation seeds from [10000, 20000); `VAL_SEEDS` and
  `MONITOR_SEEDS` (selection and the one-step monitor) are fixed values
  outside the evaluation set; evaluation seeds come from
  `make_eval_seeds`: the original ten first, the rest from [100000, 1000000),
  which collides with none of them (`tests/t_seeds.py` asserts the four
  disjointnesses, per progress §3.5). `--seed-set group` uses the group
  protocol's 50 test seeds, whose first three are 99999, 54321 and 77777. The
  variable-step model's ten validation seeds are the group protocol's too
  (31337, 24601, 8675309, 20250, 42424, 90210, 23571, 55555, 48151, 62831);
  none is among the 50 test seeds.
* **Reading `test.npz`.** `ch.py` writes four float64 arrays totalling
  **4.90 GB**, but they are half redundant: with `n_ahead = 100` and
  `gap = 10`, `y[s]` is literally `X[s + 10]`. So `fno_train.py` reads only
  the `X` arrays and casts to float32, which is what makes the whole set fit in
  VRAM. That identity is *verified*, not assumed: the label offset is detected
  from the small `val` split at load time and checked against `STRIDE`; if
  `ch.py`'s `n_ahead`/`gap` ever change, you get a clear error naming the
  constants to update. Storing sequences rather than `(X, y)` pairs is also
  what makes chain training possible. The lean dataset
  (`ch_gpu.py --lean`) is the same trajectories, seeds and frames as `X` in one
  float32 file (8.2 GB to t = 2000); a longer file is a bitwise prefix of a
  shorter one, so one file serves END = 300, 500, 1000 and 2000
  (progress §4a).
* **The dataset follows the group rule.** Same dataset and parameters as
  `ch.py`; the generator is `ch_gpu.py` (the GPU gives the same frames,
  progress §3.4). `END` was raised from 300 to 500 on 2026-09-09 with the
  advisor's agreement, and the 2990 frames up to t = 300 are bit-identical; the
  original is kept as `Archive/ch_original_end300.py` (progress §6).
  END = 1000 is part of the group's shared protocol (§1).
* **Precision.** The code runs in fp32 with `AMP = "off"`, `COMPILE = False`,
  because TF32 *training* proved destructive (progress §3.7); the prediction
  runs in the same precision (`fno_predict.setup_precision()`), which changed the
  R² by about 0.001 at most. The complex spectral path is fp32 whenever
  autocast is on. `GPU_DATA = "auto"` keeps the training set in VRAM, sampled
  and augmented there, and falls back to host memory if the set would not fit.
  `MAX_HOURS = "auto"` reads the deadline from Slurm, stops cleanly and writes
  the checkpoint; the learning rate is a closed-form cosine, so a resume does
  not inherit a stale rate. Each epoch prints ms/step, and the run prints peak
  VRAM at the end.
* **Seed hygiene for selection.** `VAL_SEEDS` are spent on model selection, so
  nothing reported has been seen during fitting or selection. The prediction
  also tunes each quench's schedule on the ten validation seeds and leaves
  out any that is also a test seed (`--seed-set fno` has one, 90210).
* **Loading checkpoints.** Four loaders read whole checkpoint files with
  `torch.load(..., weights_only=False)`: `fno_train.load_checkpoint`,
  `fno_train.check_best_model`, the training's resume and
  `fno_predict.checkpoint_conditioned`. Unpickling a file can run code, so
  load only checkpoints this project wrote.
* **Python version.** The group asked for Python 3.12; every cluster job ran
  3.11.7 (the header of each log in `Work/logs/` says so). `pyrightconfig.json`
  sets `pythonVersion` 3.11, and the laptop runs 3.14.6, so both ends differ
  from 3.12. The dataset does not depend on the version: `ch_gpu.py` reproduces
  `ch.py`'s frames.

## 7. Methods and formulas (reference)

This section writes down every formula and rule behind the numbers and the figures of this README and of `Results/`. Each one is what the code does, and the names in `code` are the constants and functions that hold it. Where the code differs from the textbook form, the code's form is given. progress.md gives the history and the measurements; this section gives the definitions. Every setting lives in a block of constants at the top of its Python file (§2); the main values are collected in 7.11.

**Notation.** $l$ is the number of lattice sites per side, $\Delta x=1$, $\Delta t=0.01$ is the solver step, $t$ is real time, $\psi$ is the order parameter, $\langle f\rangle$ is the mean of $f$ over the $l\times l$ lattice, $\bar\psi_0=\langle\psi\rangle$ is the composition (it does not change in time) and $\psi'=\psi-\langle\psi\rangle$. "Reference" and "truth" are the exact solver; "FNO" is the model rollout. $s$ indexes seeds ($N=50$ per case), $t_i$ the times of a rollout ($t_0=100$), $\lVert\cdot\rVert^2$ a sum of squares over the lattice.

### 7.1 Equation, reference solver and initial condition

**Equation.** $\partial_t\psi=\nabla^2\mu$ with $\mu=\psi^3-\psi-\nabla^2\psi$. The code integrates this noise-free form. There is no thermal-noise (Cook) term in the three solvers of the project (`ch_gpu.py`, `fno_gt.py`, `fno_train.py`); the only randomness is the initial condition.

**Laplacian.** The 5-point stencil on the periodic lattice (`torch.roll` / `np.roll`, `DX = 1`):

$$\nabla^2_5 f_{i,j}=\frac{f_{i+1,j}+f_{i-1,j}+f_{i,j+1}+f_{i,j-1}-4f_{i,j}}{\Delta x^{2}}$$

**Integrator.** Explicit Euler with `DT = SOLVER_DT = 0.01`, in float64. One step applies the stencil twice:

$$\mu^{n}=(\psi^{n})^{3}-\psi^{n}-\nabla^2_5\psi^{n},\qquad \psi^{n+1}=\psi^{n}+\Delta t\,\nabla^2_5\mu^{n}$$

On a GPU, `_integrate` keeps the frame buffer on the host and copies one frame when it is saved, so GPU memory does not grow with the number of frames. A frame "at time $t$" is the state after $\mathrm{round}(t/\Delta t)$ steps (the reference cache; the training data have a one-step offset, 7.2). A periodic stencil sums to zero, so a step leaves $\langle\psi\rangle$ unchanged up to round-off.

**Initial condition** (`initial_condition`, `make_ic`, `set_ic`; the RNG stream of `ch.py`). With `np.random.seed(seed)`:

$$\psi_0(\mathbf r)=u(\mathbf r)-\langle u\rangle+\bar\psi_0,\qquad u(\mathbf r)\sim U(-0.1,\,0.1)\ \text{independently on every site}$$

so $\langle\psi_0\rangle=\bar\psi_0$. The compositions are $\bar\psi_0\in\{-0.4,-0.2,0,0.2,0.4\}$ (`OFF_VALUES` in `ch_gpu.py`, `DATASET_OFFS` in `fno_predict.py`); $\bar\psi_0=0$ is the critical quench.

### 7.2 Dataset, references and test protocol

**Training data** (`ch_gpu.py`, file `Work/data/test_t<END:04d>.npz`). The box is $l=64$; there are 25 trajectories (`NENS`): 5 compositions with 5 realisations each. The split of `ch.py` is 20 train and 5 validation (4 and 1 per composition; with `np.random.seed(42)`, the 20 training seeds are `randint(0, 10000)` draws and the 5 validation seeds `randint(10000, 20000)` draws). Because `TRAIN_ON_VAL = True`, all 25 are trained on. A frame is kept every `GAP = 10` solver steps (0.1 time units); frame $j$ is the state after $10j+1$ steps, that is $t_j=0.1\,j+0.01$. A trajectory to $t=\mathrm{END}$ has $F=10\,\mathrm{END}$ frames. The arrays are $X$ = frames $0,\dots,F-11$ and $y$ = frames $10,\dots,F-1$, so $y_j=X_{j+10}$: the label is `N_AHEAD // GAP` = 100/10 = 10 frames, one time unit, ahead of the input (`N_AHEAD = 100` solver steps). END is the last time of the data (`--t-train`: 300, 500 or 1000). The default (variable-step) training uses $X$ only and builds its targets from chains of frames (7.4).

**Reference cache** (`fno_gt.py`). One file per (lattice $l$, composition, seed) with the exact solution (float64 solve, stored as float32) every `GT_EVERY = 4.0` time units from `GT_T0 = 100` to 10000, 2476 frames. Training validation and prediction both read it.

**Test protocol.** A rollout starts at $t_0=100$ from the exact reference state, so $R^2(t_0)=1$ by construction, on a box of $l=128$ (four times the training area at the same $\Delta x$) and runs to $t_{\rm end}=10000$. Since 2026-11-09 a run has 15 cases: the 5 compositions at each box size $l=128,256,512$ (`SIZE_SWEEP`). Before, a run had the 8 cases of the old protocol: the 5 compositions at $l=128$, and the critical quench at three larger box sizes, one of them $l=256$ (the case $l=128$, $\bar\psi_0=0$ shared). Since 2026-11-11 `report.txt` pools the "common 6 cases" as its headline: the 5 compositions at $l=128$ and the critical quench at $l=256$, the cases every run has. Each case has 50 test seeds, the group protocol's fixed list (`fno_predict.py` lines 102 to 108; the default for a dt-conditioned model, set in `apply_overrides`). The 10 validation and tuning seeds are `VAL_SEEDS` = 31337, 24601, 8675309, 20250, 42424, 90210, 23571, 55555, 48151, 62831. None of them is in the 50 test seeds, and a tuning seed that is a test seed is dropped (`tune_seeds`).

### 7.3 Model

`CahnHilliardFNO` maps $\psi(t)$ to $\psi(t+\mathrm{dt})$ for a step $\mathrm{dt}$ given at every call (`DT_CONDITIONED`, the default). Defaults: $C=$ `CHANNELS` $=38$, $M=$ `MODES` $=24$, $L=$ `LAYERS` $=5$.

**Input**, 3 channels (`MU_CHANNEL`): the field, its lattice mean as a constant map, and the chemical potential of the clamped field,

$$x=\Big(\psi,\ \langle\psi\rangle\mathbf 1,\ \mu[\tilde\psi]\Big),\qquad \tilde\psi=\operatorname{clip}(\psi,-1.2,\,1.2),\qquad \mu[\tilde\psi]=\tilde\psi^{3}-\tilde\psi-\nabla^2_5\tilde\psi$$

with `MU_CLIP = 1.2`. Only the third channel sees the clamp; $\psi$, the update and the head do not.

**Lift.** $h_0=W_2\,\mathrm{GELU}(W_1x)$: two 1×1 convolutions, $3\to C\to C$. GELU is `F.gelu`, PyTorch's default (exact) form.

**One Fourier layer**, $\ell=1,\dots,L$:

$$h_\ell=h_{\ell-1}+W^{\rm mix}_\ell\,\mathrm{GELU}\Big(\big[\mathcal K_\ell h_{\ell-1}+\mathcal L_\ell h_{\ell-1}\big]\odot(1+g_\ell)+b_\ell\Big)$$

$\mathcal K_\ell$ is the spectral convolution, $\mathcal L_\ell$ the local kernel (a 3×3 convolution $C\to C$ with circular padding, `LOCAL_KERNEL = 3`), $W^{\rm mix}_\ell$ a 1×1 convolution, and $g_\ell,b_\ell\in\mathbb R^{C}$ the per-channel scale and shift from the dt conditioning (below), broadcast over the lattice.

**Spectral convolution.** With $\hat h=\mathrm{rfft2}(h)$ (no scaling in the forward transform; the inverse carries $1/l^2$), the modes $-m\le k_x\le m$, $0\le k_y\le m$ are kept and all others are set to zero:

$$\widehat{(\mathcal Kh)}_o(k_x,k_y)=\sum_{i}R_{io}(k_x,k_y)\,\hat h_i(k_x,k_y),\qquad \mathcal Kh=\mathrm{irfft2}\big(\widehat{\mathcal Kh}\big)$$

$R$ is stored on the training lattice ($n_{\rm ref}=64$) as a $(2M+1)\times(M+1)$ complex array for every channel pair, indexed by mode number $j$, at the physical wavenumber $k=2\pi j/n_{\rm ref}$ (initialised as $C^{-2}$ times a standard normal). On a box of $l$ points the layer keeps (`MODE_SCALING = "physical"`, `modes_for`)

$$m(l)=\max\Big(1,\ \min\big(\mathrm{round}(M\,l/n_{\rm ref}),\ \lfloor (l-1)/2\rfloor\big)\Big)=24,\ 48,\ 96,\ 192\quad\text{for } l=64,\,128,\,256,\,512$$

modes. If $m(l)\neq M$, $R$ is resampled to $(2m+1)\times(m+1)$ by bilinear interpolation (`F.interpolate`, `align_corners=True`, real and imaginary parts as separate channels), so that new mode $j'$ takes the weight of source mode $j'M/m(l)$, which is the same physical wavenumber when $m(l)=M\,l/n_{\rm ref}$. On $l=n_{\rm ref}$ nothing is resampled. (`MODE_SCALING = "index"`, the stock-FNO ablation, keeps $\min(M,\lfloor (l-1)/2\rfloor)$ modes and reuses $R$ as it is.)

*Zero wavenumber* (`ZERO_DC`). $R(0,0)$ is masked to 0, so the spectral path never carries the lattice mean. *Low-k fill* (`LOWK_FILL`; only when $R$ is resampled): before the interpolation

$$R(0,0)\leftarrow\tfrac13\big[R(1,0)+R(-1,0)+R(0,1)\big]$$

(the three trained neighbours; $R(\pm1,0)$ is $k_x=\pm1$ at $k_y=0$, $R(0,1)$ is $k_y=1$ at $k_x=0$), and after the interpolation the new zero-wavenumber entry is set to 0 again.

**dt conditioning.** $\ln\mathrm{dt}$ (with $\mathrm{dt}$ clamped below at $10^{-6}$) is embedded as

$$e(\mathrm{dt})=\Big[\sin(f_k\ln\mathrm{dt})_{k=0}^{31},\ \cos(f_k\ln\mathrm{dt})_{k=0}^{31},\ \ln\mathrm{dt}\Big]\in\mathbb R^{65},\qquad f_k=64^{k/31}$$

and $c=\mathrm{GELU}\big(W_4\,\mathrm{GELU}(W_3e)\big)\in\mathbb R^{64}$ (`COND_DIM = 64`). Every layer takes $(g_\ell,b_\ell)=\mathrm{chunk}(W^{\rm film}_\ell c)$, a linear map to $2C$ numbers (FiLM) that starts at zero, so the block does not depend on dt at the start of training.

**Head, gain and update.** $P$ is a 1×1 convolution $C\to2C$, GELU, a 1×1 convolution $2C\to1$ (the last one starts at zero). The output gain is

$$G(\mathrm{dt})=\exp\big(a+p\,\ln\mathrm{dt}\big)$$

with learned scalars $a$ (`log_gain`, starts at 0) and $p$ (`gain_exp`, starts at 1), so $G$ starts at $\mathrm{dt}$. With the Laplacian head (`LAPLACIAN_OUTPUT = True`, `--variant lap`) the increment is $\delta=\nabla^2_5\big[G(\mathrm{dt})\,P(h_L)\big]$; with the plain head (`--variant base`) it is $\delta=G(\mathrm{dt})\,P(h_L)$. The update, with the hard mass projection (`HARD_MASS_CONSERVATION`, `RESIDUAL_PREDICTION`), is

$$\psi(t+\mathrm{dt})=\psi(t)+\big(\delta-\langle\delta\rangle\big)$$

so $\langle\psi\rangle$ is the same after a step of any length. With the Laplacian head $\langle\delta\rangle$ is already zero up to round-off. (`HARD_CONSTRAINTS = False` removes the projection and the zero wavenumber mask and adds the penalty $0.1\,\overline{(\langle\hat\psi\rangle-\langle\psi\rangle)^2}$, `MASS_WEIGHT`.)

**Parameter count.** Counted from the layer shapes (a complex spectral weight is 2 real numbers), with $d=64$ and 3 input channels:

$$N=\big(C^2+5C\big)+L\Big[2C^2(2M+1)(M+1)+10C^2+132C\Big]+\big(2C^2+4C+1\big)+8384+2$$

The terms are the lift, $L$ layers (spectral weights; the 3×3 and 1×1 convolutions; FiLM), the head, the dt MLP and the two gain scalars. For $C=38$, $M=24$ it gives 17,799,341, the count of §3.5; without the μ channel the lift term is $C^2+4C$ and the count is 17,799,303. The spectral weights are 17.69 M of the 17.80 M.

**Fixed-step model** (`--fixed-step`). No dt embedding, FiLM or gain, no μ channel; one call advances `DT_STEP` = `N_AHEAD * DT` = 1.0.

### 7.4 Training (default: variable time step)

**Sampling.** A sample is a chain of $n$ steps that all have one stride. The stride is $\mathrm{dt}=0.1\,d$ (`FRAME_DT = GAP * DT = 0.1`) with

$$d=\mathrm{round}\Big(\exp U\big[\ln 10,\ \ln d_{\rm hi}\big]\Big),\qquad d_{\rm hi}=\min\Big(500,\ \Big\lfloor\frac{F_X-1-350}{n}\Big\rfloor\Big),\qquad F_X=10\,\mathrm{END}-10$$

so dt is log-uniform from 1 to 50 (`DT_MIN`, `DT_MAX`) in steps of 0.1, capped so that the chain fits the $F_X$ frames of $X$. The trajectory is uniform over the 25; the first frame is a uniform integer from 350 (that is $t\ge35$, `T_MIN`) to the last start that fits; the chain is $(\psi_{f_0},\psi_{f_0+d},\dots,\psi_{f_0+nd})$.

**Augmentation** (`ChainSampler.augment`): one draw per batch, applied to the whole chain: a rotation by $k\cdot90^\circ$ with $k\in\{0,1,2,3\}$ uniform, a flip along one axis with probability $\tfrac12$ (together the 8 square symmetries), and $\psi\to-\psi$ with probability $\tfrac12$. Translations are not used.

**Input noise** (`VDT_NOISE = 0.015`). $\sigma=0.015\,u$ with $u\sim U(0,1)$ drawn once per batch. The first input of the chain becomes $\psi+(\xi-\langle\xi\rangle)$ with $\xi_{\mathbf r}\sim\mathcal N(0,\sigma^2)$ independent on every site. The targets are clean.

**Chain length.** $n=1$ for epochs 1 to 6, $n=2$ for 7 to 15, $n=4$ for 16 to 80 (a stage starts at epoch $1+\lfloor fE\rfloor$ for $f=0,\,0.08,\,0.19$, `VDT_UNROLL_STAGES`). The batch holds $\max\big(2,\lfloor128/\min(n,4)\rfloor\big)$ = 128, 64, 32 chains (`BATCH_SIZE = 128`). The gradient flows through every step (`VDT_GRAD_STEPS = 4`), and from the second step on the input of a step is the model's own previous output.

**Loss.** Let $\hat\psi_0$ be the (noisy) chain input, $\hat\psi_k=\psi_\theta(\hat\psi_{k-1},\mathrm{dt})$ the model's step $k$, and $\psi_k$ the clean frame at step $k$. The loss of step $k$ is

$$\ell_k=\underbrace{\frac{\lVert\hat\psi_k-\psi_k\rVert^2}{\lVert\psi_k-\langle\psi_k\rangle\rVert^2}}_{\texttt{r2\_loss}}+0.25\underbrace{\frac{\lVert\hat\psi_k-\psi_k\rVert^2}{\lVert\psi_k-\hat\psi_{k-1}\rVert^2}}_{\texttt{increment\_loss}}+0.25\underbrace{\frac{\lVert\nabla\hat\psi_k-\nabla\psi_k\rVert^2}{\lVert\nabla\psi_k\rVert^2}}_{\texttt{gradient\_loss}}$$

and the loss of a batch is $\frac1n\sum_{k=1}^{n}\ell_k$, averaged over the chains of the batch. The norms are sums over one sample's lattice, denominators are clamped at $10^{-12}$, and $\hat\psi_{k-1}$ in the increment term is detached. $\nabla$ is the forward difference with periodic wrap, $\lVert\nabla f\rVert^2=\sum(f_{i+1,j}-f_{i,j})^2+\sum(f_{i,j+1}-f_{i,j})^2$. The first term is $1-R^2$ of the sample (7.7); the second is 1 for a model that returns its input, however small the step; the weights are `W_INC = W_GRAD = 0.25`. The default training does not use the relative-L2 and H1 loss, the smoothness and amplification penalties, or the batch spike guard of the fixed-step loop, and `MASS_WEIGHT` acts only in the no-hard-constraints ablation.

**Optimiser.** AdamW with $\beta=(0.9,\,0.95)$ (`VDT_BETAS`), weight decay $10^{-4}$, global gradient-norm clip 1.0, 200 steps per epoch (`STEPS_PER_EPOCH`), 80 epochs (`EPOCHS`). A step with a non-finite loss or gradient is skipped. The learning rate is set once per epoch $e=0,\dots,E-1$:

$$\eta(e)=\Big[\eta_{\min}+(\eta_0-\eta_{\min})\,\tfrac12\Big(1+\cos\tfrac{\pi e}{E-1}\Big)\Big]\;0.65^{\,s(e)},\qquad \eta_0=3\times10^{-3},\ \ \eta_{\min}=10^{-5}$$

where $s(e)\in\{0,1,2\}$ is the chain-length stage (`LR`, `VDT_LR_MIN`, `VDT_STAGE_LR_MULT`). It is $3.0\times10^{-3}$ in epoch 1, $1.9\times10^{-3}$ in epoch 7, $1.2\times10^{-3}$ in epoch 16 and $4.2\times10^{-6}$ in epoch 80 (the `lr` printed on the log line of epoch $e$ is the rate of epoch $e+1$).

**Weight average (EMA).** After every optimiser step $\bar\theta\leftarrow d_n\bar\theta+(1-d_n)\theta$ with $d_n=\min\big(0.999,\,(1+n)/(10+n)\big)$, where $n$ is the number of earlier updates (`EMA_DECAY = 0.999`). $\bar\theta$ is what is validated and saved as the best model.

**Epoch guard.** An epoch is flagged diverged when its mean train loss is not finite, or more than $0.1\times200$ steps were skipped, or $\delta_{\rm rel}$ is not finite or above 0.5 (`MAX_DELTA_REL`), or its train loss is above 3 times (`DIVERGE_FACTOR`) the median of the last 10 healthy epochs. The weights and the optimiser go back to the last healthy epoch, the EMA restarts from them, and the base learning rate is multiplied by 0.8 (`LR_RECOVERY_DECAY`), at most 8 times (`MAX_RECOVERIES`; with no healthy epoch yet, or after 8, training stops; a resumed run starts with its resume checkpoint as the healthy epoch, a fresh run with none); the epoch is logged with `"diverged": true`. $\delta_{\rm rel}$ (`delta_rel`, `d/psi`) is the mean over the epoch's steps of $\big[\lVert\hat\psi_1-\hat\psi_0\rVert^2/\lVert\hat\psi_0-\langle\hat\psi_0\rangle\rVert^2\big]^{1/2}$, the relative size of the first predicted step.

**Logged values** (`fno_ch_history.json`, one entry per epoch). `train` is the mean over the epoch's non-skipped steps of the batch loss above, on the live weights, with noise and augmentation. `val1` is the one-step check on the EMA weights: for each of the strides $\mathrm{dt}\in\{4,12,40\}$ (`VAL1_DTS`) it takes 24 random (seed, frame) pairs (`VAL1_WINDOWS`; one fixed draw, reused every epoch) of the validation references ($l=128$, critical quench, 10 seeds, $t$ from 100 to 10000), makes one model step from the true state, and takes the sample mean of $1-R^2$ (`r2_loss`); `val1` is the mean over the three strides. A validation epoch adds `score`, `survival`, `per_seed`, `in_range` and `lowk` (7.5).

**Fixed-step model** (`--fixed-step`; the September runs, progress §3). Block `FIXED_STEP`: batch 32, 1500 steps per epoch, learning rate $2\times10^{-3}$, no EMA, no dt conditioning, 5 validation seeds to $t=2000$ every 5 epochs, patience 8, selection by `score`. Per step the loss is

$$\ell=\overline{\frac{\lVert\hat\psi-\psi\rVert_2}{\lVert\psi\rVert_2+10^{-8}}}+0.5\,\overline{\frac{\lVert\nabla\hat\psi-\nabla\psi\rVert_2}{\lVert\nabla\psi\rVert_2+10^{-8}}}\quad(\text{relative }L^2\text{ and relative }H^1\text{ seminorm; }\texttt{H1\_WEIGHT})$$

averaged over the unrolled steps, plus $1.0\cdot\overline{\mathrm{ReLU}(A-0.25)^2}$ with $A=\lVert S_\theta(\psi+e)-S_\theta(\psi)\rVert_2/\lVert e\rVert_2$ for a random zero-mean $e$ with $\lVert e\rVert_2=10^{-3}$ (`CONTRACTIVITY_WEIGHT`, `AMP_TARGET`, `AMP_EPS`), plus $10^{-3}$ times the spectral smoothness penalty (`SMOOTH_WEIGHT`), the mean over layers of $\overline{(R_{k_x+1,k_y}-R_{k_x,k_y})^2}+\overline{(R_{k_x,k_y+1}-R_{k_x,k_y})^2}$. The unroll length $k_{\max}$ is 1 for the first $15\,\%$ of the epochs (`WARM_FRAC`), then 2, 4, 8 in three stages of equal length that fill the span from $15\,\%$ to $55\,\%$ of the epochs (`RAMP_FRAC`), `MAX_UNROLL = 8`, held to the end; each batch draws its step count $k$ uniformly from $1,\dots,k_{\max}$, and only the last `BPTT = 2` steps carry gradient. The learning rate is a cosine from `LR` to `LR_FLOOR` $\times$ `LR` times 0.7 (`LR_STAGE_DECAY`) per doubling. Noise of standard deviation 0.01 (`NOISE_STD`) is added to the input with probability 0.5 (`NOISE_PROB`). The score is $\frac1{N_t}\sum_i\max(R^2_i,0)$ over a rollout from $t=100$ to 2000 (mean over seeds); a new best needs a score above the best by `MIN_DELTA = 2e-3`; `--select survival` uses the rule of 7.5 with $\operatorname{clip}(R^2,0,1)$ up to $t=2\,\mathrm{END}$.

### 7.5 Validation and checkpoint selection

Every 2nd epoch (`VAL_EVERY`) and at the last, on the EMA weights: the 10 `VAL_SEEDS`, $l=128$ (`VAL_LATTICE`), $\bar\psi_0=0$, rolled out from the reference at $t=100$ to $t=10000$ whatever END is (`VAL_T_START`, `VAL_T_END`) on the stride schedule of 7.6 with $\alpha=0.2$, $\mathrm{dt}\in[1,40]$, quantum 4 (`VAL_ALPHA`, `VAL_DT_MIN`, `VAL_DT_MAX`, `VAL_QUANTUM`): 249 steps. At every time $R^2_s(t_i)$ (7.7) is taken against the reference frame. A seed whose field is not finite or has $\max\lvert\psi\rvert>5$ (`VAL_DIVERGE_ABS`) is frozen at its last good state and scores $-1$ from that step on. Unlike the prediction scores, a finite $R^2$ below $-1$ is not clipped here, and the validation $R^2$ is computed in the model's precision (fp32).

**Per-seed survival.** Let $i$ be the first step with $R^2_{s,i}<0.80$ (`R2_THRESHOLD`) or not finite. Then

$$T_s=t_{i-1}+(t_i-t_{i-1})\,\frac{R^2_{s,i-1}-0.80}{R^2_{s,i-1}-R^2_{s,i}}$$

(linear interpolation between the step before and the step at the crossing); $T_s=t_i$, not interpolated, if $R^2_{s,i}$ is not finite (a divergence); $T_s=10000$ if there is no such step. `survival` is $\frac1{10}\sum_sT_s$ and `per_seed` lists the $T_s$.

**Score and in-range accuracy.**

$$\texttt{score}=\frac{1}{N_tN_s}\sum_{i,s}R^2_s(t_i),\qquad \texttt{in\_range}=\frac{1}{N_{\rm in}N_s}\sum_{t_i\le t_{\rm in},\,s}\operatorname{clip}\big(R^2_s(t_i),\,0,\,1\big),\qquad t_{\rm in}=\min(2\,\mathrm{END},\,10000)$$

Both sum over all schedule times including the start sample ($R^2=1$), and a diverged step counts $-1$ in `score` and 0 in `in_range`. `in_range` is the accuracy where the model was trained and up to twice as far; it is stored as `accuracy` in the checkpoint.

**Selection** (`SELECT = "survival"`, `better_epoch`, $\Delta=$ `SURVIVAL_MIN_DELTA` $=1.0$). Let $T,A$ be the `survival` and `in_range` of this validation and $T^\star,A^\star$ those of the best so far. This epoch becomes the best if

$$T>T^\star+\Delta\qquad\text{or}\qquad\big(T\ge T^\star-\Delta\ \ \text{and}\ \ A>A^\star\big)$$

The first validation always qualifies. `fno_ch.pt` holds the EMA weights of the last epoch that became best, the selected epoch (`best["epoch"]`). Its `epoch`, `score`, `survival`, `accuracy` (the in-range value) and `peak_survival` are printed in `report.txt` section 1. `peak_survival` is the largest survival of an epoch that became best. The default training records it but its rule compares with the current best only; the fixed-step `--select survival` rule measures the band $\Delta$ from it.

**Low-k gain in the history.** At each validation the entry `lowk` holds the gain of 7.9 at $t_0=1000$ and $t_0=9952$ ($=10000-$ `LOWK_DT`), for the 10 validation seeds.

### 7.6 Prediction: step schedule, tuning and divergence

**Schedule** (`fno_gt.make_schedule`). The rollout visits the times $t_0=100<t_1<\dots<t_{\rm end}$ with

$$\mathrm{dt}_n=\max\Big(q,\ q\cdot\mathrm{round}\Big(\frac{\operatorname{clip}(\alpha\,t_n,\ \mathrm{dt}_{\min},\ \mathrm{dt}_{\max})}{q}\Big)\Big),\qquad t_{n+1}=\min\big(t_{\rm end},\ t_n+\mathrm{dt}_n\big)$$

with the quantum $q=4$ (`GT_EVERY`, so every time is a reference frame) and $\mathrm{dt}_{\min}=1$ (`SCHED_DT_MIN`); `round` is Python's (halves go to the even integer). The times that must be visited (`schedule_must_hit`: the snapshot times 100, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, the view horizons and $t_{\rm end}$) are inserted into the list. The model is called once per step with $\mathrm{dt}_n=t_{n+1}-t_n$, the same for all seeds. For $\alpha=0.2$, $\mathrm{dt}_{\max}=40$ this is 256 steps to $t=10000$ (the validation schedule of 7.5, without inserted times, has 249).

**Tuning** (`tune_schedule`, `TUNE`), once per quench (composition) on the tuning seeds: the 20 pairs

$$(\alpha,\ \mathrm{dt}_{\max})\in\{0.10,\,0.20,\,0.35,\,0.50,\,0.70\}\times\{30,\,40,\,45,\,50\}$$

(`TUNE_ALPHAS`, `TUNE_DT_MAXS`), each rolled out on the 10 `VAL_SEEDS` at $l=128$ (any that is also a test seed dropped). The score of a pair is

$$S(\alpha,\mathrm{dt}_{\max})=\frac14\sum_{T\in\{T_{\rm end}/4,\ T_{\rm end}/2,\ 3T_{\rm end}/4,\ T_{\rm end}\}}\ \frac1{N_s}\sum_sR^2_s(T)$$

(for $t_{\rm end}=10000$: $T=2500,5000,7500,10000$, moved to the 4.0 grid), with each $R^2_s$ floored at $-1$ and a diverged seed at $-1$. The tuning schedule also visits these four times. The pair with the highest score is kept (the first on a tie), stored in `meta.json`, reused by a resubmitted job and used for all test seeds of the quench; the larger boxes use the pair of the same composition (`schedule_case` calls `schedule_pick` with the composition only). `--no-tune` uses `SCHED_ALPHA = 0.20`, `SCHED_DT_MAX = 40`.

**Edge picks and diverged candidates.** After each pick the tuner prints `[tune] k of N candidates had at least one diverged tuning seed`, and one warning for each grid edge the pick sits on (the smallest or largest $\mathrm{dt}_{\max}$ or $\alpha$; on the smallest, a smaller step was not tried). `report.txt` section 5 (STEP SCHEDULE) gives each quench's pick with its steps, score and source on one line, and the count of diverged candidates and the edge flags on the next. An old run is judged on the grid in its pick's signature. The table of every candidate is in `meta.json` (key `schedule`). The END = 500 default picked the smallest $\mathrm{dt}_{\max}$, 30, in all five quenches (progress §6).

**Divergence rule.** After every step, a seed whose field is not finite or has $\max\lvert\psi\rvert>5$ (`DIVERGE_PSI = 5.0`) is diverged. From that time to the end of the schedule its $R^2$ is $-1$ (`R2_FLOOR`), its mean $\langle\psi\rangle$ and snapshots of the FNO are NaN, its field is set to 0 and its later model outputs are ignored; the reference is still read. It stays in every mean. The rollout is not stopped by the 0.80 threshold (`EARLY_STOP = False`); `t_stop` records the first time $R^2<0.80$. When no seed is alive the model is not called again.

### 7.7 R² and the scores

**$R^2$ of one snapshot** (`fno_train.r2_score`, float64). For one seed and one time,

$$R^2(t)=1-\frac{\sum_{\mathbf r}\big(\psi_{\rm ref}(\mathbf r,t)-\psi_{\rm FNO}(\mathbf r,t)\big)^2}{\sum_{\mathbf r}\big(\psi_{\rm ref}(\mathbf r,t)-\langle\psi_{\rm ref}\rangle(t)\big)^2}$$

The mean subtracted is the lattice mean of the reference field at that time and seed (which equals $\bar\psi_0$), not a mean over time or over seeds. If the denominator is zero the value is 1 when the numerator is zero and $-\infty$ otherwise. The reported series is floored, $R^2_s(t_i)\leftarrow\max\{R^2_s(t_i),-1\}$ (`R2_FLOOR`), and is $-1$ after a divergence. $R^2_s(t_0)=1$ by construction and is a sample of the series.

**Means over seeds at a time $t$.** The *all-seeds mean* $\overline{R^2}(t)=\frac1N\sum_sR^2_s(t)$ keeps a diverged seed at $-1$. The *survivors-only mean* is the same sum over the seeds that had not diverged by $t$ (`mean_alive`; `None` if there are none). Seeds that do not reach $t$ are not counted.

**Spread.** A "sd" in a table of scalars (`R2(t_end) ± sd`, the trajectory-mean sd, the survival sd) is the sample standard deviation over seeds, $\big[\frac1{N-1}\sum_s(x_s-\bar x)^2\big]^{1/2}$ (`_sd`, ddof 1). The $\pm1$ sd bands of a curve use the population form ($\mathrm{ddof}=0$, `np.nanstd`) over the seeds still running at each time: $R^2(t)$ in `r2_sweep` and `r2_curve` of the manifest, $L_{1/2}$ in `domain_growth`, $\langle\psi\rangle$ in `composition`. Only panel (a) of `phase_ordering_kinetics` uses ddof 1 (`seed_sd`; 0 where fewer than two seeds are left).

**$R^2(t_{\rm end})$** of a case is the all-seeds mean of $R^2_s(t_{\rm end})$ (the last snapshot, 10000 for a full run); `median`, `best` and `worst` are the median (`np.median`), maximum and minimum of $R^2_s(t_{\rm end})$ over seeds. These are computed from the stored snapshot pair of each seed with `r2_score`, so a diverged seed (NaN field) is set to $-1$ but a finite value below $-1$ is not floored here, as it is in the series.

**Trajectory-mean $R^2$** of a seed is a plain average over the schedule samples,

$$\bar R^2_s=\frac1{N_t}\sum_{i=0}^{N_t-1}R^2_s(t_i)$$

with the start sample included. Every step weighs the same, so with strides that grow with $t$ this is not a time average: early times weigh more. The case value is the mean over seeds, with a sample sd.

**Survival time** (manifest, `report.txt`). $T_s$ is the first schedule time with floored $R^2_s<0.80$, not interpolated, and $t_{\rm end}$ if there is none; `survives` is the mean over seeds, with a sample sd. `never` (`n_survive_to_end`) counts the seeds with no sample below 0.80, and `mean_curve_crosses` is the first time the seed-mean curve is below 0.80. (The validation survival of 7.5 is interpolated.)

**Group and pooled $R^2$ at the reporting times** $t=2000,5000,10000$ (`R2_REPORT_TIMES`). For each case $c$ let $n_c$ be its number of seeds that reach $t$ and $\overline{R^2_c}(t)$ its all-seeds mean. A group's value is the seed-weighted mean

$$\overline{R^2}_{\rm group}(t)=\frac{\sum_{c\in\rm group}n_c\,\overline{R^2_c}(t)}{\sum_{c\in\rm group}n_c}$$

The groups are *pooled* (the 15 distinct cases), *compositions* (the 5 cases at $l=128$) and *box sizes* (the 11 cases: the critical quench at $l=128$ and every composition at $l=256,512$). The case $l=128$, $\bar\psi_0=0$ is in both groups and counted once in the pooled one. A cell of `report.txt` reads `all-seeds mean (survivors-only mean) number diverged`; the survivors-only group mean is weighted by the number of surviving seeds.

**Best, median and worst seed.** Seeds are ranked by the trajectory mean $\bar R^2_s$ (`np.nanmean` of the series). The *best* seed (the one shown in the snapshots and the film) is the highest-ranked among the seeds that never diverged and have a finite prediction at the final snapshot; if there is none, among those with a finite prediction there; if none, among all (`best_index`). The *median* seed is the lower of the middle two (index $(N-1)//2$ of the ascending ranking), the *worst* the lowest. The rank is by $\bar R^2_s$ over every seed, so a best seed need not have the highest mean.

**Curve of a case** (`r2_curve`). At each schedule time the mean (`np.nanmean`) and the population sd of the floored $R^2_s$ over the seeds still running, thinned to about 400 log-spaced samples plus the snapshot and reporting times (`CURVE_POINTS = 400`) for `manifest.json`. Case mean $R^2$ is the mean of all samples of all seeds.

**Where the numbers are reported.** `report.txt` gives the pooled, compositions and box-sizes rows at each reporting time, in the cell form above, so the two means can be read side by side. `manifest.json` has the same per case (`r2_at_times`, `mean_alive`, `n_alive`); `Results/README.md` gives each case's $R^2(t_{\rm end})$. In the schedule engine a diverged seed is scored $-1$ from its divergence to the end of the schedule; in the older engines a seed that stopped early counts only up to its stop. There are 50 test seeds per case by default (the group protocol's 50 for a dt-conditioned model, this project's own 50 for a fixed-step one).

### 7.8 Physics diagnostics

All are computed on the truth and on the FNO at every schedule time after the start (`every = 1` in the schedule engine; `DIAG_EVERY = 10` in the older engines), per seed, and then averaged over seeds with `np.nanmean`, except the snapshot spectra of panels (b) to (d) of 7.10, which are averaged over seeds first.

**Structure factor** (`structure_factor`). With $\hat\psi'(\mathbf k)=\sum_{\mathbf r}\psi'(\mathbf r)e^{-i\mathbf k\cdot\mathbf r}$ (`np.fft.fft2`, no scaling), $\mathbf k=2\pi\,\mathrm{fftfreq}(l)$ and $\Delta k=2\pi/l$:

$$S(k,t)=\Big\langle\frac{\lvert\hat\psi'(\mathbf k)\rvert^2}{l^2}\Big\rangle_{\mathbf k\ \text{with}\ \mathrm{round}(\lvert\mathbf k\rvert/\Delta k)=k/\Delta k}$$

The shells are $n=0,1,\dots,l/2$ ($\lvert\mathbf k\rvert$ is rounded to the nearest multiple of $\Delta k$ and clipped at $l/2$), the value is the mean over the modes in a shell, and the shell is at $k=n\,\Delta k$. $S$ of the zero shell is 0.

**Correlation function** (`correlation_function`). $c(\mathbf r)=\frac1{l^2}\sum_{\mathbf x}\psi'(\mathbf x)\psi'(\mathbf x+\mathbf r)$ is the circular autocorrelation, computed as $\mathrm{Re}\,\mathrm{ifft2}\lvert\hat\psi'\rvert^2/l^2$. It is averaged over shells $r=\mathrm{round}\big(\sqrt{r_x^2+r_y^2}\big)$ with $r_x=\min(i,l-i)$, shells clipped at $l/2$, and normalised,

$$C(r,t)=\bar c(r)/\bar c(0)$$

**Length scales** (`all_lengths`; all in lattice units):

- $L_{\rm zero}$, the first zero of $C$ (`LENGTH_KEY`, used for $L(t)$ in the kinetics figure, the finite-size mark and the exponent "while $L\le l/8$"): $i$ is the first shell with $C_i>0$ and $C_{i+1}\le0$, then $L=r_i+(r_{i+1}-r_i)\,\dfrac{C_i}{C_i-C_{i+1}}$; NaN if there is none.
- $L_{1/2}$ (`L_half`, the main curve of `domain_growth`): $j$ is the first shell $\ge1$ with $C_j\le\tfrac12C_0$, then $L=r_{j-1}+(r_j-r_{j-1})\,\dfrac{C_{j-1}-C_0/2}{C_{j-1}-C_j}$.
- $L_{\rm peak}=2\pi/k_m$, with $k_m$ the shell of the maximum of $S$ over $k>0$, refined by a parabola through the three shells $y_0,y_1,y_2$ around it: $k_m=k_i+\operatorname{clip}\Big(\tfrac12\dfrac{y_0-y_2}{y_0-2y_1+y_2},-1,1\Big)\Delta k$ (no refinement at the edge of the range).
- $L_{\rm moment}=2\pi/\langle k\rangle$ with $\langle k\rangle=\dfrac{\int_{0<k\le1}k\,\big[k^{d-1}S(k)\big]\,dk}{\int_{0<k\le1}k^{d-1}S(k)\,dk}$, $d=2$ (`D`), cut-off $k_{\rm cut}=1.0$, trapezoid rule over the shells.
- $L_{\rm grad}=1/\big\langle(\psi_{i+1,j}-\psi_{i,j})^2+(\psi_{i,j+1}-\psi_{i,j})^2\big\rangle$, forward differences with periodic wrap, the inverse interfacial density.

$L(t)$ of a case is the mean over seeds of each seed's own series, not the length of a mean $C$. At a snapshot (panels (b) to (d) below) $S$ and $C$ are averaged over the seeds first and $L$ is the mean over seeds of each seed's own $L_{\rm zero}$.

**Growth exponent** (`fit_growth_exponent`). The least-squares slope of $\ln L$ against $\ln t$ (`np.polyfit(log t, log L, 1)`), one point per schedule sample (equal weight, finite $t>0$ and $L>0$ only), NaN with fewer than 3 points:

$$n=\frac{\sum_i(\ln t_i-\overline{\ln t})(\ln L_i-\overline{\ln L})}{\sum_i(\ln t_i-\overline{\ln t})^2}$$

Windows: *late*, $[t_{\rm end}/4,\ t_{\rm end}]$ (2500 to 10000; the exponent of the tables, `report.txt` column "n pred", and the title of `domain_growth`); *whole rollout* (first to last sample; the title of kinetics panel (a)); *training window* [first sample, END] and *beyond training* [END, $t_{\rm end}$], when END lies inside the curve; and *while $L\le l/8$* (the first sample to the last one before the seed-mean $L_{\rm zero}$ exceeds $l/8$, at least 3 points). The reference exponent is the Lifshitz-Slyozov value $1/3$ (a drawn slope, not a fit).

**Finite-size regime** (`FINITE_SIZE = 8`). The seed-mean truth $L_{\rm zero}(t)$ is past $l/8$ (16, 32, 64 at $l=128,256,512$) from the first sample where $\overline{L_{\rm zero}}>l/8$; the figures shade it and the manifest stores that time (`first_time_mean_L_zero_above`).

**Porod window** (`porod_window`). With $k_m$ the peak of the seed-mean truth $S$ at the last snapshot, the Porod line $S\propto k^{-3}$ is drawn on

$$k\in\Big[\,2k_m,\ \min\big(5k_m,\ 1/w\big)\Big],\qquad w=\sqrt2\ (\texttt{INTERFACE\_WIDTH}),$$

only if the upper end exceeds $1.2$ times the lower (`POROD_MIN_DECADE`). The line is $S_a\,(k/k_g)^{-3}$ with $k_g=\sqrt{k_{\rm lo}k_{\rm hi}}$ and $S_a$ the truth $S$ interpolated at $k_g$.

**Dynamical scaling.** For each snapshot, the points $\big(kL,\ L^{-d}S(k)\big)$ with $d=2$ and $L=\overline{L_{\rm zero}}$ of that snapshot (the truth's $L$ for the truth curve, the FNO's for the FNO curve); the scaling form tested is $S=L^{d}f(kL)$. The scaling function is $C(r)$ against $r/L$ for $r/L\in[0,5]$.

**Free energy density** (`fno_train.free_energy`, forward differences, periodic, $\Delta x=1$):

$$\langle f\rangle=\Big\langle\tfrac14(\psi^2-1)^2+\tfrac12\big[(\psi_{i+1,j}-\psi_{i,j})^2+(\psi_{i,j+1}-\psi_{i,j})^2\big]\Big\rangle$$

its functional derivative with respect to $\psi$ is the $\mu$ of 7.1. The monotone check is, for each seed, that every consecutive difference of the FNO's series is $\le10^{-9}$; the panel title reads "FNO monotone: yes", "no" (one seed) or "$k$ of $n$ seeds". Only the FNO's series is checked.

**Composition and drift** (`psi_to_a`, `composition_series`, `composition_entry`). With the seed-mean $\bar\psi(t)=\langle\psi\rangle$ of truth and FNO:

$$c_A=100\,\frac{1+\bar\psi}{2}\ \%\ \ (\text{code: }50(1+\bar\psi)),\qquad c_B=100-c_A$$

`drift_pred` $=\bar\psi_{\rm FNO}(t)-\bar\psi_{\rm truth}(t_0)$; `shift_pp` $=c_A^{\rm FNO}(t_{\rm last})-c_A^{\rm truth}(t_0)$ in percentage points, $t_{\rm last}$ being the last time at which the FNO mean is finite; `max_abs_psi_drift_pred` $=\max_t\lvert\bar\psi_{\rm FNO}(t)-\bar\psi_{\rm truth}(t_0)\rvert$; `drift_rate_psi_per_t` is the least-squares slope of $\bar\psi_{\rm FNO}$ against $t$ (`np.polyfit(t-t[0], mean, 1)`) and `drift_rate_pp_per_1000t` is $5\times10^4$ times it ($=50\cdot1000$). The per-seed mass drift is $\lvert\langle\psi_{\rm FNO}\rangle-\langle\psi_{\rm ref}(t_0)\rangle\rvert$ at each time (`mass_drift`; its maximum is `max_mass_drift`). The conservation figure in `extras/` plots $\lvert\langle\psi_{\rm FNO}\rangle-\langle\psi_{\rm truth}\rangle\rvert$ at each time, its mean over seeds and its worst seed, against the float64 epsilon. `composition.csv` and the report list $\bar\psi$ and $c_A$ at the start and at each snapshot time.

**$P(\psi)$ and the lever rule** (`figure_psi_pdf`). All sites of all seeds that reach the final snapshot are pooled; 25 equal bins on $[-1.1,1.1]$ (`PDF_BINS`, `PDF_RANGE`), width 0.088, `density=True`: each histogram integrates to 1 over the binned range, and values outside it are counted as `clipped_points`. Non-finite FNO values (diverged seeds) are left out. The lever rule gives the fraction of the A-rich phase,

$$f_A=\frac{1+\bar\psi_0}{2}$$

and the measured fractions are $f_{\rm truth}=\langle\psi>0\rangle$ and $f_{\rm FNO}$ over the pooled sites; `lever_shift` $=f_{\rm truth}-f_A$ and `lever_shift_pred` $=f_{\rm FNO}-f_A$.

### 7.9 Model diagnostics

**Tile invariance** (`tile_invariance`, `diagnostics.json`, `report.txt` section 6). One step of $\mathrm{dt}=10$ (`DIAG_TILE_DT`) on a $64\times64$ critical-quench field of the first validation seed at $t=500$ (`DIAG_TILE_T`), against the same step on its 2×2 tiling, with a float64 copy of the model on the CPU. With $S$ the step and $\mathrm{tile}$ the 2×2 repetition,

$$\max\big\lvert S(\mathrm{tile}\,\psi)-\mathrm{tile}\,S(\psi)\big\rvert,\qquad \frac{\lVert S(\mathrm{tile}\,\psi)-\mathrm{tile}\,S(\psi)\rVert_2}{\lVert\mathrm{tile}\,S(\psi)\rVert_2}$$

are reported, and the model is called invariant if the first is $\le$ `DIAG_TILE_TOL` $=10^{-5}$.

**Low-k increment gain** (`lowk_gains`, `lowk_masks`; `report.txt` section 7). From the true state at $t_0$ the model takes one step of $\mathrm{dt}=$ `LOWK_DT` $=48$. Let $\delta P=\psi_\theta(\psi(t_0),48)-\psi(t_0)$ and $\delta T=\psi(t_0+48)-\psi(t_0)$ (reference), both in float64, with hats for the 2-D FFT. For a set $\Omega$ of modes,

$$g_\Omega=\frac{\sum_{\mathbf k\in\Omega}\sum_s\mathrm{Re}\big(\overline{\widehat{\delta P}_s(\mathbf k)}\;\widehat{\delta T}_s(\mathbf k)\big)}{\sum_{\mathbf k\in\Omega}\sum_s\big\lvert\widehat{\delta T}_s(\mathbf k)\big\rvert^2}$$

$g=1$ means the increment has the true size in those modes, $g=0$ that the model leaves them alone. With $\mathbf n=l\,\mathrm{fftfreq}(l)$, $\rho=l/n_{\rm ref}=2$ and $r=\lvert\mathbf n\rvert$ the four sets are: *below* $0<r<\rho$ (longer than the training box), *first* $\rho\le r<1.5\rho$, *mid* $1.5\rho\le r\le4\rho$, *high* $r>4\rho$; $r=0$ is in none. The start times are $t_0=1000$ (`LOWK_T0`) and the last frame with a next one 48 later (9952); `report.txt` prints `below/first/mid/high@t0`. The prediction uses the first 3 validation seeds (`DIAG_LOWK_SEEDS`) at $l=128$; the training history uses all 10.

**Speed-up** (`report.txt` section 3). $\dfrac{\sum t_{\rm solver}}{\sum t_{\rm FNO}}$, the solver seconds of the references that this job had to solve (a reference read from the cache counts 0) over the FNO's model seconds (the time of the model calls, with GPU synchronisation, divided by seeds and calls and multiplied by the steps of each seed). It reads "not available" when nothing was solved.

### 7.10 Figures

Files in `Results/<run>/l<size>/` are named `<kind>_<case>.png` with the case `psi0m0.40` … `psi0p0.40` (the five compositions at that box size, `psi0p0.00` is critical). Before 2026-11-09 the run folder was flat, with the five compositions at $l=128$ and the larger box sizes as separate cases `l<size>`. The shown snapshot times are 100, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000. $\bar R^2_c(t)$ is the seed-mean curve of a case (7.7).

| figure | what it plots |
|---|---|
| `snapshots_<case>`, `snapshots_long_<case>` | the best seed of 7.7. Three rows: $\psi_{\rm truth}$ and $\psi_{\rm FNO}$ (colour $-1$ to $1$) and the absolute error $\lvert\psi_{\rm truth}-\psi_{\rm FNO}\rvert$ (colour 0 to 1). One column per snapshot time ($\le2000$ in `snapshots`, $\ge2000$ in `snapshots_long`); each column title is the time $t$, and the FNO panel carries the $R^2$ of that seed at that time under it ("diverged" or "undefined" when there is no value; the first column is the input frame, $R^2=1$). There is no figure title: the seed shown, its $R^2(t_{\rm end})$ with the median and the worst over seeds (a diverged seed at $-1$) and the number of diverged seeds are in the folder's `figure_notes.txt`. |
| `phase_ordering_kinetics_<case>` | six panels, seed means. (a) $L_{\rm zero}(t)$ of truth and FNO, log-log, $\pm1$ sd (ddof 1), the line $L_g(t/t_g)^{1/3}$ through the geometric means $(t_g,L_g)$ of the truth curve, the exponents of the whole rollout in the title. (b) $S(k,t)$ at the snapshot times (solid truth, dashed FNO) with the Porod line. (c) $S/L^{d}$ against $kL$. (d) $C$ against $r/L$. (e) $\langle f\rangle(t)$ (7.8) and the monotone check. (f) $\bar\psi(t)$ of truth and FNO with $c_A$ on the right axis and the final composition in the title |
| `domain_growth_<case>` | $L_{1/2}(t)$ of truth and FNO, $\pm1$ sd (ddof 0), log-log, with $L_{\rm zero}$ as thin dashed lines. The line $1.22\cdot\overline{L_{1/2}}^{\rm truth}(t_a)\,(t/t_a)^{1/3}$ from $t_a$ to $t_b$: $t_a$ is the first sample $\ge0.45\,t_b$ and $t_b=t_{\rm last}$, or, when the seed-mean $L_{\rm zero}$ passes $l/8$, $t_b$ is the last sample before it does; the factor 1.22 lifts the line off the data. The title gives the late exponent $[t_{\rm end}/4,t_{\rm end}]$ of $L_{1/2}$ and the exponent while $L_{\rm zero}\le l/8$; the legend the training-window exponent of $L_{1/2}$. Grey: the finite-size regime; dashed vertical: END |
| `psi_distribution_<case>` | $P(\psi)$ at the final time (7.8), truth as a filled histogram, FNO as markers, log $y$; the lever-rule line of text |
| `composition_<case>` | $\bar\psi(t)$ of truth and FNO, seed mean $\pm1$ sd (ddof 0), log $t$, with $c_A=50(1+\bar\psi)$ on the right axis |
| `composition_all.png` | $c_A(t)$ of the five compositions of one box size on one log-$t$ axis (the start and the snapshot times), truth solid, FNO dashed |
| `r2_time.png` (in each `l<size>/`, since 2026-11-09) | $\bar R^2_c(t)\pm1$ sd over the seeds of each composition at that box size, linear $t$ axis |
| `r2_sweep_per_size/l<size>.png` (since 2026-11-11; one panel per box size in one file from 2026-11-09 to 2026-11-10) | one image for each box size that has two or more compositions, a line $\bar R^2_c(t)\pm1$ sd (ddof 0) for each composition ($\bar\psi_0=0$ heaviest). Dotted line at END, linear $t$ axis, own axis labels and legend, no title and no panel letter, 300 dpi PNG, no PDF. All images of the folder share one $t$ range and one $R^2$ range: from the lowest point drawn (the $-1$ sd edge of a band) to 1, plus 4 % of that span at each end (at least 0.005). The folder has its own `figure_notes.txt` |
| `r2_sweep_per_case/<composition>.png` (e.g. `psi0p0.40.png`; since 2026-11-11) | the same with one image for each composition that has two or more box sizes, a line for each size. Own `figure_notes.txt`. The one-file versions (`r2_sweep_per_size.png/.pdf`, `r2_sweep_per_case.png/.pdf`) and `r2_sweep`, `r2_compositions`, `r2_boxes` are in `Archive/results_2026-11-11_pre_split/` and `Archive/r2_end_figures_2026-11-09/` |
| `comparison.png` | the critical quench's $\bar R^2(t)$ of each untagged run (default size, training seed 0; every run when fewer than two are untagged), log $t$; colour by training horizon, solid for the Laplacian head, dashed for the plain head; the legend gives $\bar R^2$ at the run's last time |
| `comparison_t<END>.png` (changed 2026-10-28, see below) | the model sizes of one END on one axis |
| `training_curves.png` (added 2026-10-28) | the history of one training, below |
| `comparison_<case>.png` (added 2026-10-28) | `comparison.png` for one case, below |

**`training_curves.png` (added 2026-10-28).** Drawn into a run folder when the model's `fno_ch_history.json` exists; three stacked panels on one epoch axis. (a) `train` and `val1` against epoch on a log axis, with a dotted vertical line labelled "unroll $n$" where the chain length changes; epochs flagged `"diverged"` (for example `train` = 26478 at epoch 38 of the 64-channel, 12-mode run) are left out of the curves and marked. (b) `score` and `in_range` at the validation epochs (every 2nd). (c) `survival` (7.5), with the range of `per_seed` over the 10 validation seeds shaded. The selected epoch (7.5) is a vertical line with a dot on every curve. Each panel has its legend above it; the colours are a colour-blind-safe set. The title gives the parameter count, the selected epoch of the last, and that epoch's `score`, `in_range` and `survival`.

**`comparison_<case>.png` (added 2026-10-28).** As `comparison.png`, for one case: the case's $\bar R^2_c(t)$ (from the stored `r2_curve` of the manifests) of each untagged run (default size, training seed 0), written for every case present except the critical quench, which is `comparison.png` itself (until 2026-10-29 it was also drawn as `comparison_psi0p0.00.png`), and only when at least two of those runs share a case other than the critical one. Colour is the training horizon, taken from all such runs so that a colour means the same horizon in every file; the title gives the composition and $l$.

**`comparison_t<END>.png` (changed 2026-10-28).** The runs with the Laplacian head trained to one END, one panel per mode count present (shared $y$ axis), colour per width (a six-step blue ramp, fixed for widths 16, 24, 32, 38, 45, 64; the default size is black, solid and heavier), line style per training seed (dashed, dotted for the others). It plots the pooled curve

$$\overline{R^2}_{\rm pool}(t)=\frac{\sum_{c=1}^{8}n_c\,\bar R^2_c(t)}{\sum_{c=1}^{8}n_c}$$

the seed-weighted mean of the 8 cases' mean curves, each case's curve interpolated in $\ln t$ (`np.interp`) onto the union of the cases' time grids restricted to the window they share. The legend gives each model's width, its number of trainable parameters (since 2026-10-29) and $\overline{R^2}_{\rm pool}$ at the last time of the curve (with "$k$ of 8 cases" if fewer cases are present). The $y$ axis runs from 0.7 to 1.01 (lower if data fall below), with a dotted vertical line at END; each legend is below its panel. Before this date the figure showed the critical quench's curve of each run.

### 7.11 Constants used above

| group | name = value (file) |
|---|---|
| solver, data | `DT = 0.01`, `DX = 1`, `N_AHEAD = 100`, `GAP = 10`, `NENS = 25`, `OFF_VALUES = [-0.4, -0.2, 0.0, 0.2, 0.4]`, `GT_T0 = 100`, `GT_EVERY = 4.0` (`fno_train.py`, `ch_gpu.py`, `fno_gt.py`) |
| model | `CHANNELS = 38`, `MODES = 24`, `LAYERS = 5`, `LOCAL_KERNEL = 3`, `COND_DIM = 64`, `MODE_SCALING = "physical"`, `ZERO_DC = True`, `LOWK_FILL = True`, `MU_CHANNEL = True`, `MU_CLIP = 1.2`, `LAPLACIAN_OUTPUT = True`, `HARD_MASS_CONSERVATION = True` (`fno_train.py`) |
| training | `EPOCHS = 80`, `BATCH_SIZE = 128`, `STEPS_PER_EPOCH = 200`, `LR = 3e-3`, `VDT_LR_MIN = 1e-5`, `VDT_STAGE_LR_MULT = 0.65`, `WEIGHT_DECAY = 1e-4`, `VDT_BETAS = (0.9, 0.95)`, `GRAD_CLIP = 1.0`, `EMA_DECAY = 0.999`, `DT_MIN = 1.0`, `DT_MAX = 50.0`, `T_MIN = 35.0`, `FRAME_DT = 0.1`, `VDT_UNROLL_STAGES = ((0.0, 1), (0.08, 2), (0.19, 4))`, `VDT_GRAD_STEPS = 4`, `VDT_NOISE = 0.015`, `W_INC = 0.25`, `W_GRAD = 0.25`, `TRAIN_ON_VAL = True`, `MAX_DELTA_REL = 0.5`, `DIVERGE_FACTOR = 3.0`, `MAX_RECOVERIES = 8`, `LR_RECOVERY_DECAY = 0.8`, `PRECISION = "fp32"` (`fno_train.py`) |
| validation | `VAL_SEEDS` (10, 7.2), `VAL_LATTICE = 128`, `VAL_T_START = 100`, `VAL_T_END = 10000`, `VAL_EVERY = 2`, `VAL_ALPHA = 0.2`, `VAL_DT_MIN = 1.0`, `VAL_DT_MAX = 40.0`, `VAL_DIVERGE_ABS = 5.0`, `VAL1_DTS = (4.0, 12.0, 40.0)`, `VAL1_WINDOWS = 24`, `SELECT = "survival"`, `SURVIVAL_MIN_DELTA = 1.0`, `LOWK_T0 = 1000`, `LOWK_DT = 48` (`fno_train.py`) |
| prediction | `START_TIME = 100`, `SCHED_ALPHA = 0.20`, `SCHED_DT_MIN = 1.0`, `SCHED_DT_MAX = 40.0`, `TUNE_ALPHAS = (0.10, 0.20, 0.35, 0.50, 0.70)`, `TUNE_DT_MAXS = (30, 40, 45, 50)`, `SIZE_SWEEP = (128, 256, 512)`, 50 test seeds, `DIVERGE_PSI = 5.0`, `EARLY_STOP = False` (`fno_predict.py`) |
| scores | `R2_THRESHOLD = 0.80`, `R2_FLOOR = -1.0`, `R2_REPORT_TIMES = (2000, 5000, 10000)`, `CURVE_POINTS = 400` (`fno_predict.py`, `make_submission.py`) |
| diagnostics | `LENGTH_KEY = "L_zero"`, `FINITE_SIZE = 8.0`, `INTERFACE_WIDTH = sqrt(2)`, `POROD_KM_RANGE = (2.0, 5.0)`, `POROD_MIN_DECADE = 1.2`, `D = 2`, `DIAG_TILE = 64`, `DIAG_TILE_T = 500`, `DIAG_TILE_DT = 10`, `DIAG_TILE_TOL = 1e-5`, `DIAG_LOWK_SEEDS = 3`, `PDF_BINS = 25`, `PDF_RANGE = (-1.1, 1.1)` (`fno_predict.py`, `make_submission.py`) |
