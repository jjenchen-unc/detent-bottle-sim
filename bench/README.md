# bench: demos, shared evaluation and an ACT baseline

Everything here runs from the repo root and treats `DetentBottle-v0` exactly as shipped. Physics,
cameras, action and observation spaces are unchanged.

## Why

`reset(seed)` always gives the same state and the reference opener uses fixed world coordinates.
So imitation data from it is a single trajectory, and a policy tested at the default pose could
just be memorising it. `bench` varies the bottle pose (`bottle_pose=(x, y, yaw_deg)`) for both
training and testing.

## Findings so far

- **Env bug, fixed in `detent_bottle/scene.py`:** `bottle_pose` yaw is documented in degrees,
  but the scene compiles with `angle="radian"`. So `yaw_deg=10` rotated the bottle by -147 deg.
  It now converts degrees to radians. `tests/test_bench.py::test_bottle_pose_yaw_is_degrees`
  covers it.
- **Expert reach (one axis at a time, `bench/sweep.py`):** it succeeds at x and y offsets of
  +-2 to +-8 cm, except x = -4 cm, where it folds the loop but misses the button. For yaw it
  succeeds at -10 and +10 to +45 deg, and fails at -20, -30 and -45 deg (the loop is never
  folded). These poses are kinematically reachable, so the failure is in contact.
  The expert is not perfect even close to the default pose. `collect.py` keeps only successful
  runs, and the expert's own success on each test split is the reference score.
  `data/expert_v1` holds 100 successful demos from 106 train poses (590 MB at 96 px). Every
  failure folded the loop but missed the button press. Five of the six were at |y| of 3 to 4 cm
  with x near 0.
- **Pose ranges (`bench/poses.py`):** train and test_id use x, y +-4 cm and yaw +-10 deg.
  test_ood goes up to +-6 cm and +-20 deg. The fixed close-up cameras still see the latch
  across this range.
- **The env does not detect a tipped bottle:** `bottle_fell` needs z < .76 m, which only happens
  if the bottle leaves the table. A bottle lying on its side runs to the time limit. The harness
  (`bench/rollout.py`) ends those episodes as `bottle_tipped` once the tilt passes 60 deg. The
  env itself is unchanged.
- **ACT cannot learn from the reference opener's demos, most likely because of its idle waits.** Every
  segment lasts a fixed number of ticks. In most of them the arm reaches its target early, then
  sits still (25-46 ticks, joint motion exactly 0, camera frames nearly identical) until the
  clock runs out, and only then does the gripper close or the next move start. A policy that
  sees only the current observation has no visible cue for when the wait is over. ACT trained on 100 such
  demos (`expert_v1`) reaches the bottle within a few mm but never closes either gripper, so it
  scores 0/50. In open-loop replay on held-out demos its targets are within 1-3 mm at the grip
  and pinch, so it learned where to go but not when to close. We have only tested ACT. Diffusion
  policy or a VLA trained on the same demos may behave differently. `ScriptedExpert(settle=True)`
  (`collect.py --settle`) ends a segment once both tools are on target and the robot has been
  still for 3 ticks, keeping the original tick count as the upper bound. Episodes drop from
  ~714 to ~524 ticks. Contact segments (pinch, fold, press) still run their full length. The
  default `settle=False` reproduces `expert_v1` action for action.
- **Results so far (`python -m bench.report`):** p-values are two-sided exact McNemar tests on
  per-pose outcomes, since every policy runs the same test poses.

  | policy | split | success | loop folded | notes |
  |---|---|---|---|---|
  | scripted expert (privileged pose) | test_id (30) | 96.7% | 100% | 1 missed button press |
  | scripted expert (privileged pose) | test_ood (20) | 85% | 90% | fails at yaw near -15 deg, plus missed presses |
  | settle expert (privileged pose) | test_id (30) | 100% | 100% | ~520 ticks per episode instead of ~715 |
  | settle expert (privileged pose) | test_ood (20) | 90% | 90% | both failures at yaw -14.5 and -17 deg |
  | ACT quick pass (53 demos, frozen backbone, layer 3, 5k steps) | test_id (first 20, 1200-tick cap) | 0% | 5% | Copies the overall motion but tips the bottle while gripping it (9/20) |
  | ACT on `expert_v1` (100 timed demos, layer 3, 20k steps) | test_id (30) / test_ood (20) | 0% / 0% | 0% / 0% | Never closes either gripper (the idle-wait problem above) |
  | **ACT on `expert_v2`** (100 settle demos, layer 3, 20k steps) | test_id (30) | **86.7%** | 86.7% | ~500 ticks per success. 4 failures: 2 at x=+3.7 cm, y=+2 cm never reach the loop; 1 folds the loop but misses the button; 1 tips the bottle mid-fold |
  | **ACT on `expert_v2`** | test_ood (20) | **35%** | 45% | Mostly fails outside the training range, usually before folding the loop, often after pushing the bottle 4-7 cm |
  | ACT on `expert_v3_wide_a` + `_b` (187 clean `train_wide` demos, layer 3, 25k steps) | test_id (30) | 26.7% | 30% | Pinches about 1 cm beside the loop, then plays out the rest of the task and leaves the bottle leaning 15-30 deg. 3/23 at y < 0, 5/7 at y >= 0 |
  | ACT on `expert_v3_wide_a` + `_b` | test_ood (20) | 20% | 35% | Inside its training range, yet worse than v2. 0/12 at y < 0. 4 tipped bottles |
  | ACT on `expert_v3_wide_a` + `_noise` (94 clean + 83 noisy `train_wide` demos, layer 3, 25k steps) | test_id (30) | 63.3% | 80% | 15/23 at y < 0. Wins 16 poses the clean wide model lost and loses 5 it won. 5 of 11 failures fold the loop but miss the button |
  | ACT on `expert_v3_wide_a` + `_noise` | test_ood (20) | 40% | 50% | Same as v2 (7/20) within noise. 1 tipped bottle |
  | ACT on `expert_v2` + `_wide_a` + `_wide_b` (287 clean demos, layer 3, 25k steps) | test_id (30) | 56.7% | 93.3% | Folds the loop more often than v2, but 12 of 13 failures then miss the button and push the bottle (tilt 15-30 deg, 1-7 cm). 11 of the 13 are at y < 0 |
  | ACT on `expert_v2` + `_wide_a` + `_wide_b` | test_ood (20) | 50% | 75% | Best test_ood so far, but not significantly above v2 (p = .55) |
  | ACT on `expert_v2` + `_wide_a` + `_noise` (194 clean + 83 noisy, layer 3, 25k steps) | test_id (30) | 66.7% | 76.7% | All 10 failures at y < 0. 7 never fold the loop |
  | ACT on `expert_v2` + `_wide_a` + `_noise` | test_ood (20) | 15% | 40% | Below the clean mix (10/20), not significantly (p = .09). 12 of 17 failures never fold the loop |

- **Spreading the same demo budget over the wide range made ACT worse everywhere.** The wide
  box is 4.5 times the volume of the train box. Only 36 of the 187 wide demos fall inside the
  test_id box, against all 100 for `expert_v2`. The wide model's reach targets are as good as
  v2's, but its pinch lands about 1 cm off, which is enough to miss the loop. At this data scale
  ACT needs dense demos where it is tested.
- **Swapping half the clean wide demos for noisy ones helped a lot on test_id, but density is a
  confound.** 19/30 against 8/30 (p = .03), with 10 fewer demos in total. On test_ood
  the gain (8/20 against 4/20) is within noise. However, the noisy collection drew 40 of its
  120 poses inside the test_id box by chance, and failed less often there. So the noisy model
  has 51 demos in that box, against 36 for the clean wide model. Part of the gain may come from
  that.
- **With the density matched, the noise bursts did not help.** `expert_v2` + `_wide_a` + `_noise`
  (277 demos, 151 in the box) against `expert_v2` + `_wide_a` + `_wide_b` (287 clean, 136 in the
  box) scored 20/30 against 17/30 on test_id (p = .6) and 3/20 against 10/20 on test_ood
  (p = .09). Neither difference is significant. The noisy set also leaves fewer demos outside the test_id box (126 against 151,
  and only 75 of its 126 are clean), which may explain part of the test_ood gap. So there is no consistent evidence that
  these noise bursts help ACT, and the earlier gain was most likely the density confound. The
  noisy collection keeps only the 69% of runs the bursts did not break, so it may simply not
  contain the recoveries the method relies on.
- **Adding the wide demos to `expert_v2` traded test_id for test_ood.** test_id fell from 26/30
  to 17/30 (p = .02) and test_ood rose from 7/20 to 10/20 (p = .55). The loop is folded in
  28/30 test_id episodes, so the reach and fold improved, and the losses are almost all at the
  button press. Its validation L1 was still falling at 25k steps (.067, .054, .052, .048, .046
  at 5k-step intervals). It sees each demo about a third as often as v2 did, so it may be
  undertrained. A longer run would test that.

- **Training speed (MPS), 5 cameras at 96 px, batch 32:** on an M5 (10-core GPU), 1.6 steps/s
  with the ResNet trained, 4 steps/s with `--freeze-backbone`, and 0.82 steps/s with the ResNet
  trained and `--feature-layer 3`. On an M5 Pro (16-core GPU), 3.2 steps/s with the ResNet
  trained and `--feature-layer 3`, so 20k steps take about 1.7 h. bf16 autocast and
  channels-last did not help.

## Pieces

| module | what it does |
|---|---|
| `bench/expert.py` | The reference opener moved rigidly into the bottle's frame and driven through the Gym action space. It is privileged (told the pose) |
| `bench/actions.py` | Delta action ↔ absolute tool-target conversion. Mirrors `DetentBottleEnv.step` exactly |
| `bench/poses.py` | Pose ranges and the fixed `train` / `test_id` / `test_ood` splits |
| `bench/rollout.py` | Runs one episode and summarises it. `python -m bench.rollout x y yaw` runs the expert |
| `bench/sweep.py` | Runs the expert over many poses in parallel (reachability check) |
| `bench/collect.py` | Parallel demo collection in Gym format. Keeps only successful episodes |
| `bench/evaluate.py` | Shared evaluation harness for any policy |
| `bench/act/` | ACT baseline: model, in-memory dataset, trainer, policy wrapper |

## Setup

```bash
pip install -e ".[dev,act]"   # [act] adds torch and torchvision, which only bench/act needs
```

Collection and evaluation of the scripted expert run on the core install.

## Collect demonstrations

Use `--settle` for any policy that learns from these demos (see Findings). `data/expert_v2` is 100
settle demos (100/100 kept, 523 MB).

```bash
python -m bench.collect --n 100 --settle --out data/expert_v2         # demos without idle waits (recommended)
python -m bench.collect --n 100 --out data/expert_v1                  # timed demos, as the reference opener runs
python -m bench.collect --n 100 --noise .1 --out data/expert_noise10  # noisy execution, clean labels
```

Each `episode_XXXX.npz` holds `obs/<key>` for every Gym observation key, `action` (T, 14), the
env's Cartesian targets before each action (`target_pos`, `target_rotvec`), privileged
`bottle_qpos`, the expert `stage` label and the `pose`. The default is 96 px images from all
five cameras. That is about 6 MB per episode, because zlib can reuse the previous frame at this
size. At 128 px it is about 34 MB.

With `--noise`, the executed arm motion gets Gaussian noise but the recorded `action` is the
clean expert action (DART-style). This gives recovery data. Noise on every tick keeps the arm
from ever settling, so a `--settle` expert falls back to the clock. Use bursts instead:
`--noise .15 --noise-every 30 --noise-len 6` adds 6-tick bursts about every 30 ticks, roughly
3 mm of target jitter. On 12 wide-range poses the settle expert then succeeds 10/12 (12/12
clean), at about 574 ticks per episode. With `--noise .25` it succeeded 8/12.

`--split train_wide` draws training poses from the whole ±6 cm, ±20 deg box that test_ood lives
in. The test splits stay fixed, but for a model trained on `train_wide`, test_ood is inside its
training range. Its test_ood score therefore measures coverage, not out-of-distribution
generalisation.

```bash
python -m bench.collect --n 100 --settle --split train_wide --out data/expert_v3_wide_a
python -m bench.collect --n 100 --settle --split train_wide --start 100 --out data/expert_v3_wide_b
python -m bench.collect --n 120 --settle --split train_wide --start 200 --noise .15 --noise-every 30 --noise-len 6 --out data/expert_v3_wide_noise
python -m bench.act.train --data data/expert_v3_wide_a data/expert_v3_wide_b --out runs/act/v3_wide --steps 25000 --feature-layer 3 --val-episodes 4
python -m bench.act.train --data data/expert_v3_wide_a data/expert_v3_wide_noise --out runs/act/v3_wide_noise --steps 25000 --feature-layer 3 --val-episodes 4
```

Clean wide-range collection kept 187/200 (94 + 93, about 500 MB per 100). All 13 failures were
at yaw below -14 deg, mostly the loop never being grasped, which matches the expert's weak spot
on test_ood. So `train_wide` has few demos there. The noisy collection kept 83/120 (69%, 689
MB), lower than the 12-pose check suggested. Its 37 failures are spread over the whole box: 31
ran out of time without folding the loop and 6 tipped the bottle. Only the runs where the bursts
did no lasting harm are kept.

## Evaluate a policy

Any policy is a factory `make_policy(pose, **kwargs)` that returns `policy(obs, sim) -> action`.
Learned policies must ignore `pose` and `sim`. They may set `cameras`, `image_size` and
`reset(sim)` attributes.

```bash
python -m bench.evaluate --policy bench.rollout:expert_policy --split test_id test_ood --name expert
python -m bench.evaluate --policy bench.act.policy:make_policy --kwargs '{"ckpt": "runs/act/v1/last.pt"}' --split test_id test_ood --name act_v1
```

Results go to `runs/eval/<name>/` (`<split>.jsonl` per episode and `summary.json`). Reported:
success rate, how often the loop was folded past the detent (>150 deg), how often the lid opened
(>60 deg), mean steps for successful episodes, and termination reasons.

## Train ACT

```bash
python -m bench.act.train --data data/expert_v1 --out runs/act/v1 --steps 20000
python -m bench.act.train --data data/expert_v1 --out runs/act/v1 --steps 20000 --resume  # after an interruption
```

`last.pt` is written every 1000 steps with the model, optimiser state and step, so `--resume`
loses at most 1000 steps. Pass the same arguments as the original run. On a laptop, wrap the
command in `caffeinate -i` so macOS does not sleep mid-run.

The model predicts 50-tick chunks of absolute tool targets per arm (position, 6-D rotation,
jaw). At run time it tracks the env's targets itself from the known reset values, ensembles
overlapping chunks (ACT temporal ensembling) and converts the result to a delta action.
