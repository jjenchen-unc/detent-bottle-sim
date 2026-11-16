# detent-bottle-sim

A bimanual MuJoCo manipulation task: two Panda arms with parallel-jaw grippers must open a
free-standing flip-top bottle whose lid is held by a rocker latch behind a detented safety loop.
Packaged as a Gymnasium env (`DetentBottle-v0`) for RL / imitation learning, plus a scripted
reference opener and an LLM tool-calling ("agentic") runner.

## The task

- **Bottle**: a 0.35 kg free body standing on the table near the origin. Nothing holds it in place;
  push it too hard and it slides or tips over.
- **Mechanism**: the lid sits on a spring hinge that wants to open. A rocker button on the front
  hooks the lid shut. A U-shaped safety loop is hinged in front of the button and blocks it while
  up. The loop has a **detent**: below 30 deg it is pulled back up (closed), above 150 deg it is
  pulled down (open), and in between it is just damped. So the loop has to be folded down past
  150 deg before the button can be pressed. Then the button releases the hook and the spring
  opens the lid.
- **Only robot joints are actuated** (14 arm torque motors and 4 finger position servos). The
  bottle has no actuators, welds or scripted state writes. Everything happens through contact.
  The detent is a passive joint-torque law on the loop hinge, not a control input.
- **Success** (same as the original console task): lid opened > 60 deg while the bottle stays
  upright (z-axis tilt < ~25 deg) for 0.4 s in a row (8 control ticks).
- **Episode end**: success, bottle falls (`z < 0.76`), wrist force > 100 N for 0.2 s,
  numerical instability, or 2400 control ticks (120 s simulated time).

Physics runs at dt = 5e-5 s (1000 substeps per 0.05 s control tick), so the small latch contacts
stay stable. Expect about 8 control ticks/s of wall time on one CPU core before rendering.

## Install

Requires Python >= 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest; add [agent] for the LLM runner, [video] for mp4 output
export MUJOCO_GL=egl             # headless GPU rendering; use MUJOCO_GL=osmesa on CPU-only machines
pytest -q
```

## Quick start

```python
import gymnasium as gym
import detent_bottle  # registers DetentBottle-v0

env = gym.make('DetentBottle-v0', image_size=128)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Constructor kwargs: `image_size` (default 128), `cameras` (subset of the five below; `()`
turns off rendering), `max_steps` (2400), `bottle_pose=(x, y, yaw_deg)` (moves the bottle away
from the default `(0, 0, 0)`; this rebuilds the model).

### Action: `Box(-1, 1, (14,))`

Seven values per arm, left arm first:

| index | meaning | scale |
|---|---|---|
| 0-2 | tool target translation dx, dy, dz (world frame) | 0.02 m per unit |
| 3-5 | tool target rotation, axis-angle in world frame, pre-multiplied | 0.10 rad per unit |
| 6 | jaw: -1 = fully open (0.09 m gap), +1 = closed (0 m), linear in between | |

The deltas move a persistent Cartesian **target** per arm. The target is clipped to the
workspace X[-.30,.35], Y[-.40,.40], Z[.81,1.40] m. The original console actuates the robot the
same way, and the conversion below is unchanged from it. On each 0.05 s control tick, the
controller moves a waypoint at most 8 mm and 0.08 rad from the **measured** tool pose toward the
target. Damped-least-squares IK then gives joint targets. Joint torques come from a PD law
(kp 450, kd 40) plus gravity/Coriolis compensation, clipped to the Panda limits (80/12 N m). The
jaw is a position servo with a 12 N force limit. The "tool" point is midway between the distal
fingertip ends (site `{left,right}_tip`).

### Observation: `Dict`

| key | shape | content |
|---|---|---|
| `agentview`, `frontview`, `topview` | (S, S, 3) uint8 | fixed free cameras (wide table view; low close-up from the latch side; close-up from above) |
| `left_wrist`, `right_wrist` | (S, S, 3) uint8 | cameras rigidly mounted on each hand, 70 mm along tool -Y, aimed past the fingertips |
| `joint_pos`, `joint_vel` | (14,) | arm joints, left then right |
| `gripper_qpos` | (4,) | finger slide positions (each finger 0-0.045 m) |
| `eef_pos`, `eef_quat` | (6,), (8,) | measured tool position and orientation (wxyz) per arm |
| `wrist_ft` | (12,) | wrist force (3) + torque (3) per arm, sensor frame |
| `finger_force` | (4,) | fingertip normal (touch) force per finger |

Images are rendered without robot marker sites or overlays. **No object state is in the
observation.**

### Privileged state (`info`, for evaluation / asymmetric critics only)

`info['privileged']` holds `bottle_qpos` (10 values: bottle free joint xyz + quat wxyz, then lid,
button and loop hinge angles in rad), `bottle_pos`, and `lid_deg` / `loop_deg` / `button_deg`.
`info` also contains `success`, `termination_reason`, `open_time` and `step`.

### Reward and termination

Sparse: `reward = 1` on the tick that success first holds, else 0. `terminated` is true on
success, `bottle_fell`, `wrist_force_limit` or `numerical_instability`. `truncated` is true at the
2400-tick limit.

## Scripted reference opener

`detent_bottle/policy.py` is a fixed known-geometry sequence of tool targets. It uses no
observations. The right hand grips the bottle body, the left hand pinches the safety loop and
folds it down through the detent, and then the left fingertips press the rocker button.

```bash
MUJOCO_GL=egl python scripts/scripted_demo.py --seed 42
# ... latch=detent success=True reason=success steps=715 sim_s=35.75
python scripts/scripted_demo.py --save demo.npz          # per-tick qpos / targets / jaw targets
python scripts/scripted_demo.py --video demo.mp4         # needs pip install -e ".[video]"
```

The task is deterministic, so the opener produces the same run for any seed. Use it as a sanity
check and as one source of demonstrations.

## Agentic runner (`agent/`)

This is the "v3" LLM runner. A tool-calling model sees the RGB views and robot proprioception and
acts through tools: `move_arm`, `move_both_arms`, `wait`, `touch` (force probe), `locate` (RGB-D
back-projection of pixels), `check_reach`, `find_reach`, `observe`, `tare`, `request_views` and
`update_notes`. The runner adds a camera router, persistent notes, context compaction every 5
turns, strategy reviews and a USD budget stop. Note that this runner frames its fixed views on
the workspace centre and uses outboard "boom" wrist cameras, as the original v3 runner did. These
views differ from the Gym env's cameras. Physics is identical.

```bash
pip install -e ".[agent]"
python -m agent.run --dry                      # fake model, zero API calls; smoke test
export OPENAI_API_KEY=...                      # read from the environment only
export OPENAI_MODEL=gpt-6-astra                # optional; OPENAI_BASE_URL is also honoured
python -m agent.run --live --budget 10         # spends real money, capped by --budget (USD)
python -m agent.run --resume runs/<dir>        # continue a crashed run
```

Run it from the repo root. Each run writes a directory under `runs/` (ignored by git) containing
the prompts, every request and response, actions, `trajectory.npz`, cost ledger and
`final.json`. Options: `--sensing none|locate|touch|both` removes the probing tools.
`--privileged` is an evaluation-only ablation that sends true object poses every turn. The
price table in `agent/costs.py` is an estimate and is not your account's billing.

## Known limits

- **MuJoCo is pinned to 3.4.0**, the version the reference console ran on. This package was
  checked against that console: for the same action sequence, qpos (bottle and all robot joints)
  matched exactly (max error 0.0) over the full 715-tick scripted opening and over 250 ticks of
  random actions. Other MuJoCo versions may change the contact results.
- Offscreen rendering needs `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, slow).
- The simulation is slow (dt 5e-5) and there is no built-in domain randomisation. `reset(seed)`
  gives the same initial state every time. Use `bottle_pose` if you want pose variation.
- The task, physics constants and success rule are copied from the console and have not been
  tuned for learning. With a sparse reward, exploration is hard. Demonstrations from the scripted
  opener or shaping from `info['privileged']` will probably be needed.

## Credits and licences

Author: Jr-Jen Chen. MIT licence (see `LICENSE`).
The Panda arm model and meshes in `detent_bottle/assets/panda/` come from robosuite 1.4.0
(MIT, Stanford Vision and Learning Lab and UT Robot Perception and Learning Lab). See
`detent_bottle/assets/panda/LICENSE.robosuite`.

## Optional π₀.₅ integration

See [integration setup and limitations](docs/pi05_integration.md) for the pretrained DROID client, hybrid controller, evaluation CLI, and policy-server contract.
