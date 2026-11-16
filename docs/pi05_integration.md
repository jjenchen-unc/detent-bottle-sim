# Optional π₀.₅ integration

This package connects the simulator to a separately hosted OpenPI DROID policy. It does not
install OpenPI, download model weights, or change the simulator's default physics.

## Control path

The DROID checkpoint produces a 15-step, 8-value single-arm joint-action chunk. The simulator
expects Cartesian target deltas for both arms. `integration_pi05.actions` resamples the DROID
chunk at the simulator's 20 Hz control rate, maps the seven joint values through forward
kinematics to the left tool target, and maps the gripper value to jaw closure. The existing
script grips and stabilizes the bottle with the right arm. By default, learned control starts
when the scripted sequence reaches the left-arm task. For capability evaluation, the optional
`after_pinch` handoff runs the script through the loop pinch first.

This is hybrid control: the learned policy controls one arm while a known script stabilizes the
bottle. Results with this setup do not establish fully learned bimanual control. The adapter
preserves the simulator's Cartesian controller and physics; it does not directly command
simulator joint state.

The runtime client isolates connection and inference calls in a disposable process with separate
deadlines. When controlled sampling is enabled, the client verifies a stationary A/B/A request,
explicit inference-noise provenance, policy identity, and (when supplied) server-instance
identity before robot motion. Requests, actions, trajectories, transport events, and software
metadata are written to a new run directory.

## Install and run

Install the simulator and development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Prepare a separate OpenPI checkout at a recorded revision and install its matching
`openpi-client` package in the simulator environment. Model weights and OpenPI are not bundled.
Launch the policy server in a GPU allocation managed by your compute environment; allocation
scripts and storage paths are deployment-specific and are not part of this simulator package.

With a compatible policy server reachable at the configured host and port, run a scripted
baseline or a hybrid episode:

```bash
python -m integration_pi05.evaluate --controller scripted --seed 0 --out runs/scripted-0
python -m integration_pi05.evaluate --controller hybrid --config configs/pi05/default.json \
  --seed 0 --out runs/hybrid-0
```

Output paths must be new. The default configuration uses the pinned public
`pi05_droid` checkpoint contract and agent-view/left-wrist images. `configs/pi05/frontview.json`
shows how to select front-view and left-wrist images with explicit inference-noise sampling.
For a trained candidate, supply a configuration whose checkpoint URI and expected-policy
contract exactly match the serving process. `mechanism_response` may carry a fixed normalized
response descriptor when the selected checkpoint was trained with that prompt field. The
generic evaluator does not perform a physical probe or estimate dynamics.

## Scope and interpretation

`integration_pi05.capability` audits a LoRA candidate in one full-horizon, post-pinch, no-probe
episode and separates pipeline/setup failures from valid policy outcomes. One successful episode establishes
feasibility only; it is not a success-rate estimate. For repeated trials, use independent task
seeds or a prespecified paired design and report transport/evaluator errors separately from
task failures.

The π₀.₅ path is an optional integration example, not a replacement for the simulator's scripted
baseline or other policy interfaces. Keep policy-specific dependencies and hardware launch
requirements outside the simulator's core installation.
