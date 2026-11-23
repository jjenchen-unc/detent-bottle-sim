"""Build the paper's figures, data tables and numbers from data/ and runs/.

python paper/make_figures.py            # everything (renders need ~3 min, mostly the ACT rollout)
python paper/make_figures.py --no-render  # skip the MuJoCo renders, reuse the PNGs already there

Writes paper/figures/*.png (renders), paper/figures/plotdata/*.dat (pgfplots tables) and
paper/figures/stats.tex (\\newcommand macros, so every number in main.tex comes from the runs).
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench.poses import OOD_RANGE, TRAIN_RANGE, split   # noqa: E402

FIG = os.path.join(ROOT, 'paper', 'figures')
DAT = os.path.join(FIG, 'plotdata')
MODELS = [   # eval run, label in figures, macro prefix in stats.tex, demo directories (None for the privileged expert)
    ('expert', 'Expert timed', 'ExpertTimed', None),   # table only
    ('expert_settle', 'Expert', 'Expert', None),
    ('act_full_l3', 'ACT timed', 'Timed', ['expert_v1']),
    ('act_v2_l3', 'ACT v2', 'Vtwo', ['expert_v2']),
    ('act_v3_wide', 'A', 'A', ['expert_v3_wide_a', 'expert_v3_wide_b']),
    ('act_v3_wide_noise', 'B', 'B', ['expert_v3_wide_a', 'expert_v3_wide_noise']),
    ('act_v3_mix', 'C', 'C', ['expert_v2', 'expert_v3_wide_a', 'expert_v3_wide_b']),
    ('act_v3_mix_noise', 'D', 'D', ['expert_v2', 'expert_v3_wide_a', 'expert_v3_wide_noise']),
]
JOINT_TOL, GRIP_TOL, IDLE_MIN = 1e-4, 1e-5, 10   # same stillness test as bench/expert.py; idle = still for 10+ ticks
LOOP_MOVED_DEG = 20   # the loop rests below 5 deg, so past 20 deg it was pinched and pushed


def path(*p):
    return os.path.join(ROOT, *p)


def episodes(run, s):
    return [json.loads(line) for line in open(path('runs', 'eval', run, f'{s}.jsonl'))]


def demo_files(dirs):
    return [f for d in dirs for f in sorted(glob.glob(path('data', d, 'episode_*.npz')))]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def pct(k, n):
    return f'{100 * k / n:.1f}'.rstrip('0').rstrip('.')


def in_box(P, half=TRAIN_RANGE):
    P = np.atleast_2d(P)
    return np.all(np.abs(P) <= half, axis=1)


def write_dat(name, header, rows):
    with open(os.path.join(DAT, name), 'w') as f:
        f.write(' '.join(header) + '\n')
        for r in rows: f.write(' '.join(str(v) for v in r) + '\n')


def idle_runs(ep):
    """Ticks where the robot is completely still for IDLE_MIN+ ticks in a row, as (start, end) pairs."""
    q = np.diff(ep['obs/joint_pos'], axis=0)
    g = np.diff(ep['obs/gripper_qpos'], axis=0)
    still = (np.abs(q).max(1) < JOINT_TOL) & (np.abs(g).max(1) < GRIP_TOL)
    runs, start = [], None
    for t, s in enumerate(np.r_[still, False]):
        if s and start is None: start = t
        if not s and start is not None:
            if t - start >= IDLE_MIN: runs.append((start + 1, t + 1))
            start = None
    return runs


def trace(ep):
    """Per-tick tool speed (mm/s, faster arm) and the ticks where either gripper is told to close."""
    eef = ep['obs/eef_pos'].reshape(len(ep['action']), 2, 3)
    speed = np.r_[0, 1000 * 20 * np.linalg.norm(np.diff(eef, axis=0), axis=-1).max(1)]   # 20 control ticks per second
    jaw = ep['action'][:, [6, 13]]
    closes = [t for t in range(1, len(jaw)) if np.any((jaw[t] > jaw[t - 1] + .3) & (jaw[t] > 0))]
    return speed, closes


def idle_figure(stats):
    timed, settle = (dict(np.load(path('data', d, 'episode_0000.npz'))) for d in ('expert_v1', 'expert_v2'))
    assert np.allclose(timed['pose'], settle['pose'])
    for name, ep in (('timed', timed), ('settle', settle)):
        speed, closes = trace(ep)
        write_dat(f'trace_{name}.dat', ['t', 'speed'], [(t, round(float(v), 2)) for t, v in enumerate(speed)])
        with open(os.path.join(DAT, f'trace_{name}_marks.tex'), 'w') as f:
            for a, b in idle_runs(ep):
                f.write(f'\\fill[idle] (axis cs:{a},0) rectangle (axis cs:{b},\\ymaxtrace);\n')
            for t in closes:
                f.write(f'\\draw[grip] (axis cs:{t},0) -- (axis cs:{t},\\ymaxtrace);\n')
    stats['TracePose'] = f"x = {100 * timed['pose'][0]:+.1f}\\,cm, y = {100 * timed['pose'][1]:+.1f}\\,cm, yaw = {timed['pose'][2]:+.1f}$^\\circ$"
    stats['TraceTimedTicks'], stats['TraceSettleTicks'] = len(timed['action']), len(settle['action'])
    # the whole timed dataset
    idle, total, longest, lens = 0, 0, [], []
    for f in demo_files(['expert_v1']):
        ep = np.load(f)
        r = idle_runs(ep)
        idle += sum(b - a for a, b in r); total += len(ep['action']); lens.append(len(ep['action']))
        longest += [b - a for a, b in r]
    stats['TimedIdlePct'] = f'{100 * idle / total:.0f}'
    stats['TimedIdleRuns'] = f'{len(longest) / len(lens):.1f}'
    stats['TimedIdleMedian'], stats['TimedIdleMax'] = f'{np.median(longest):.0f}', max(longest)
    stats['TimedLen'] = f'{np.mean(lens):.0f}'
    stats['SettleLen'] = f"{np.mean([len(np.load(f)['action']) for f in demo_files(['expert_v2'])]):.0f}"


def pose_figure(stats):
    cm = lambda P: [(round(100 * p[0], 2), round(100 * p[1], 2), round(float(p[2]), 1)) for p in P]
    narrow = np.array([np.load(f)['pose'] for f in demo_files(['expert_v2'])])
    wide = np.array([np.load(f)['pose'] for f in demo_files(['expert_v3_wide_a', 'expert_v3_wide_b'])])
    for name, P in (('train', narrow), ('train_wide', wide), ('test_id', split('test_id')), ('test_ood', split('test_ood'))):
        write_dat(f'poses_{name}.dat', ['x', 'y', 'yaw'], cm(P))
    stats['TrainX'], stats['TrainYaw'] = f'{100 * TRAIN_RANGE[0]:.0f}', f'{TRAIN_RANGE[2]:.0f}'
    stats['OodX'], stats['OodYaw'] = f'{100 * OOD_RANGE[0]:.0f}', f'{OOD_RANGE[2]:.0f}'
    stats['WideVolume'] = f'{np.prod(OOD_RANGE / TRAIN_RANGE):.1f}'


def results(stats):
    rows, table = [], []
    for i, (run, label, key, dirs) in enumerate(MODELS):
        summary = json.load(open(path('runs', 'eval', run, 'summary.json')))['results']
        row = [i, label.replace(' ', '~')]
        line = dict(run=run, label=label, key=key)
        for s in ('test_id', 'test_ood'):
            eps = episodes(run, s)
            k, n = sum(e['success'] for e in eps), len(eps)
            lo, hi = wilson(k, n)
            row += [round(100 * k / n, 1), round(100 * (k / n - lo), 1), round(100 * (hi - k / n), 1)]
            line[s] = (k, n, summary[s]['loop_folded_rate'], sum(e['max_loop_deg'] > LOOP_MOVED_DEG for e in eps) / n)
        if dirs:
            P = np.array([np.load(f)['pose'] for f in demo_files(dirs)])
            line['demos'], line['in_box'] = len(P), int(in_box(P).sum())
        table.append(line)
        if run != 'expert': rows.append([len(rows)] + row[1:])
    write_dat('results.dat', ['i', 'label', 'id', 'idminus', 'idplus', 'ood', 'oodminus', 'oodplus'], rows)
    # LaTeX table body
    names = dict(expert='Scripted expert, timed (privileged pose)', expert_settle='Scripted expert, settle (privileged pose)', act_full_l3='ACT on timed demos (\\texttt{expert\\_v1})',
                 act_v2_l3='\\textbf{ACT v2} (settle demos, \\texttt{expert\\_v2})', act_v3_wide='A: wide only',
                 act_v3_wide_noise='B: wide + noisy wide', act_v3_mix='C: v2 + wide', act_v3_mix_noise='D: v2 + wide + noisy wide')
    with open(os.path.join(FIG, 'results_table.tex'), 'w') as f:
        for line in table:
            demos = f"{line['demos']} ({line['in_box']})" if 'demos' in line else '--'
            bold = line['run'] == 'act_v2_l3'
            cells = []
            for s in ('test_id', 'test_ood'):
                k, n, fold, moved = line[s]
                ok = f'{100 * k / n:.1f}'
                if bold and s == 'test_id': ok = '\\textbf{' + ok + '}'
                cells.append(f'{100 * moved:.0f} & {100 * fold:.0f} & {ok}')
            f.write(f"{names[line['run']]} & {demos} & {' & '.join(cells)} \\\\\n")
            if line['run'] in ('expert_settle', 'act_v2_l3'): f.write('\\midrule\n')
        f.write('\\bottomrule\n')   # here, not in main.tex: a rule right after \\input breaks inside tabular
    by = {line['run']: line for line in table}
    for line in table:
        key = line['key']
        for s, tag in (('test_id', 'Id'), ('test_ood', 'Ood')):
            k, n, _, moved = line[s]
            stats[f'{key}{tag}'] = pct(k, n)
            stats[f'{key}{tag}K'] = f'{k}/{n}'
            stats[f'{key}{tag}MovedK'] = f'{round(moved * n)}/{n}'
        if 'demos' in line: stats[f'{key}Demos'], stats[f'{key}InBox'] = line['demos'], line['in_box']

    def p(a, b, s):   # exact McNemar: every policy runs the same poses, so compare them pose by pose
        A, B = ({tuple(np.round(e['pose'], 4)): e['success'] for e in episodes(r, s)} for r in (a, b))
        assert A.keys() == B.keys(), (a, b, s)
        only_a, only_b = sum(A[q] and not B[q] for q in A), sum(B[q] and not A[q] for q in A)
        v = binomtest(min(only_a, only_b), only_a + only_b).pvalue if only_a + only_b else 1.
        return '< .001' if v < .001 else '= ' + f'{v:.2f}'.lstrip('0')
    stats.update(PwideNoiseId=p('act_v3_wide', 'act_v3_wide_noise', 'test_id'), PwideNoiseOod=p('act_v3_wide', 'act_v3_wide_noise', 'test_ood'),
                 PmixId=p('act_v2_l3', 'act_v3_mix', 'test_id'), PmixOod=p('act_v2_l3', 'act_v3_mix', 'test_ood'),
                 PmixNoiseId=p('act_v3_mix', 'act_v3_mix_noise', 'test_id'), PmixNoiseOod=p('act_v3_mix', 'act_v3_mix_noise', 'test_ood'),
                 PwideId=p('act_v2_l3', 'act_v3_wide', 'test_id'))
    stats['DensityRatio'] = f"{by['act_v2_l3']['in_box'] / by['act_v3_wide']['in_box']:.1f}"


def outcome_maps():
    for run in ('act_v2_l3', 'act_v3_wide', 'act_v3_mix'):
        rows = episodes(run, 'test_id') + episodes(run, 'test_ood')
        for ok in (True, False):
            write_dat(f'outcome_{run}_{"ok" if ok else "fail"}.dat', ['x', 'y'],
                      [(round(100 * r['pose'][0], 2), round(100 * r['pose'][1], 2)) for r in rows if r['success'] == ok])


def collection_stats(stats):
    for d, key in (('expert_v1', 'CollV'), ('expert_v3_wide_a', 'CollWideA'), ('expert_v3_wide_b', 'CollWideB'), ('expert_v3_wide_noise', 'CollNoise')):
        rows = [json.loads(line) for line in open(path('data', d, 'manifest.jsonl'))]
        stats[key] = f"{sum(r['success'] for r in rows)}/{len(rows)}"


def renders():
    """Stage strip and camera inputs of the settle expert at the default pose, and the pinch of the
    settle expert next to that of model A at a test_id pose where A fails."""
    import imageio.v2 as imageio
    from detent_bottle.gym_env import CAMERAS, DetentBottleEnv
    from bench.act.policy import ACTPolicy
    from bench.rollout import expert_policy

    def run(policy, pose, until, grab, cams=(), size=96):
        env = DetentBottleEnv(image_size=size, cameras=cams, max_steps=2400, bottle_pose=tuple(pose))
        obs, info = env.reset(seed=0)
        if hasattr(policy, 'reset'): policy.reset(env.sim)
        shots, t = {}, 0
        while True:
            action = policy(obs, env.sim)
            for key, fn in grab.items():
                if key not in shots and fn(t, policy, action, obs): shots[key] = (t, {c: env.sim.render(c, 400) for c in ('frontview', 'agentview')}, obs)
            obs, _, term, trunc, info = env.step(action)
            t += 1
            if term or trunc or until(t, info): break
        env.close()
        return shots, info

    stage_end = {}
    expert = expert_policy((0., 0., 0.), settle=True)
    labels = []
    probe = lambda t, pol, a, o: labels.append(pol.stage) or False
    run(expert, (0., 0., 0.), lambda t, i: False, dict(probe=probe))
    for t in range(len(labels) - 1):
        if labels[t + 1] != labels[t]: stage_end[labels[t]] = t
    picks = [('start', 0), ('grip', stage_end['1 / Grip and stabilize the bottle body']), ('pinch', stage_end['2 / Pinch the safety loop']),
             ('fold', stage_end['3 / Fold the safety loop down through contact']), ('press', stage_end['4 / Approach the rocker button']),
             ('open', len(labels) - 1)]
    expert = expert_policy((0., 0., 0.), settle=True)
    shots, info = run(expert, (0., 0., 0.), lambda t, i: False,
                      {name: (lambda tt: lambda t, pol, a, o: t == tt)(tick) for name, tick in picks}, cams=CAMERAS)
    assert info['success']
    for name, _ in picks:
        imageio.imwrite(os.path.join(FIG, f'stage_{name}.png'), shots[name][1]['frontview'])
    imageio.imwrite(os.path.join(FIG, 'overview.png'), shots['start'][1]['agentview'])
    for c in CAMERAS:   # what the policy sees, 96 px, upscaled without smoothing
        imageio.imwrite(os.path.join(FIG, f'input_{c}.png'), np.kron(shots['pinch'][2][c], np.ones((4, 4, 1), np.uint8)))

    # the pinch: first tick the left jaw is commanded closed, plus 12 ticks for the fingers to close
    pose = next(r['pose'] for r in episodes('act_v3_wide', 'test_id') if abs(r['pose'][1] + .0098) < 1e-4)
    closed = lambda: {'t': None}
    for name, policy, cams in (('expert', expert_policy(pose, settle=True), ()),
                               ('model_a', ACTPolicy(path('runs', 'act', 'v3_wide', 'last.pt'), threads=4), None)):
        mark = closed()

        def grab(t, pol, a, o, mark=mark):
            if mark['t'] is None and a[6] > .8: mark['t'] = t
            return mark['t'] is not None and t == mark['t'] + 12
        shots, _ = run(policy, pose, lambda t, i, mark=mark: mark['t'] is not None and t > mark['t'] + 13, dict(pinch=grab),
                       cams=cams if cams is not None else policy.cameras)
        imageio.imwrite(os.path.join(FIG, f'pinch_{name}.png'), shots['pinch'][1]['frontview'][:300, 80:340])   # crop to the loop
    return pose


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--no-render', action='store_true')
    args = p.parse_args()
    os.makedirs(DAT, exist_ok=True)
    stats = {}
    idle_figure(stats); pose_figure(stats); results(stats); outcome_maps(); collection_stats(stats)
    if not args.no_render:
        pose = renders()
        stats['PinchPose'] = f"x = {100 * pose[0]:+.1f}\\,cm, y = {100 * pose[1]:+.1f}\\,cm, yaw = {pose[2]:+.1f}$^\\circ$"
        json.dump(dict(PinchPose=stats['PinchPose']), open(os.path.join(DAT, 'render_stats.json'), 'w'))
    else:
        stats.update(json.load(open(os.path.join(DAT, 'render_stats.json'))))
    with open(os.path.join(FIG, 'stats.tex'), 'w') as f:
        f.write('% generated by paper/make_figures.py\n')
        for k, v in sorted(stats.items()): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    print('\n'.join(f'{k} = {v}' for k, v in sorted(stats.items())))
