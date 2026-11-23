"""Compare evaluation results: python -m bench.report [runs/eval/*]"""
import glob
import json
import os
import sys

COLUMNS = ('success_rate', 'loop_folded_rate', 'lid_opened_rate', 'mean_success_steps', 'episodes')


def table(dirs):
    rows = []
    for d in dirs:
        path = os.path.join(d, 'summary.json')
        if not os.path.exists(path): continue
        with open(path) as f: meta = json.load(f)
        for split, r in meta['results'].items():
            reasons = ', '.join(f'{k} {v}' for k, v in sorted(r['reasons'].items(), key=lambda kv: -kv[1]))
            rows.append([os.path.basename(d), split] + [str(r[c]) for c in COLUMNS] + [reasons])
    header = ['run', 'split', 'success', 'loop folded', 'lid opened', 'steps (succ.)', 'n', 'termination reasons']
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header, ['-' * w for w in widths]] + rows:
        print('  '.join(str(x).ljust(w) for x, w in zip(row, widths)))


if __name__ == '__main__':
    table(sys.argv[1:] or sorted(glob.glob('runs/eval/*')))
