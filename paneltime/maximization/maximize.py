"""Run one or several local optimizer starts and select the final fit."""

import numpy as np

from .dfpmax import dfpmax


def maximize(panel, args):
    """Fit locally; multistart prefers the best finite, converged attempt."""
    if not panel.options.multistart:
        return dfpmax(args.args_v, panel)

    starts = get_multistart_directions(panel, args)
    results = [dfpmax(start, panel) for start in starts]
    finite = [i for i, result in enumerate(results) if np.isfinite(result['f'])]
    converged = [i for i in finite if results[i]['conv'] in (1, 2)]
    candidates = converged or finite
    if not candidates:
        raise RuntimeError("No multistart fit returned a finite log-likelihood")

    best = max(candidates, key=lambda i: results[i]['f'])
    result = results[best]
    result['multistart_results'] = [
        {'start': i, 'log_likelihood': float(attempt['f']),
         'convergence_code': int(attempt['conv']), 'iterations': int(attempt['its']),
         'selected': i == best}
        for i, attempt in enumerate(results)
    ]
    return result


def get_multistart_directions(panel, args):
    """Keep the original start, then explore neutral and persistent dynamics."""
    starts = [np.array(args.args_v, dtype=float, copy=True)]
    if panel.args.initial_user_defined:
        return starts
    for rho, ma, gamma, psi in ((0.0, 0.0, 0.7, 0.15), (0.5, -0.3, 0.85, 0.05)):
        start = starts[0].copy()
        for group, first in (('rho', rho), ('lambda', ma)):
            positions = list(args.positions[group])
            if positions:
                start[positions] = 0.0
                start[positions[0]] = first
        if not panel.options.EGARCH:
            for group, total in (('gamma', gamma), ('psi', psi)):
                positions = list(args.positions[group])
                if positions:
                    weights = 0.5 ** np.arange(len(positions))
                    start[positions] = total * weights / weights.sum()
        if not any(np.array_equal(start, existing) for existing in starts):
            starts.append(start)
    return starts
