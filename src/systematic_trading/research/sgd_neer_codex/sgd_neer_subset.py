"""Select SGD crosses at existing index entries; preserve their USD hedges.

Attribution is additive in log levels. Selection uses only contemporaneous
deviations and quotes. Component weights and notionals are frozen until the
existing index exit. Forward points are intentionally excluded.
"""
import numpy as np
import pandas as pd

from research.sgd_neer import normalised_weights
from research.sgd_neer_codex.sgd_neer_multiday import MinuteAccountant


def cross_structure(pairs):
    """SGD/X log-return loadings on quoted pairs, and normalized index weights."""
    if 'USDSGD' not in pairs:
        raise ValueError('USDSGD is required for synthetic SGD crosses.')
    currencies = ['USD' if p == 'USDSGD' else p[3:] if p.startswith('USD') else p[:3]
                  for p in pairs]
    matrix = pd.DataFrame(0., index=currencies, columns=pairs)
    for ccy, pair in zip(currencies, pairs):
        matrix.loc[ccy, 'USDSGD'] = -1.
        if pair != 'USDSGD':
            matrix.loc[ccy, pair] = 1. if pair.startswith('USD') else -1.
    weights = pd.Series(list(normalised_weights(list(pairs)).values()), index=currencies)
    return matrix, weights


def component_deviations(levels, kind, horizon_hours):
    """Unstandardized components whose weighted sum is the index deviation."""
    n = int(horizon_hours*60)
    if kind == 'equilibrium':
        reference = levels.ewm(halflife=n, min_periods=max(20*1440, 2*n)).mean().shift(1)
        return levels-reference
    if kind == 'return':
        moves = levels.diff(n)
        reference = moves.ewm(halflife=63*1440, min_periods=40*1440).mean().shift(1)
        return moves-reference
    raise ValueError(f'Unknown signal kind: {kind}')


def covariance_at_entries(levels, entries):
    """Prior EWM covariance of actual 24-market-hour cross returns.

    Same 20-day half life / 40-day warm-up as the index risk estimator. Missing
    endpoints stay missing; overlapping samples are not independent evidence.
    Materialize only entry-time matrices, not the full minute covariance panel.
    """
    moves = levels.diff(1440)
    out = np.empty((len(entries), len(levels.columns), len(levels.columns)))
    for i, left in enumerate(levels.columns):
        estimate = moves[left].ewm(halflife=20*1440, min_periods=40*1440)
        for j in range(i+1):
            values = estimate.cov(moves.iloc[:, j]).shift(1).reindex(entries).to_numpy()
            out[:, i, j] = out[:, j, i] = values
    return out


def select_crosses(deviation, weights, spreads, rule):
    """Return a Boolean subset selected from positive signed contributions.

    cost_top2 requires half the signed cross deviation to exceed 1.5 synthetic
    full spreads; rank eligible contributions per weighted spread cost. This
    is a heuristic edge/cost filter, not a fitted prediction of cross reversion.
    """
    deviation, weights, spreads = map(np.asarray, (deviation, weights, spreads))
    if rule == 'full':
        return np.ones(len(weights), dtype=bool)
    direction = np.sign(weights@deviation)
    contribution = direction*weights*deviation
    eligible = np.isfinite(contribution) & (contribution > 0)
    if rule == 'cost_top2':
        eligible &= (.5*direction*deviation > 1.5*spreads) & (spreads > 0)
        score = contribution/(weights*spreads)
    else:
        score = contribution
    ranked = np.flatnonzero(eligible)
    ranked = ranked[np.argsort(-score[ranked], kind='stable')]
    chosen = np.zeros(len(weights), dtype=bool)
    if rule == 'aligned':
        chosen[ranked] = True
    elif rule in ('top1', 'top2', 'cost_top2'):
        chosen[ranked[:1 if rule == 'top1' else 2]] = True
    elif rule == 'coverage80':
        if len(ranked):
            count = np.searchsorted(np.cumsum(contribution[ranked]), .8*contribution[ranked].sum())+1
            chosen[ranked[:count]] = True
    else:
        raise ValueError(f'Unknown subset rule: {rule}')
    return chosen


def subset_targets(signal, deviations, spreads, covariances, matrix, weights, rule, sizing):
    """Freeze selected SGD-cross positions between the original index events.

    retained: keep original constituent weights, decreasing total SGD exposure.
    normalized: selected weights sum to one, retaining original SGD exposure.
    risk_matched: normalized weights and entry size rescaled to the full basket's
    prior covariance risk, capped at three aggregate SGD-cross units.
    """
    if sizing not in ('retained', 'normalized', 'risk_matched'):
        raise ValueError(f'Unknown sizing mode: {sizing}')
    entries = signal[signal != 0].index
    if not entries.equals(deviations.index) or not entries.equals(spreads.index):
        raise ValueError('Entry features must match the nonzero signal events exactly.')
    output = pd.DataFrame(0., index=signal.index, columns=matrix.columns)
    decisions = []
    w = weights.to_numpy()
    for k, (timestamp, q) in enumerate(signal[signal != 0].items()):
        d, cost = deviations.iloc[k].to_numpy(), spreads.iloc[k].to_numpy()
        if not np.isfinite(d).all() or not np.isfinite(cost).all():
            raise ValueError('Entry attribution and costs must be finite.')
        chosen = select_crosses(d, w, cost, rule)
        selected = w*chosen
        mass = selected.sum()
        actual_q, exante_ratio, cap_hit = q, np.nan, False
        if mass:
            if sizing != 'retained':
                selected /= mass
            cov = covariances[k]
            full_var, subset_var = w@cov@w, selected@cov@selected
            if not np.isfinite(cov).all() or min(full_var, subset_var) <= 0:
                raise ValueError('Entry covariance must be finite with positive basket variance.')
            if sizing == 'risk_matched':
                requested = q*np.sqrt(full_var/subset_var)
                cap_hit = abs(requested) > 3.
                actual_q = np.clip(requested, -3., 3.)
            exante_ratio = abs(actual_q/q)*np.sqrt(subset_var/full_var)
        else:
            actual_q, exante_ratio = 0., 0.
        cross_positions = actual_q*selected
        pair_positions = cross_positions@matrix.to_numpy()
        output.loc[timestamp] = pair_positions
        row = dict(timestamp=timestamp, original_units=q, actual_units=actual_q,
                   selected_count=int(chosen.sum()), selected_weight=mass,
                   retained_sgd_exposure=abs(cross_positions.sum()/q),
                   exante_risk_ratio=exante_ratio, risk_cap_hit=bool(cap_hit),
                   gross_leg_notional=np.abs(pair_positions).sum(),
                   estimated_roundtrip_cost=np.abs(cross_positions)@cost,
                   index_deviation=w@d)
        for ccy, value, weight, keep, exposure in zip(weights.index, d, w, chosen, cross_positions):
            row.update({f'deviation_{ccy}': value, f'contribution_{ccy}': weight*value,
                        f'selected_{ccy}': bool(keep), f'exposure_{ccy}': exposure})
        decisions.append(row)
    return output, pd.DataFrame(decisions)


class PairAccountant(MinuteAccountant):
    """Sparse five-leg accounting with exact daily allocation of gap returns.

    Split the quote history at fills and daily boundaries; price differences
    telescope within each constant-position segment. Costs use actual fill
    spreads after aggregating synthetic-cross hedges. No financing is applied.
    """
    def __init__(self, data):
        super().__init__(data)
        self.log_prices = np.log(data.mids.to_numpy())
        self.all_codes = np.full(len(self.index), -1, dtype=int)
        self.all_codes[self.keep] = self.codes
        self.day_ends = np.flatnonzero(np.r_[self.all_codes[1:] != self.all_codes[:-1], True])

    def aggregate(self, locations, values):
        codes = self.all_codes[locations]
        valid = codes >= 0
        return np.bincount(codes[valid], weights=values[valid], minlength=self.n_dates)

    def run(self, targets, delay=0):
        if delay < 0 or int(delay) != delay:
            raise ValueError('Delay must be a nonnegative integer number of common quotes.')
        targets = targets.reindex(columns=self.data.mids.columns)
        if not targets.index.is_monotonic_increasing or targets.index.has_duplicates:
            raise ValueError('Targets must have unique increasing timestamps.')
        if not np.isfinite(targets.to_numpy()).all():
            raise ValueError('Targets must be finite for all pairs.')
        # A missing-quote target takes effect at the first subsequent common
        # quote, then undergoes the same quote-row delay as the dense engine.
        locations = self.index.searchsorted(targets.index, side='left')+delay
        keep = locations < len(self.index)
        sparse = pd.DataFrame(targets.to_numpy()[keep], index=locations[keep])
        sparse = sparse.groupby(level=0).last()
        if 0 not in sparse.index:
            sparse.loc[0] = 0.
        sparse.loc[len(self.index)-1] = 0.  # Final liquidation overrides targets.
        sparse = sparse.sort_index()
        fill_locations, positions = sparse.index.to_numpy(), sparse.to_numpy()
        before = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
        changes = positions-before
        traded = (changes != 0).any(axis=1)
        at, after, prior, changes = (x[traded] for x in [fill_locations, positions, before, changes])
        leg_costs = changes*np.log1p(np.sign(changes)*self.spreads[at]*.5)

        boundaries = np.unique(np.r_[0, fill_locations, self.day_ends])
        left, right = boundaries[:-1], boundaries[1:]
        active = positions[np.searchsorted(fill_locations, left, side='right')-1]
        gross = (active*(self.log_prices[right]-self.log_prices[left])).sum(axis=1)
        frame = pd.DataFrame(index=self.dates)
        frame['gross'] = self.aggregate(right, gross)
        frame['cost'] = self.aggregate(at, leg_costs.sum(axis=1))
        frame['turnover'] = self.aggregate(at, np.abs(changes).sum(axis=1))
        # Segment counts initially count previous-position exposure. Adjust
        # event quotes to match the dense engine's after-fill active counts.
        frame['active_minutes'] = self.aggregate(right, (np.abs(active).sum(axis=1) > 0)*(right-left))
        frame['active_minutes'] += self.aggregate(at, (np.abs(after).sum(axis=1) > 0).astype(int)-
                                                   (np.abs(prior).sum(axis=1) > 0).astype(int))
        gross_exposure = self.aggregate(right, np.abs(active).sum(axis=1)*(right-left))
        gross_exposure += self.aggregate(at, np.abs(after).sum(axis=1)-np.abs(prior).sum(axis=1))
        frame['gross_exposure_minutes'] = gross_exposure
        roll_positions = positions[np.searchsorted(fill_locations, self.roll_locations-1, side='right')-1]
        frame['rollover_exposure'] = self.aggregate(self.roll_locations, np.abs(roll_positions).sum(axis=1))
        frame['net'] = frame.gross-frame.cost
        events = pd.DataFrame({'timestamp': self.index[at], 'before': np.abs(prior).sum(axis=1),
                               'after': np.abs(after).sum(axis=1), 'cost': leg_costs.sum(axis=1),
                               'change': np.abs(changes).sum(axis=1)})
        for j, pair in enumerate(self.data.mids.columns):
            frame[f'cost_{pair}'] = self.aggregate(at, leg_costs[:, j])
            events[f'before_{pair}'], events[f'after_{pair}'] = prior[:, j], after[:, j]
            events[f'cost_{pair}'] = leg_costs[:, j]
        return frame, events
