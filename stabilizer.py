"""Temporal decision layer: turns noisy per-window predictions into committed signs.

A sign is committed only when
  1. it is predicted with confidence >= conf_thr for n_stable consecutive predictions,
  2. the cooldown since the last commit has elapsed, and
  3. it is a different sign from the last commit OR the signer has 'released'
     (idle / no hand for release_n predictions) -- so holding a pose does not repeat.
"""
import config

NO_HAND = "no_hand"


class Stabilizer:
    def __init__(self, conf_thr=config.CONF_THRESHOLD, n_stable=config.STABLE_PREDICTIONS,
                 cooldown_s=config.COOLDOWN_S, release_n=config.RELEASE_PREDICTIONS,
                 unknown=config.UNKNOWN):
        self.conf_thr, self.n_stable = conf_thr, n_stable
        self.cooldown_s, self.release_n, self.unknown = cooldown_s, release_n, unknown
        self.reset()

    def reset(self):
        self.cand = None
        self.cand_count = 0
        self.cand_start = None
        self.last_label = None
        self.last_t = -1e9
        self.idle = 0
        self.released = True

    def update(self, label, conf, t_now, t_capture):
        """Returns {'label','cand_start','t'} when a sign is committed, else None."""
        is_idle = label in (self.unknown, NO_HAND)
        if is_idle:
            self.idle += 1
            if self.idle >= self.release_n:
                self.released = True
        else:
            self.idle = 0

        if is_idle or conf < self.conf_thr:
            self.cand, self.cand_count = None, 0
            return None

        if label != self.cand:
            self.cand, self.cand_count, self.cand_start = label, 1, t_capture
        else:
            self.cand_count += 1

        if self.cand_count >= self.n_stable:
            cooled = (t_now - self.last_t) >= self.cooldown_s
            new_sign = (label != self.last_label) or self.released
            if cooled and new_sign:
                out = {"label": label, "cand_start": self.cand_start, "t": t_now}
                self.last_label, self.last_t, self.released = label, t_now, False
                self.cand, self.cand_count = None, 0
                return out
        return None
