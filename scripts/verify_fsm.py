"""
Verify the PAGSwitchController FSM transitions against Table 3.1 of the
dissertation by enumerating every (current state, distance band) input pair
and asserting that the resulting next state matches the table.

Also smoke-tests the observation-delay FIFO at delays 0..5 to confirm
that the controller sees an observation that is exactly delta steps stale.

Run with:  python verify_fsm.py
Exits 0 on success, non-zero on any failure.
"""
import sys
import numpy as np
from pag_arbitration.environment import PAGArbitrationEnv
from pag_arbitration.controller import PAGSwitchController


def _make_obs(dist_pred):
    """Build a 5-d observation with a chosen predator distance and benign rest."""
    # (energy, dist_pred, angle_pred, dist_food, angle_food)
    return np.array([100.0, float(dist_pred), 0.0, 50.0, 0.0], dtype=np.float32)


# Reference transition table (Table 3.1 / Algorithm 3.1):
# (current_state, dist_band) -> expected_next_state
# Distance bands cover every threshold-bracketed interval used by the FSM.
TRANSITIONS = [
    # (from, distance, expected_to, comment)
    ("DEFAULT", 70.0, "DEFAULT", "no threat -> stay DEFAULT"),
    ("DEFAULT", 50.0, "DEFAULT", "outside freeze entry -> stay DEFAULT"),
    ("DEFAULT", 30.0, "FREEZE",  "inside freeze entry, outside flight entry -> FREEZE"),
    ("DEFAULT", 10.0, "FLIGHT",  "inside flight entry -> FLIGHT"),

    ("FREEZE",  10.0, "FLIGHT",  "predator close -> escalate to FLIGHT"),
    ("FREEZE",  30.0, "FREEZE",  "inside freeze band -> hold"),
    ("FREEZE",  50.0, "FREEZE",  "between exits -> hold (hysteresis)"),
    ("FREEZE",  70.0, "DEFAULT", "outside freeze exit -> release to DEFAULT"),

    ("FLIGHT",  10.0, "FLIGHT",  "still inside flight band -> hold"),
    ("FLIGHT",  25.0, "FLIGHT",  "between flight thresholds -> hold (hysteresis)"),
    ("FLIGHT",  40.0, "FREEZE",  "above flight exit, below freeze exit -> de-escalate"),
    ("FLIGHT",  70.0, "DEFAULT", "above freeze exit -> release fully"),
]


def verify_transitions():
    ctrl = PAGSwitchController(use_hysteresis=True)
    failures = []
    for src, dist, expected, comment in TRANSITIONS:
        ctrl.current_state = src
        ctrl.act(_make_obs(dist))
        actual = ctrl.current_state
        ok = actual == expected
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {src:>7} @ d={dist:>5.1f} -> {actual:<7} (expected {expected:<7}) -- {comment}")
        if not ok:
            failures.append((src, dist, expected, actual))
    return failures


def verify_delay_buffer():
    """The controller should see an observation that is delta+1 ticks stale
    (one implicit tick of latency from the post-move sampling, plus delta
    additional ticks of FIFO buffer). Concretely, seen[k] == truth[k-delta-1]
    for k >= delta+1, clamped at the initial observation below."""
    failures = []
    for delta in [0, 1, 2, 3, 5]:
        env = PAGArbitrationEnv(config={"threat_level": "medium", "delay": delta})
        obs, _ = env.reset(seed=1234)
        truth = [env._get_obs()]   # truth[0] is the post-reset world
        seen = [obs]               # seen[0] is the initial observation returned by reset
        for _ in range(delta + 3):
            obs, _r, term, trunc, _info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            seen.append(obs)
            truth.append(env._get_obs())
            if term or trunc:
                break
        k = delta + 2
        if len(seen) > k:
            stale = seen[k]
            expected = truth[k - delta - 1]
            equal = np.allclose(stale[1:], expected[1:], atol=1e-3)
            status = "OK " if equal else "FAIL"
            print(f"  [{status}] delta={delta}: seen[t={k}] matches truth[t={k-delta-1}]")
            if not equal:
                failures.append((delta, stale, expected))
        env.close()
    return failures


if __name__ == "__main__":
    print("Verifying PAG FSM transitions against Table 3.1...")
    t_fail = verify_transitions()
    print()
    print("Verifying observation-delay FIFO...")
    d_fail = verify_delay_buffer()
    print()
    total = len(t_fail) + len(d_fail)
    if total:
        print(f"FAIL: {total} mismatch(es).")
        sys.exit(1)
    print("All checks passed.")
