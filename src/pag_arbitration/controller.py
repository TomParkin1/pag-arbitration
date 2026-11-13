import numpy as np

class ContinuousBlendController:
    """
    Arbitration controller utilising cooperative command fusion.
    Employs a Softmax function over utilities to generate continuous behavioural weights.
    """
    def __init__(self, beta=1.0, threat_radius=40.0):
        self.beta = beta
        self.threat_radius = threat_radius
        self.reset_tracking()

    def reset_tracking(self):
        self.mode = None
        self.switch_count = 0

    def act(self, obs):
        energy, dist_pred, angle_pred, dist_food, angle_food = obs

        u_forage = (100.0 - energy) / 100.0
        u_avoid = np.clip(1.0 - (dist_pred / self.threat_radius), 0.0, 1.0)

        a_forage = np.array([np.cos(angle_food), np.sin(angle_food)])
        a_avoid = np.array([-np.cos(angle_pred), -np.sin(angle_pred)])

        exp_f = np.exp(self.beta * u_forage)
        exp_a = np.exp(self.beta * u_avoid)
        sum_exp = exp_f + exp_a

        w_forage = exp_f / sum_exp
        w_avoid = exp_a / sum_exp

        # Track dominant behavioural mode for chattering analysis.
        new_mode = "AVOID" if w_avoid > w_forage else "FORAGE"
        if self.mode is not None and new_mode != self.mode:
            self.switch_count += 1
        self.mode = new_mode

        blended_action = (w_forage * a_forage) + (w_avoid * a_avoid)
        return np.clip(blended_action, -1.0, 1.0)


class UtilitySwitchController:
    """
    Arbitration controller utilising a 'winner-take-all' mathematical switch.
    Calculates competing utilities but strictly executes the highest without blending.
    This acts as the non-biological, mathematical baseline for discrete action selection.
    """
    def __init__(self, threat_radius=40.0):
        self.threat_radius = threat_radius
        self.reset_tracking()

    def reset_tracking(self):
        self.mode = None
        self.switch_count = 0

    def act(self, obs):
        energy, dist_pred, angle_pred, dist_food, angle_food = obs

        # Utilities match the ContinuousBlend formulation to keep the comparison fair.
        u_forage = (100.0 - energy) / 100.0
        u_avoid = np.clip(1.0 - (dist_pred / self.threat_radius), 0.0, 1.0)

        if u_avoid > u_forage:
            new_mode = "AVOID"
            action = np.array([-np.cos(angle_pred), -np.sin(angle_pred)])
        else:
            new_mode = "FORAGE"
            action = np.array([np.cos(angle_food), np.sin(angle_food)])

        if self.mode is not None and new_mode != self.mode:
            self.switch_count += 1
        self.mode = new_mode

        return action


class PAGSwitchController:
    """
    Arbitration controller utilising a biologically inspired subsumption architecture.
    Uses a base controller (UtilitySwitch) for default behaviour, but allows a
    simulated midbrain PAG layer to override motor outputs during critical threats.
    Implements a Finite-State Machine (FSM) with hysteresis thresholds.
    """
    def __init__(self, base_controller=None, use_hysteresis=True):
        self.base_controller = base_controller or UtilitySwitchController()
        self.use_hysteresis = use_hysteresis

        if use_hysteresis:
            # Asymmetric thresholds: exit band sits beyond the predator's vision radius
            # so a freshly-released agent is not instantly re-acquired.
            self.freeze_entry = 40.0
            self.freeze_exit = 55.0
            self.flight_entry = 20.0
            self.flight_exit = 30.0
        else:
            # Collapsed thresholds — entry and exit coincide, removing deadband.
            self.freeze_entry = 40.0
            self.freeze_exit = 40.0
            self.flight_entry = 20.0
            self.flight_exit = 20.0

        self.reset_tracking()

    def reset_tracking(self):
        self.current_state = "DEFAULT"
        self.mode = "DEFAULT"
        self.switch_count = 0

    def act(self, obs):
        energy, dist_pred, angle_pred, dist_food, angle_food = obs

        prev_state = self.current_state

        if self.current_state == "DEFAULT":
            if dist_pred < self.flight_entry:
                self.current_state = "FLIGHT"
            elif dist_pred < self.freeze_entry:
                self.current_state = "FREEZE"

        elif self.current_state == "FREEZE":
            if dist_pred < self.flight_entry:
                self.current_state = "FLIGHT"
            elif dist_pred > self.freeze_exit:
                self.current_state = "DEFAULT"

        elif self.current_state == "FLIGHT":
            if dist_pred > self.flight_exit:
                if dist_pred < self.freeze_exit:
                    self.current_state = "FREEZE"
                else:
                    self.current_state = "DEFAULT"

        if self.current_state != prev_state:
            self.switch_count += 1
        self.mode = self.current_state

        if self.current_state == "DEFAULT":
            action = self.base_controller.act(obs)
        elif self.current_state == "FREEZE":
            action = np.array([0.0, 0.0])
        elif self.current_state == "FLIGHT":
            action = np.array([-np.cos(angle_pred), -np.sin(angle_pred)])

        return action