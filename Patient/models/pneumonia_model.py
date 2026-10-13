from typing import Any, Dict, List, Optional, Tuple

from .base_model import BaseHealthModel, EventType


State = Tuple[int, int, int]
Action = Tuple[int, int, int]


class PneumoniaStateTransitionModel(BaseHealthModel):
    """Pneumonia state transition model with antibiotic tolerance."""

    HEALTHY: State = (1, 0, 0)
    MILD: State = (0, 1, 0)
    SEVERE: State = (0, 0, 1)

    NO_INTERVENTION: Action = (0, 0, 0)
    ANTIBIOTIC_A: Action = (0, 0, 1)
    ANTIBIOTIC_B: Action = (0, 1, 0)

    def __init__(
        self,
        k: int = 3,
        u: int = 5,
        v: int = 6,
        p: int = 4,
        q: int = 5,
        m: int = 7,
        n: int = 8,
    ):
        params = {
            "k": k,
            "u": u,
            "v": v,
            "p": p,
            "q": q,
            "m": m,
            "n": n,
            "type": "pneumonia_state_transition",
        }
        super().__init__("PneumoniaStateTransitionModel", params)
        self.reset_internal_state()

    def _validate_params(self) -> None:
        for name in ("k", "u", "v", "p", "q", "m", "n"):
            value = self.model_params[name]
            assert isinstance(value, int), f"{name} must be an integer"
            assert value > 0, f"{name} ({value}) must be positive"

        p = self.model_params["p"]
        q = self.model_params["q"]
        m = self.model_params["m"]
        n = self.model_params["n"]
        assert m > p, f"m ({m}) must be greater than p ({p})"
        assert n > q, f"n ({n}) must be greater than q ({q})"

    def get_initial_state(self) -> State:
        return self.SEVERE

    def get_available_interventions(self) -> List[Action]:
        return [self.NO_INTERVENTION, self.ANTIBIOTIC_A, self.ANTIBIOTIC_B]

    def reset_internal_state(self) -> None:
        self._previous_action: Optional[Action] = None
        self._run_length = 0
        self._regimen_started_from_severe = False
        self._tolerance_A = False
        self._tolerance_B = False
        self._last_event = EventType.STATE_MAINTENANCE

    @staticmethod
    def _run_length_from_history(intervention_history: List[Action]) -> int:
        if not intervention_history:
            return 0
        last_action = intervention_history[-1]
        run_length = 0
        for action in reversed(intervention_history):
            if action != last_action:
                break
            run_length += 1
        return run_length

    def _transition(
        self,
        current_state: State,
        action: Action,
        run_length: int,
        regimen_started_from_severe: bool,
        tolerance_A: bool,
        tolerance_B: bool,
    ) -> Tuple[State, EventType, bool, bool]:
        k = self.model_params["k"]
        u = self.model_params["u"]
        v = self.model_params["v"]
        p = self.model_params["p"]
        q = self.model_params["q"]
        m = self.model_params["m"]
        n = self.model_params["n"]

        next_state = current_state
        event = EventType.STATE_MAINTENANCE

        if current_state == self.HEALTHY:
            return self.HEALTHY, event, tolerance_A, tolerance_B

        if current_state == self.MILD:
            if action == self.NO_INTERVENTION:
                if run_length >= k:
                    next_state = self.SEVERE
                    event = EventType.DETERIORATION
                else:
                    next_state = self.MILD
            elif action == self.ANTIBIOTIC_A:
                if regimen_started_from_severe:
                    if run_length >= m:
                        tolerance_A = True
                        event = EventType.TOLERANCE_INDUCTION
                    next_state = self.MILD
                elif tolerance_B and not tolerance_A:
                    if run_length >= p:
                        next_state = self.HEALTHY
                        event = EventType.COMPLETE_TRANSITION
                    else:
                        next_state = self.MILD
                elif not tolerance_A:
                    if run_length >= u:
                        next_state = self.HEALTHY
                        event = EventType.COMPLETE_TRANSITION
                    else:
                        next_state = self.MILD
                else:
                    next_state = self.MILD
            elif action == self.ANTIBIOTIC_B:
                if regimen_started_from_severe:
                    if run_length >= n:
                        tolerance_B = True
                        event = EventType.TOLERANCE_INDUCTION
                    next_state = self.MILD
                elif tolerance_A and not tolerance_B:
                    if run_length >= q:
                        next_state = self.HEALTHY
                        event = EventType.COMPLETE_TRANSITION
                    else:
                        next_state = self.MILD
                elif not tolerance_B:
                    if run_length >= v:
                        next_state = self.HEALTHY
                        event = EventType.COMPLETE_TRANSITION
                    else:
                        next_state = self.MILD
                else:
                    next_state = self.MILD

            return next_state, event, tolerance_A, tolerance_B

        if current_state == self.SEVERE:
            if action == self.NO_INTERVENTION:
                next_state = self.SEVERE
            elif action == self.ANTIBIOTIC_A:
                if run_length >= p:
                    next_state = self.MILD
                    event = EventType.PARTIAL_TRANSITION
                else:
                    next_state = self.SEVERE
                if regimen_started_from_severe and run_length >= m:
                    tolerance_A = True
                    event = EventType.TOLERANCE_INDUCTION
            elif action == self.ANTIBIOTIC_B:
                if run_length >= q:
                    next_state = self.MILD
                    event = EventType.PARTIAL_TRANSITION
                else:
                    next_state = self.SEVERE
                if regimen_started_from_severe and run_length >= n:
                    tolerance_B = True
                    event = EventType.TOLERANCE_INDUCTION

        return next_state, event, tolerance_A, tolerance_B

    def check_event(
        self,
        intervention_history: List[Action],
        state_history: Optional[List[State]] = None,
    ) -> EventType:
        if not intervention_history:
            return EventType.STATE_MAINTENANCE

        current_state = state_history[-1] if state_history else self.get_initial_state()
        action = intervention_history[-1]
        run_length = self._run_length_from_history(intervention_history)
        action_changed = len(intervention_history) == 1 or intervention_history[-2] != action
        regimen_started_from_severe = (
            current_state == self.SEVERE and action in (self.ANTIBIOTIC_A, self.ANTIBIOTIC_B)
            if action_changed
            else self._regimen_started_from_severe
        )
        _, event, _, _ = self._transition(
            current_state=current_state,
            action=action,
            run_length=run_length,
            regimen_started_from_severe=regimen_started_from_severe,
            tolerance_A=self._tolerance_A,
            tolerance_B=self._tolerance_B,
        )
        return event

    def get_next_state(
        self,
        current_state: State,
        intervention_history: List[Action],
        state_history: Optional[List[State]] = None,
    ) -> State:
        if not intervention_history:
            return current_state

        action = intervention_history[-1]
        if action == self._previous_action:
            self._run_length += 1
        else:
            self._previous_action = action
            self._run_length = 1
            self._regimen_started_from_severe = (
                current_state == self.SEVERE and action in (self.ANTIBIOTIC_A, self.ANTIBIOTIC_B)
            )

        next_state, event, tolerance_A, tolerance_B = self._transition(
            current_state=current_state,
            action=action,
            run_length=self._run_length,
            regimen_started_from_severe=self._regimen_started_from_severe,
            tolerance_A=self._tolerance_A,
            tolerance_B=self._tolerance_B,
        )
        self._tolerance_A = tolerance_A
        self._tolerance_B = tolerance_B
        self._last_event = event
        return next_state

    def simulate(
        self,
        intervention_sequence: List[Action],
        initial_state: Optional[State] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        self.reset_internal_state()
        return super().simulate(intervention_sequence, initial_state, return_details)

    def get_internal_state(self) -> Dict[str, Any]:
        return {
            "previous_action": self._previous_action,
            "run_length": self._run_length,
            "regimen_started_from_severe": self._regimen_started_from_severe,
            "tolerance_A": self._tolerance_A,
            "tolerance_B": self._tolerance_B,
            "last_event": self._last_event.value,
        }

    @staticmethod
    def state_name(state: State) -> str:
        if state == PneumoniaStateTransitionModel.HEALTHY:
            return "healthy"
        if state == PneumoniaStateTransitionModel.MILD:
            return "mild"
        if state == PneumoniaStateTransitionModel.SEVERE:
            return "severe"
        return "unknown"

    @staticmethod
    def action_name(action: Action) -> str:
        if action == PneumoniaStateTransitionModel.NO_INTERVENTION:
            return "none"
        if action == PneumoniaStateTransitionModel.ANTIBIOTIC_A:
            return "antibiotic_A"
        if action == PneumoniaStateTransitionModel.ANTIBIOTIC_B:
            return "antibiotic_B"
        return "unknown"

    @staticmethod
    def event_name(event: EventType, action: Optional[Action] = None) -> str:
        if event == EventType.TOLERANCE_INDUCTION:
            if action == PneumoniaStateTransitionModel.ANTIBIOTIC_A:
                return "antibiotic_A_tolerance"
            if action == PneumoniaStateTransitionModel.ANTIBIOTIC_B:
                return "antibiotic_B_tolerance"
            return "tolerance_induction"
        return event.value
