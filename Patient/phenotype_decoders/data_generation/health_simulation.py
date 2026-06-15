import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


State = Tuple[int, int, int]
Action = Tuple[int, int, int]

HEALTHY: State = (1, 0, 0)
MILD: State = (0, 1, 0)
SEVERE: State = (0, 0, 1)

NO_ACTION: Action = (0, 0, 0)
INTERVENTION_A: Action = (0, 0, 1)
INTERVENTION_B: Action = (0, 1, 0)
JOINT_AB: Action = (0, 1, 1)

ACTION_CODE_TO_TUPLE: Dict[int, Action] = {
    0: NO_ACTION,
    1: INTERVENTION_A,
    2: INTERVENTION_B,
    3: JOINT_AB,
}

NOISE_MEAN = 0.0
NOISE_STD = 0.7

STATE_TO_LOGIT_CENTER = {
    HEALTHY: np.array([4.0, 1.0, -4.0], dtype=np.float32),
    MILD: np.array([0.0, 4.0, 0.0], dtype=np.float32),
    SEVERE: np.array([-4.0, 1.0, 4.0], dtype=np.float32),
}

IMAGE_REFERENCE_STATES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.9, 0.1],
        [0.2, 0.8, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.8, 0.2],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
        [0.0, 0.2, 0.8],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class PneumoniaParams:
    k: int = 3
    u: int = 5
    v: int = 6
    p: int = 4
    q: int = 5
    m: int = 7
    n: int = 8


@dataclass(frozen=True)
class FundusParams:
    partial_n: int = 4
    complete_m: int = 8
    relapse_k: int = 4
    synergy_m: int = 5


@dataclass(frozen=True)
class OCTParams:
    period_T: int = 4
    stabilize_n: int = 6


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    output_dir: str
    T: int
    model: str
    params: object
    seed: int
    description: str


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "pneumonia": DatasetConfig(
        name="pneumonia",
        output_dir="Patient/data_generation/generated_sequences/pneumonia",
        T=30,
        model="pneumonia_tolerant",
        params=PneumoniaParams(),
        seed=42,
        description="Pneumonia tolerance model with A/B treatment and switch-to-recovery paths.",
    ),
    "fundus": DatasetConfig(
        name="fundus",
        output_dir="Patient/data_generation/generated_sequences/fundus",
        T=30,
        model="fundus_delayed_synergy",
        params=FundusParams(),
        seed=150,
        description="Color fundus delayed improvement, relapse after incomplete therapy, and synergistic therapy.",
    ),
    "oct": DatasetConfig(
        name="oct",
        output_dir="Patient/data_generation/generated_sequences/oct",
        T=30,
        model="oct_periodic_stabilizing",
        params=OCTParams(),
        seed=150,
        description="OCT recurrent activity with periodic alternation and sustained-treatment stabilization.",
    ),
}


def make_action_sequence(T: int, segments: List[Tuple[int, int]]) -> List[int]:
    seq: List[int] = []
    for action_code, length in segments:
        if action_code not in ACTION_CODE_TO_TUPLE:
            raise ValueError("action_code must be 0, 1, 2, or 3")
        if length < 0:
            raise ValueError("segment length cannot be negative")
        seq.extend([action_code] * length)
    if len(seq) < T:
        seq.extend([0] * (T - len(seq)))
    return seq[:T]


def action_codes_to_tuples(action_codes: List[int]) -> List[Action]:
    return [ACTION_CODE_TO_TUPLE[code] for code in action_codes]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return (exp_values / exp_values.sum()).astype(np.float32)


def add_gaussian_noise_to_state(
    true_state: State,
    mean: float = NOISE_MEAN,
    std: float = NOISE_STD,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if true_state not in STATE_TO_LOGIT_CENTER:
        raise ValueError(f"Unknown state: {true_state}")
    center_logits = STATE_TO_LOGIT_CENTER[true_state]
    noisy_logits = center_logits + rng.normal(mean, std, size=center_logits.shape).astype(np.float32)
    return softmax(noisy_logits)


def round_to_nearest_image_state(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    prob = prob / prob.sum()
    distances = np.linalg.norm(IMAGE_REFERENCE_STATES - prob, axis=1)
    return IMAGE_REFERENCE_STATES[int(np.argmin(distances))].copy()


def validate_pneumonia_params(params: PneumoniaParams) -> None:
    for name, value in vars(params).items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if params.m <= params.p:
        raise ValueError("m must be greater than p")
    if params.n <= params.q:
        raise ValueError("n must be greater than q")


def pneumonia_model(
    action_codes: List[int],
    initial_state: State,
    params: PneumoniaParams,
) -> Tuple[List[Action], List[State]]:
    validate_pneumonia_params(params)
    interventions = action_codes_to_tuples(action_codes)
    states: List[State] = [initial_state]

    previous_action: Optional[int] = None
    run_length = 0
    regimen_started_from_severe = False
    tolerance_A = False
    tolerance_B = False

    for t in range(len(action_codes) - 1):
        current_state = states[-1]
        action = action_codes[t]

        if action == previous_action:
            run_length += 1
        else:
            previous_action = action
            run_length = 1
            regimen_started_from_severe = current_state == SEVERE and action in (1, 2)

        next_state = current_state

        if current_state == HEALTHY:
            next_state = HEALTHY
        elif current_state == MILD:
            if action == 0:
                next_state = SEVERE if run_length >= params.k else MILD
            elif action == 1:
                if regimen_started_from_severe:
                    if run_length >= params.m:
                        tolerance_A = True
                    next_state = MILD
                elif tolerance_B and not tolerance_A:
                    next_state = HEALTHY if run_length >= params.p else MILD
                elif not tolerance_A:
                    next_state = HEALTHY if run_length >= params.u else MILD
                else:
                    next_state = MILD
            elif action == 2:
                if regimen_started_from_severe:
                    if run_length >= params.n:
                        tolerance_B = True
                    next_state = MILD
                elif tolerance_A and not tolerance_B:
                    next_state = HEALTHY if run_length >= params.q else MILD
                elif not tolerance_B:
                    next_state = HEALTHY if run_length >= params.v else MILD
                else:
                    next_state = MILD
        elif current_state == SEVERE:
            if action == 0:
                next_state = SEVERE
            elif action == 1:
                next_state = MILD if run_length >= params.p else SEVERE
                if regimen_started_from_severe and run_length >= params.m:
                    tolerance_A = True
            elif action == 2:
                next_state = MILD if run_length >= params.q else SEVERE
                if regimen_started_from_severe and run_length >= params.n:
                    tolerance_B = True

        states.append(next_state)

    return interventions, states


def fundus_model(
    action_codes: List[int],
    initial_state: State,
    params: FundusParams,
) -> Tuple[List[Action], List[State]]:
    interventions = action_codes_to_tuples(action_codes)
    states: List[State] = [initial_state]

    previous_action: Optional[int] = None
    run_length = 0
    stable_healthy = initial_state == HEALTHY
    partial_under_action: Optional[int] = None
    relapse_remaining: Optional[int] = None

    for t in range(len(action_codes) - 1):
        current_state = states[-1]
        action = action_codes[t]

        if action == previous_action:
            run_length += 1
        else:
            previous_action = action
            run_length = 1
            if action != partial_under_action and not stable_healthy:
                partial_under_action = None

        next_state = current_state

        if stable_healthy or current_state == HEALTHY:
            next_state = HEALTHY
            stable_healthy = True
        elif action == 3:
            next_state = HEALTHY if run_length >= params.synergy_m else current_state
            if next_state == HEALTHY:
                stable_healthy = True
                relapse_remaining = None
        elif action in (1, 2):
            if run_length >= params.complete_m:
                next_state = HEALTHY
                stable_healthy = True
                relapse_remaining = None
            elif run_length >= params.partial_n:
                next_state = MILD
                partial_under_action = action
                relapse_remaining = params.relapse_k
            else:
                next_state = current_state
        elif action == 0:
            if current_state == MILD and relapse_remaining is not None:
                if relapse_remaining > 0:
                    next_state = MILD
                    relapse_remaining -= 1
                else:
                    next_state = SEVERE
                    partial_under_action = None
                    relapse_remaining = None
            elif current_state == MILD:
                next_state = MILD if run_length < params.relapse_k else SEVERE
            else:
                next_state = current_state

        states.append(next_state)

    return interventions, states


def oct_model(
    action_codes: List[int],
    initial_state: State,
    params: OCTParams,
) -> Tuple[List[Action], List[State]]:
    interventions = action_codes_to_tuples(action_codes)
    states: List[State] = [initial_state]

    previous_action: Optional[int] = None
    run_length = 0
    stable_healthy = initial_state == HEALTHY

    for t in range(len(action_codes) - 1):
        action = action_codes[t]
        if action == previous_action:
            run_length += 1
        else:
            previous_action = action
            run_length = 1

        if stable_healthy or states[-1] == HEALTHY:
            next_state = HEALTHY
            stable_healthy = True
        elif action == 2 and run_length >= params.stabilize_n:
            next_state = HEALTHY
            stable_healthy = True
        else:
            phase = ((t + 1) // params.period_T) % 2
            if initial_state == MILD:
                next_state = MILD if phase == 0 else SEVERE
            else:
                next_state = SEVERE if phase == 0 else MILD

        states.append(next_state)

    return interventions, states


def generate_pneumonia_cases(config: DatasetConfig) -> Dict[str, Tuple[List[Action], List[State]]]:
    params: PneumoniaParams = config.params
    T = config.T
    r = params.u + 1
    s = params.v + 1
    w = params.q + 1
    z = params.p + 1
    case_specs = {
        "case_a_healthy_no_intervention": (HEALTHY, make_action_sequence(T, [(0, T)])),
        "case_b_mild_no_intervention": (MILD, make_action_sequence(T, [(0, T)])),
        "case_c_mild_A_to_healthy": (MILD, make_action_sequence(T, [(1, r), (0, T - r)])),
        "case_d_mild_B_to_healthy": (MILD, make_action_sequence(T, [(2, s), (0, T - s)])),
        "case_e_severe_no_intervention": (SEVERE, make_action_sequence(T, [(0, T)])),
        "case_f_severe_A_tolerance": (SEVERE, make_action_sequence(T, [(1, T)])),
        "case_g_severe_B_tolerance": (SEVERE, make_action_sequence(T, [(2, T)])),
        "case_h_severe_A_then_B_to_healthy": (
            SEVERE,
            make_action_sequence(T, [(1, params.m), (2, w), (0, T - params.m - w)]),
        ),
        "case_i_severe_B_then_A_to_healthy": (
            SEVERE,
            make_action_sequence(T, [(2, params.n), (1, z), (0, T - params.n - z)]),
        ),
    }
    return {name: pneumonia_model(actions, initial, params) for name, (initial, actions) in case_specs.items()}


def generate_fundus_cases(config: DatasetConfig) -> Dict[str, Tuple[List[Action], List[State]]]:
    params: FundusParams = config.params
    T = config.T
    case_specs = {
        "case_a_healthy_no_intervention": (HEALTHY, make_action_sequence(T, [(0, T)])),
        "case_b_mild_no_intervention": (MILD, make_action_sequence(T, [(0, T)])),
        "case_c_severe_no_intervention": (SEVERE, make_action_sequence(T, [(0, T)])),
        "case_d_severe_A_partial_relapse": (
            SEVERE,
            make_action_sequence(T, [(1, params.partial_n), (0, T - params.partial_n)]),
        ),
        "case_e_severe_A_complete": (
            SEVERE,
            make_action_sequence(T, [(1, params.complete_m + 1), (0, T - params.complete_m - 1)]),
        ),
        "case_f_severe_B_complete": (
            SEVERE,
            make_action_sequence(T, [(2, params.complete_m + 1), (0, T - params.complete_m - 1)]),
        ),
        "case_g_severe_AB_synergy": (
            SEVERE,
            make_action_sequence(T, [(3, params.synergy_m + 1), (0, T - params.synergy_m - 1)]),
        ),
        "case_h_severe_A_partial_then_AB": (
            SEVERE,
            make_action_sequence(T, [(1, params.partial_n), (3, params.synergy_m), (0, T)]),
        ),
        "case_i_mild_AB_to_healthy": (
            MILD,
            make_action_sequence(T, [(3, params.synergy_m + 1), (0, T - params.synergy_m - 1)]),
        ),
    }
    return {name: fundus_model(actions, initial, params) for name, (initial, actions) in case_specs.items()}


def generate_oct_cases(config: DatasetConfig) -> Dict[str, Tuple[List[Action], List[State]]]:
    params: OCTParams = config.params
    T = config.T
    case_specs = {
        "case_a_healthy_no_intervention": (HEALTHY, make_action_sequence(T, [(0, T)])),
        "case_b_mild_no_intervention_periodic": (MILD, make_action_sequence(T, [(0, T)])),
        "case_c_severe_no_intervention_periodic": (SEVERE, make_action_sequence(T, [(0, T)])),
        "case_d_severe_B_short_periodic": (
            SEVERE,
            make_action_sequence(T, [(2, params.stabilize_n - 2), (0, T)]),
        ),
        "case_e_severe_B_stable": (
            SEVERE,
            make_action_sequence(T, [(2, params.stabilize_n + 1), (0, T)]),
        ),
        "case_f_mild_B_stable": (
            MILD,
            make_action_sequence(T, [(2, params.stabilize_n + 1), (0, T)]),
        ),
        "case_g_severe_A_no_effect_periodic": (SEVERE, make_action_sequence(T, [(1, T)])),
        "case_h_severe_delayed_B_stable": (
            SEVERE,
            make_action_sequence(T, [(0, params.period_T), (2, params.stabilize_n + 1), (0, T)]),
        ),
        "case_i_severe_interrupted_B_periodic": (
            SEVERE,
            make_action_sequence(T, [(2, params.stabilize_n - 2), (0, 2), (2, params.stabilize_n - 2), (0, T)]),
        ),
    }
    return {name: oct_model(actions, initial, params) for name, (initial, actions) in case_specs.items()}


def generate_cases(config: DatasetConfig) -> Dict[str, Tuple[List[Action], List[State]]]:
    if config.model == "pneumonia_tolerant":
        return generate_pneumonia_cases(config)
    if config.model == "fundus_delayed_synergy":
        return generate_fundus_cases(config)
    if config.model == "oct_periodic_stabilizing":
        return generate_oct_cases(config)
    raise ValueError(f"Unknown model: {config.model}")


def generate_csv_rows(
    interventions: List[Action],
    states: List[State],
    seed: Optional[int] = None,
    noise_mean: float = NOISE_MEAN,
    noise_std: float = NOISE_STD,
) -> List[List[float]]:
    if len(interventions) != len(states):
        raise ValueError("interventions and states must have the same length")

    rng = np.random.default_rng(seed)
    rows: List[List[float]] = []
    for timestamp, (intervention, true_state) in enumerate(zip(interventions, states)):
        noisy_prob = add_gaussian_noise_to_state(
            true_state=true_state,
            mean=noise_mean,
            std=noise_std,
            rng=rng,
        )
        rows.append(
            [
                timestamp,
                intervention[0],
                intervention[1],
                intervention[2],
                float(noisy_prob[0]),
                float(noisy_prob[1]),
                float(noisy_prob[2]),
            ]
        )
    return rows


def save_rows(rows: List[List[float]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "a1", "a2", "a3", "s1", "s2", "s3"])
        writer.writerows(rows)


def save_dataset(config: DatasetConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in output_dir.glob("case_*.csv"):
        old_csv.unlink()

    cases = generate_cases(config)
    all_probs = []
    mapped_categories = set()

    print(f"\n[{config.name}] {config.description}")
    print(f"T={config.T}, noise_std={NOISE_STD}, model={config.model}, params={config.params}")

    for idx, (case_name, (interventions, states)) in enumerate(cases.items()):
        rows = generate_csv_rows(interventions, states, seed=config.seed + idx)
        save_rows(rows, output_dir / f"{case_name}.csv")
        for row in rows:
            prob = np.array(row[4:7], dtype=np.float32)
            all_probs.append(prob)
            mapped_categories.add(tuple(round_to_nearest_image_state(prob).round(1)))
        print(f"Saved {output_dir / f'{case_name}.csv'}")

    arr = np.vstack(all_probs)
    sums = arr.sum(axis=1)
    print(f"Rows: {len(arr)}")
    print(f"Probability sum max error: {np.max(np.abs(sums - 1.0)):.8f}")
    print(f"Mapped image categories: {len(mapped_categories)} / {len(IMAGE_REFERENCE_STATES)}")
