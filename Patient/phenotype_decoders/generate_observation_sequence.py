import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from Patient.models.pneumonia_model import PneumoniaStateTransitionModel
from Patient.phenotype_decoders import DECODER_SPECS
from Patient.phenotype_decoders.base_decoder import normalize_state, state_label
from Patient.phenotype_decoders.data_generation.health_simulation import sample_intervention_values
from Patient.phenotype_decoders.multi_decoder import multi_decoder


State = Tuple[int, int, int]
Action = Tuple[int, int, int]
ObservationState = Tuple[float, float, float]

HEALTHY: State = PneumoniaStateTransitionModel.HEALTHY
MILD: State = PneumoniaStateTransitionModel.MILD
SEVERE: State = PneumoniaStateTransitionModel.SEVERE

OBSERVATION_HEALTHY: ObservationState = (1.0, 0.0, 0.0)
OBSERVATION_MILD: ObservationState = (0.0, 1.0, 0.0)
OBSERVATION_SEVERE: ObservationState = (0.0, 0.0, 1.0)

DEFAULT_OBSERVATION_NOISE_MEAN = 0.0
DEFAULT_OBSERVATION_NOISE_STD = 0.1

PNEUMONIA_STATE_TO_OBSERVATION = {
    HEALTHY: OBSERVATION_HEALTHY,
    MILD: OBSERVATION_MILD,
    SEVERE: OBSERVATION_SEVERE,
}

IMAGE_REFERENCE_STATES: Tuple[ObservationState, ...] = (
    OBSERVATION_HEALTHY,
    OBSERVATION_MILD,
    OBSERVATION_SEVERE,
)

ACTION_CODE_TO_TUPLE: Dict[int, Action] = {
    0: PneumoniaStateTransitionModel.NO_INTERVENTION,
    1: PneumoniaStateTransitionModel.ANTIBIOTIC_A,
    2: PneumoniaStateTransitionModel.ANTIBIOTIC_B,
}
ACTION_TUPLE_TO_CODE = {action: code for code, action in ACTION_CODE_TO_TUPLE.items()}

ACTION_TOKEN_TO_CODE = {
    "0": 0,
    "none": 0,
    "no": 0,
    "n": 0,
    "1": 1,
    "a": 1,
    "antibiotic_a": 1,
    "2": 2,
    "b": 2,
    "antibiotic_b": 2,
}


def selected_decoders(names: Iterable[str]) -> Dict[str, object]:
    if "all" in names:
        return DECODER_SPECS
    selected = {}
    for name in names:
        if name not in DECODER_SPECS:
            raise ValueError(f"Unknown decoder {name}; choose from {sorted(DECODER_SPECS)} or all")
        selected[name] = DECODER_SPECS[name]
    return selected


def normalize_observation(observation: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(observation), dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("Observation must contain exactly three values")
    if np.any(values < 0):
        raise ValueError("Observation probabilities cannot be negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Observation probabilities must sum to a positive value")
    return values / total


def state_to_observation_state(state: State) -> ObservationState:
    state_tuple = tuple(state)
    if state_tuple not in PNEUMONIA_STATE_TO_OBSERVATION:
        raise ValueError(f"Unknown pneumonia state: {state_tuple}")
    return PNEUMONIA_STATE_TO_OBSERVATION[state_tuple]


def sample_noisy_observation(
    state: State,
    rng: Optional[np.random.Generator] = None,
    noise_mean: float = DEFAULT_OBSERVATION_NOISE_MEAN,
    noise_std: float = DEFAULT_OBSERVATION_NOISE_STD,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    center = np.asarray(state_to_observation_state(state), dtype=np.float32)
    noisy = rng.normal(center + noise_mean, noise_std).astype(np.float32)
    noisy = np.clip(noisy, 0.0, None)
    total = float(noisy.sum())
    if total <= 0.0:
        return center.copy()
    return (noisy / total).astype(np.float32)


def observe_state_sequence(
    states: Iterable[State],
    seed: Optional[int] = None,
    noise_mean: float = DEFAULT_OBSERVATION_NOISE_MEAN,
    noise_std: float = DEFAULT_OBSERVATION_NOISE_STD,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        sample_noisy_observation(
            state=state,
            rng=rng,
            noise_mean=noise_mean,
            noise_std=noise_std,
        )
        for state in states
    ]


def round_observation_to_image_state(observation: Iterable[float]) -> ObservationState:
    normalized = normalize_observation(observation)
    return IMAGE_REFERENCE_STATES[int(np.argmax(normalized))]


def state_to_image_state(state: State) -> ObservationState:
    return tuple(float(value) for value in state_to_observation_state(state))


def parse_state(value: str) -> State:
    cleaned = value.strip().replace("(", "").replace(")", "")
    values = normalize_state(float(part.strip()) for part in cleaned.split(",") if part.strip())
    return tuple(int(x) for x in IMAGE_REFERENCE_STATES[int(np.argmax(values))])


def parse_action_token(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in ACTION_TOKEN_TO_CODE:
        raise ValueError(f"Unknown action token: {value}; use 0/none, 1/A, or 2/B")
    return ACTION_TOKEN_TO_CODE[normalized]


def parse_action_tuple(value: str) -> int:
    parts = [part.strip() for part in value.strip().replace("(", "").replace(")", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Action tuple must contain three values: {value}")
    try:
        action = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"Action tuple values must be integers: {value}") from exc
    if action not in ACTION_TUPLE_TO_CODE:
        raise ValueError(
            f"Unsupported action tuple: {value}; use (0,0,0), (0,0,1), or (0,1,0)"
        )
    return ACTION_TUPLE_TO_CODE[action]


def skip_separators(value: str, pos: int) -> int:
    while pos < len(value) and value[pos] in " \t\r\n,;":
        pos += 1
    return pos


def parse_optional_repeat(value: str, pos: int) -> Tuple[int, int]:
    while pos < len(value) and value[pos] in " \t\r\n":
        pos += 1
    if pos >= len(value) or value[pos] not in "xX*":
        return 1, pos
    pos += 1
    while pos < len(value) and value[pos] in " \t\r\n":
        pos += 1
    match = re.match(r"\d+", value[pos:])
    if not match:
        raise ValueError("Repeat operator x/* must be followed by a positive integer")
    repeat = int(match.group(0))
    if repeat <= 0:
        raise ValueError("Repeat count must be positive")
    return repeat, pos + len(match.group(0))


def parse_action_sequence(value: str) -> List[Action]:
    actions: List[Action] = []
    pos = 0
    while True:
        pos = skip_separators(value, pos)
        if pos >= len(value):
            break

        if value[pos] == "(":
            end = value.find(")", pos)
            if end == -1:
                raise ValueError(f"Unclosed action tuple in: {value[pos:]}")
            code = parse_action_tuple(value[pos : end + 1])
            pos = end + 1
            repeat, pos = parse_optional_repeat(value, pos)
        else:
            start = pos
            while pos < len(value) and value[pos] not in " \t\r\n,;":
                pos += 1
            token = value[start:pos]
            repeat = 1
            action_token = token
            repeat_match = re.fullmatch(r"(.+?)[xX*](\d+)", token)
            if repeat_match and repeat_match.group(1).strip().lower().replace("-", "_") in ACTION_TOKEN_TO_CODE:
                action_token = repeat_match.group(1)
                repeat = int(repeat_match.group(2))
            else:
                extra_repeat, new_pos = parse_optional_repeat(value, pos)
                if extra_repeat != 1:
                    repeat = extra_repeat
                    pos = new_pos
            code = parse_action_token(action_token)

        if repeat <= 0:
            raise ValueError("Repeat count must be positive")
        actions.extend([ACTION_CODE_TO_TUPLE[code]] * repeat)
    return actions


def write_sequence_csv(
    path: Path,
    action_values: List[np.ndarray],
    observations: List[np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "a1", "a2", "a3", "o1", "o2", "o3"])
        for timestamp, (action, observation) in enumerate(zip(action_values, observations)):
            writer.writerow(
                [
                    timestamp,
                    float(action[0]),
                    float(action[1]),
                    float(action[2]),
                    float(observation[0]),
                    float(observation[1]),
                    float(observation[2]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pneumonia transition model and generate observation sequence images."
    )
    parser.add_argument(
        "--initial-state",
        default="(0,0,1)",
        help="Initial hidden state. Use (1,0,0), (0,1,0), or (0,0,1). Continuous values are mapped by argmax.",
    )
    parser.add_argument(
        "--actions",
        required=True,
        help=(
            "Intervention sequence. Use 0/none/(0,0,0), 1/A/(0,0,1), 2/B/(0,1,0). "
            "Repeats like 1x4 or (0,0,0)*3 are supported."
        ),
    )
    parser.add_argument("--modalities", nargs="+", default=["all"], help="all, pneumonia, fundus, oct")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("Patient/phenotype_decoders/checkpoints"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Patient/phenotype_decoders/observation_data/generated_observations/observation_sequence"
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--images-from",
        choices=["true", "observation"],
        default="true",
        help="Use true hidden states for exact class images, or noisy observations mapped by argmax.",
    )
    args = parser.parse_args()

    initial_state = parse_state(args.initial_state)
    actions = parse_action_sequence(args.actions)

    model = PneumoniaStateTransitionModel()
    result = model.simulate(actions, initial_state=initial_state, return_details=True)
    states = result["states"]
    observations = observe_state_sequence(states, seed=args.seed)
    action_rng = np.random.default_rng(args.seed + 1)
    action_values = [
        sample_intervention_values(
            intervention=action,
            mean=DEFAULT_OBSERVATION_NOISE_MEAN,
            std=DEFAULT_OBSERVATION_NOISE_STD,
            rng=action_rng,
        )
        for action in [PneumoniaStateTransitionModel.NO_INTERVENTION] + actions
    ]

    if args.images_from == "true":
        image_states = [state_to_image_state(state) for state in states]
    else:
        image_states = [round_observation_to_image_state(observation) for observation in observations]

    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"initial_{state_label(initial_state)}_steps_{len(actions)}_seed_{args.seed}"
    run_dir = args.output_dir / run_name
    csv_path = run_dir / "observation_sequence.csv"
    write_sequence_csv(csv_path, action_values, observations)

    decode_inputs = image_states if args.images_from == "true" else observations
    modality_names = tuple(selected_decoders(args.modalities).keys())
    for name in modality_names:
        modality_dir = run_dir / name
        modality_dir.mkdir(parents=True, exist_ok=True)
        for old_image in modality_dir.glob("*.png"):
            old_image.unlink()

    for idx, decode_input in enumerate(decode_inputs):
        images = multi_decoder(
            decode_input,
            modalities=modality_names,
            checkpoint_dir=args.checkpoint_dir,
        )
        for name, image in zip(modality_names, images):
            image.save(run_dir / name / f"frame_{idx:04d}.png")

    for name in modality_names:
        modality_dir = run_dir / name
        print(f"Saved {len(decode_inputs)} {name} image(s) to {modality_dir}")

    print(f"Saved observation sequence CSV to {csv_path}")
    print(f"Generated {len(states)} time steps")
    print(f"Initial state: {initial_state}")
    print(f"Final true state: {states[-1]}")


if __name__ == "__main__":
    main()
