import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from Patient.models.pneumonia_model import PneumoniaStateTransitionModel
from Patient.phenotype_decoders.base_decoder import decode_states, load_decoder
from Patient.phenotype_decoders.data_generation.health_simulation import sample_intervention_values
from Patient.phenotype_decoders.generate_observation_sequence import (
    DEFAULT_OBSERVATION_NOISE_MEAN,
    DEFAULT_OBSERVATION_NOISE_STD,
    observe_state_sequence,
    selected_decoders,
    state_to_image_state,
    write_sequence_csv,
)


State = Tuple[int, int, int]
Action = Tuple[int, int, int]


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_state: State
    actions: List[Action]
    description: str
    trigger_rule: str


def format_state(state: State) -> str:
    return f"({state[0]},{state[1]},{state[2]})"


def parameter_text(model: PneumoniaStateTransitionModel) -> str:
    return ",".join(
        f"{key}={model.model_params[key]}"
        for key in ("k", "u", "v", "p", "q", "m", "n")
    )


def action_counts(model: PneumoniaStateTransitionModel, actions: List[Action]) -> str:
    return ",".join(
        [
            f"NO={actions.count(model.NO_INTERVENTION)}",
            f"A={actions.count(model.ANTIBIOTIC_A)}",
            f"B={actions.count(model.ANTIBIOTIC_B)}",
        ]
    )


def alternating(length: int, actions: Tuple[Action, ...]) -> List[Action]:
    return [actions[index % len(actions)] for index in range(length)]


def build_model(params: Dict[str, int]) -> PneumoniaStateTransitionModel:
    if params["m"] <= params["p"]:
        raise ValueError(f"m must be greater than p; got m={params['m']}, p={params['p']}")
    if params["n"] <= params["q"]:
        raise ValueError(f"n must be greater than q; got n={params['n']}, q={params['q']}")
    return PneumoniaStateTransitionModel(
        k=params["k"],
        u=params["u"],
        v=params["v"],
        p=params["p"],
        q=params["q"],
        m=params["m"],
        n=params["n"],
    )


def sample_noisy_params(
    base_model: PneumoniaStateTransitionModel,
    rng: np.random.Generator,
    std: float,
    clip: int,
) -> Dict[str, int]:
    params = {}
    for key in ("k", "u", "v", "p", "q", "m", "n"):
        base_value = int(base_model.model_params[key])
        delta = int(round(float(rng.normal(0.0, std))))
        delta = max(-clip, min(clip, delta))
        params[key] = max(1, base_value + delta)

    params["m"] = max(params["m"], params["p"] + 1)
    params["n"] = max(params["n"], params["q"] + 1)
    return params


def scenario_severe_a_to_mild(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 100 + [model.ANTIBIOTIC_A] * 99
    return Scenario(
        name="severe_A_to_mild_late",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; delayed continuous A changes severe to mild",
        trigger_rule="A starts at t=101; transition time is t=100+p_sample",
    )


def scenario_severe_b_to_mild(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 98 + [model.ANTIBIOTIC_B] * 101
    return Scenario(
        name="severe_B_to_mild_late",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; delayed continuous B changes severe to mild",
        trigger_rule="B starts at t=99; transition time is t=98+q_sample",
    )


def scenario_mild_no_to_severe(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = alternating(125, (model.ANTIBIOTIC_A, model.ANTIBIOTIC_B)) + [model.NO_INTERVENTION] * 74
    return Scenario(
        name="mild_NO_to_severe_late",
        initial_state=model.MILD,
        actions=actions,
        description="initial mild; delayed no-intervention run changes mild to severe",
        trigger_rule="NO starts at t=126; transition time is t=125+k_sample",
    )


def scenario_mild_a_to_healthy(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = alternating(90, (model.ANTIBIOTIC_B, model.NO_INTERVENTION)) + [model.ANTIBIOTIC_A] * 109
    return Scenario(
        name="mild_A_to_healthy_late",
        initial_state=model.MILD,
        actions=actions,
        description="initial mild; delayed continuous A changes mild to healthy",
        trigger_rule="A starts at t=91; transition time is t=90+u_sample",
    )


def scenario_mild_b_to_healthy(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = alternating(120, (model.ANTIBIOTIC_A, model.NO_INTERVENTION)) + [model.ANTIBIOTIC_B] * 79
    return Scenario(
        name="mild_B_to_healthy_late",
        initial_state=model.MILD,
        actions=actions,
        description="initial mild; delayed continuous B changes mild to healthy",
        trigger_rule="B starts at t=121; transition time is t=120+v_sample",
    )


def scenario_severe_a_then_b(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 65 + [model.ANTIBIOTIC_A] * 55 + [model.ANTIBIOTIC_B] * 79
    return Scenario(
        name="severe_A_then_B_two_stage",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; A changes severe to mild, then switched B changes mild to healthy",
        trigger_rule="A starts at t=66; B starts at t=121; transitions depend on p_sample and v_sample",
    )


def scenario_severe_b_then_a(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 79 + [model.ANTIBIOTIC_B] * 46 + [model.ANTIBIOTIC_A] * 74
    return Scenario(
        name="severe_B_then_A_two_stage",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; B changes severe to mild, then switched A changes mild to healthy",
        trigger_rule="B starts at t=80; A starts at t=126; transitions depend on q_sample and u_sample",
    )


def scenario_severe_a_tolerance_then_b(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 60 + [model.ANTIBIOTIC_A] * 95 + [model.ANTIBIOTIC_B] * 44
    return Scenario(
        name="severe_A_tolerance_then_B",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; prolonged A can induce tolerance, then B completes recovery",
        trigger_rule="A starts at t=61; B starts at t=156; transitions depend on p_sample,m_sample,q_sample",
    )


def scenario_severe_b_tolerance_then_a(model: PneumoniaStateTransitionModel) -> Scenario:
    actions = [model.NO_INTERVENTION] * 59 + [model.ANTIBIOTIC_B] * 86 + [model.ANTIBIOTIC_A] * 54
    return Scenario(
        name="severe_B_tolerance_then_A",
        initial_state=model.SEVERE,
        actions=actions,
        description="initial severe; prolonged B can induce tolerance, then A completes recovery",
        trigger_rule="B starts at t=60; A starts at t=146; transitions depend on q_sample,n_sample,p_sample",
    )


def scenario_mild_mixed_no_transition(model: PneumoniaStateTransitionModel) -> Scenario:
    return Scenario(
        name="mild_mixed_no_transition",
        initial_state=model.MILD,
        actions=alternating(199, (model.ANTIBIOTIC_A, model.ANTIBIOTIC_B, model.NO_INTERVENTION)),
        description="initial mild; actions alternate before any threshold can be reached",
        trigger_rule="no consecutive action run reaches k/u/v",
    )


def scenario_severe_mixed_no_transition(model: PneumoniaStateTransitionModel) -> Scenario:
    return Scenario(
        name="severe_mixed_no_transition",
        initial_state=model.SEVERE,
        actions=alternating(199, (model.ANTIBIOTIC_A, model.ANTIBIOTIC_B, model.NO_INTERVENTION)),
        description="initial severe; actions alternate before p/q can be reached",
        trigger_rule="no consecutive antibiotic run reaches p/q",
    )


def scenario_severe_no_intervention(model: PneumoniaStateTransitionModel) -> Scenario:
    return Scenario(
        name="severe_NO_no_transition",
        initial_state=model.SEVERE,
        actions=[model.NO_INTERVENTION] * 199,
        description="initial severe; no intervention keeps severe unchanged",
        trigger_rule="NO has no recovery effect from severe",
    )


def scenario_healthy_control(model: PneumoniaStateTransitionModel) -> Scenario:
    return Scenario(
        name="healthy_control_mixed",
        initial_state=model.HEALTHY,
        actions=(
            [model.NO_INTERVENTION] * 40
            + [model.ANTIBIOTIC_A, model.ANTIBIOTIC_B] * 45
            + [model.ANTIBIOTIC_A] * 30
            + [model.ANTIBIOTIC_B, model.ANTIBIOTIC_A] * 20
        )[:199],
        description="initial healthy; healthy is absorbing even under mixed interventions",
        trigger_rule="healthy state is absorbing",
    )


def build_scenarios(model: PneumoniaStateTransitionModel) -> List[Scenario]:
    scenarios = [
        scenario_severe_a_to_mild(model),
        scenario_severe_b_to_mild(model),
        scenario_mild_no_to_severe(model),
        scenario_mild_a_to_healthy(model),
        scenario_mild_b_to_healthy(model),
        scenario_severe_a_then_b(model),
        scenario_severe_b_then_a(model),
        scenario_severe_a_tolerance_then_b(model),
        scenario_severe_b_tolerance_then_a(model),
        scenario_mild_mixed_no_transition(model),
        scenario_severe_mixed_no_transition(model),
        scenario_severe_no_intervention(model),
        scenario_healthy_control(model),
    ]

    for scenario in scenarios:
        if len(scenario.actions) != 199:
            raise ValueError(f"{scenario.name} must contain 199 actions, got {len(scenario.actions)}")
    return scenarios


def state_changes(states: List[State]) -> List[Tuple[int, State, State]]:
    return [
        (timestamp, states[timestamp - 1], states[timestamp])
        for timestamp in range(1, len(states))
        if states[timestamp] != states[timestamp - 1]
    ]


def format_changes(changes: Iterable[Tuple[int, State, State]]) -> str:
    return ";".join(
        f"t={timestamp}:{format_state(before)}->{format_state(after)}"
        for timestamp, before, after in changes
    )


def save_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario_name",
        "sample_id",
        "run_name",
        "time_steps",
        "action_steps",
        "initial_state",
        "final_state",
        "transitions",
        "base_model_params",
        "sample_model_params",
        "param_noise_std",
        "param_noise_clip",
        "trigger_rule",
        "action_counts",
        "description",
        "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    workspace = Path.cwd().resolve()
    resolved = path.resolve()
    if workspace not in (resolved, *resolved.parents):
        raise ValueError(f"Refusing to clean output outside workspace: {resolved}")
    shutil.rmtree(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate noisy-parameter 200-step pneumonia image observation sequences."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k", type=int, default=45)
    parser.add_argument("--u", type=int, default=55)
    parser.add_argument("--v", type=int, default=65)
    parser.add_argument("--p", type=int, default=50)
    parser.add_argument("--q", type=int, default=40)
    parser.add_argument("--m", type=int, default=90)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--replicates-per-scenario", type=int, default=10)
    parser.add_argument("--param-noise-std", type=float, default=2.0)
    parser.add_argument("--param-noise-clip", type=int, default=4)
    parser.add_argument("--modalities", nargs="+", default=["all"], help="all, pneumonia, fundus, oct")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("Patient/phenotype_decoders/checkpoints"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Patient/phenotype_decoders/observation_data/generated_observations/pneumonia_200_sequences"
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before writing the new dataset.",
    )
    args = parser.parse_args()

    base_params = {"k": args.k, "u": args.u, "v": args.v, "p": args.p, "q": args.q, "m": args.m, "n": args.n}
    base_model = build_model(base_params)
    scenarios = build_scenarios(base_model)

    if args.clean_output:
        clean_output_dir(args.output_dir)

    decoders = selected_decoders(args.modalities)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_decoders = {
        name: load_decoder(spec, args.checkpoint_dir, device)
        for name, spec in decoders.items()
    }

    summary_rows: List[Dict[str, object]] = []
    param_rng = np.random.default_rng(args.seed)
    for scenario_index, scenario in enumerate(scenarios):
        for sample_index in range(args.replicates_per_scenario):
            sample_seed = args.seed + scenario_index * 100 + sample_index
            sample_params = sample_noisy_params(base_model, param_rng, args.param_noise_std, args.param_noise_clip)
            sample_model = build_model(sample_params)

            result = sample_model.simulate(scenario.actions, initial_state=scenario.initial_state, return_details=True)
            states = result["states"]
            if len(states) != 200:
                raise ValueError(f"{scenario.name} sample {sample_index} generated {len(states)} states; expected 200")

            observations = observe_state_sequence(states, seed=sample_seed)
            action_rng = np.random.default_rng(sample_seed + 1000)
            action_values = [
                sample_intervention_values(
                    intervention=action,
                    mean=DEFAULT_OBSERVATION_NOISE_MEAN,
                    std=DEFAULT_OBSERVATION_NOISE_STD,
                    rng=action_rng,
                )
                for action in [sample_model.NO_INTERVENTION] + scenario.actions
            ]

            run_name = f"sample_{sample_index:02d}"
            run_dir = args.output_dir / scenario.name / run_name
            csv_path = run_dir / "observation_sequence.csv"
            write_sequence_csv(csv_path, action_values, observations)

            image_states = [state_to_image_state(state) for state in states]
            for modality, decoder in loaded_decoders.items():
                modality_dir = run_dir / modality
                modality_dir.mkdir(parents=True, exist_ok=True)
                for old_image in modality_dir.glob("*.png"):
                    old_image.unlink()
                images = decode_states(decoder, image_states, device, round_to_training_state=True)
                for timestamp, image in enumerate(images):
                    image.save(modality_dir / f"frame_{timestamp:04d}.png")

            changes = state_changes(states)
            summary_rows.append(
                {
                    "scenario_name": scenario.name,
                    "sample_id": sample_index,
                    "run_name": f"{scenario.name}/{run_name}",
                    "time_steps": len(states),
                    "action_steps": len(scenario.actions),
                    "initial_state": format_state(scenario.initial_state),
                    "final_state": format_state(states[-1]),
                    "transitions": format_changes(changes),
                    "base_model_params": parameter_text(base_model),
                    "sample_model_params": parameter_text(sample_model),
                    "param_noise_std": args.param_noise_std,
                    "param_noise_clip": args.param_noise_clip,
                    "trigger_rule": scenario.trigger_rule,
                    "action_counts": action_counts(sample_model, scenario.actions),
                    "description": scenario.description,
                    "seed": sample_seed,
                }
            )
            print(
                f"Saved {scenario.name}/{run_name}: CSV={csv_path}, "
                f"transitions={format_changes(changes) or 'none'}"
            )

    summary_path = args.output_dir / "scenario_summary.csv"
    save_summary(summary_path, summary_rows)
    print(f"Saved scenario summary to {summary_path}")


if __name__ == "__main__":
    main()
