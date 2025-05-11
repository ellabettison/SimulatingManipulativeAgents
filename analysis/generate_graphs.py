import json
from typing import List

from typing_extensions import Dict
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def compute_reject_rates_by_scenario(dataset: List[Dict]) -> Dict[str, float]:
    scenario_counts = defaultdict(lambda: {'reject': 0, 'total': 0})
    for item in dataset:
        if not item.get("ai_is_malicious"):
            continue
        for chat in item.get("scenario_chats", []):
            scenario = chat.get("scenario")
            for turn in chat.get("chat_history", []):
                feedback = turn.get("feedback")
                if feedback:
                    scenario_counts[scenario]['total'] += 1
                    if feedback.lower() == 'reject':
                        scenario_counts[scenario]['reject'] += 1
    return {
        scenario: counts['reject'] / counts['total'] if counts['total'] > 0 else 0
        for scenario, counts in scenario_counts.items()
    }


def plot_avg_accept_rates_by_scenario_and_setup(datasets_by_model: Dict[str, List[List[Dict]]]):
    """
    Args:
        datasets_by_model: dict where values are [zero_turn, one_turn, one_turn + planning] datasets per model
    """
    setups = ['Zero-turn', 'One-turn', 'One-turn + Planning']
    setup_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Accumulate scenario → setup → list of rates
    scenario_setup_rates = defaultdict(lambda: defaultdict(list))

    for model, setup_datasets in datasets_by_model.items():
        for setup_idx, dataset in enumerate(setup_datasets):
            rates = compute_reject_rates_by_scenario(dataset)
            for scenario, rate in rates.items():
                scenario_setup_rates[scenario][setup_idx].append(rate)

    all_scenarios = sorted(scenario_setup_rates.keys())
    num_scenarios = len(all_scenarios)
    num_setups = len(setups)
    x = np.arange(num_scenarios)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for setup_idx in range(num_setups):
        y_values = []
        for scenario in all_scenarios:
            setup_rates = scenario_setup_rates[scenario].get(setup_idx, [])
            avg_rate = np.mean(setup_rates) if setup_rates else 0
            y_values.append(avg_rate)

        offset = (setup_idx - 1) * width  # center 3 bars around x
        bars = ax.bar(x + offset, y_values, width,
                      color=setup_colors[setup_idx], label=setups[setup_idx])

        # Annotate bars
        for i, height in enumerate(y_values):
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(x[i] + offset, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    ax.set_ylabel("Average Reject Rate")
    ax.set_title("Average Reject Rate by Scenario and Setup")
    ax.set_xticks(x)
    ax.set_xticklabels(all_scenarios, rotation=45, ha='right')
    ax.legend(title="Setup")
    plt.tight_layout()
    plt.show()



def compute_reject_rate(dataset):
    total = 0
    accepted = 0
    for item in dataset:
        if not item.get("ai_is_malicious"):
            continue
        for chat in item.get("scenario_chats", []):
            for turn in chat.get("chat_history", []):
                feedback = turn.get("feedback")
                if feedback:
                    total += 1
                    if feedback.lower() == "reject":
                        accepted += 1
    return accepted / total if total > 0 else 0

def plot_acceptance_rates_grouped_by_setup(datasets_by_model):
    """
    Args:
        datasets_by_model: dict with keys as model names and values as tuples of datasets:
            {
                'DeepSeek': (zero_turn_ds, one_turn_ds, one_turn_planning_ds),
                'Gemini':   (zero_turn_ds, one_turn_ds, one_turn_planning_ds),
                'LLaMA':    (zero_turn_ds, one_turn_ds, one_turn_planning_ds),
            }
    """
    setups = ['Zero-turn', 'One-turn', 'One-turn + Planning']
    models = list(datasets_by_model.keys())
    num_setups = len(setups)
    num_models = len(models)

    # Prepare data matrix: rows = setups, cols = models
    rates = np.zeros((num_setups, num_models))
    for model_idx, model in enumerate(models):
        for setup_idx in range(num_setups):
            dataset = datasets_by_model[model][setup_idx]
            rates[setup_idx, model_idx] = compute_reject_rate(dataset)

    x = np.arange(num_setups)
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []

    for i, model in enumerate(models):
        bar = ax.bar(x + offsets[i], rates[:, i], width, label=model)
        bars.append(bar)

    ax.set_ylabel('Reject Rate')
    ax.set_title('Average Reject Rates by Turn Setup')
    ax.set_xticks(x)
    ax.set_xticklabels(setups)
    ax.legend()

    # Annotate bars with values
    for model_bars in bars:
        for bar in model_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    datasets_by_model = {}
    datasets_by_setup = {}
    for model in ['deepseek-chat', 'gemini-2.0-flash', 'llama3.3-70b']:
        datasets_by_model[model] = []
        for setup in ['0_interactions_noplan', '1_interactions_noplan', '1_interactions_plan']:
            if setup not in datasets_by_setup:
                datasets_by_setup[setup] = []
            data = json.load(open(f"merged_results/ai_interaction_results_{model}_{setup}.json"))
            datasets_by_setup[setup].append(data)
            datasets_by_model[model].append(data)
    plot_acceptance_rates_grouped_by_setup(datasets_by_model)
    plot_avg_accept_rates_by_scenario_and_setup(datasets_by_model)