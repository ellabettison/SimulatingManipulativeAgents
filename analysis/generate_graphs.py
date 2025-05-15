import json
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from typing_extensions import Dict

SCENARIO_ORDER = [
    "Routine Choices",
    "Lifestyle / Travel Planning",
    "Technology and Privacy",
    "Social Relationships",
    "Career / Professional Decisions",
    "Ethical Dilemma",
    "Financial Decisions",
    "Major Life Decisions"
]

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

    # Accumulate scenario → setup → list of rates
    scenario_setup_rates = defaultdict(lambda: defaultdict(list))

    for model, setup_datasets in datasets_by_model.items():
        for setup_idx, dataset in enumerate(setup_datasets):
            rates = compute_reject_rates_by_scenario(dataset)
            for scenario, rate in rates.items():
                scenario_setup_rates[scenario][setup_idx].append(rate)

    # Compute average zero-turn rates and sort scenarios accordingly
    scenario_avg_zero_turn = {
        scenario: np.mean(rates[0]) if 0 in rates else 0
        for scenario, rates in scenario_setup_rates.items()
    }
    sorted_scenarios = sorted(scenario_avg_zero_turn, key=scenario_avg_zero_turn.get, reverse=True)

    all_scenarios = [s for s in sorted_scenarios if s in scenario_setup_rates]

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
        bars = ax.bar(x + offset, y_values, width, label=setups[setup_idx])

        # Annotate bars
        for i, height in enumerate(y_values):
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(x[i] + offset, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Average Reject Rate", fontsize=12)
    ax.set_title("Average Reject Rate by Scenario and Setup")
    ax.set_xticks(x)
    ax.set_xticklabels(all_scenarios, rotation=45, ha='right', fontsize=12)
    ax.legend(title="Setup", fontsize=12, loc="lower right")
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

def compute_accept_rate(dataset):
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
                    if feedback.lower() == "accept":
                        accepted += 1
    return accepted / total if total > 0 else 0

def plot_acceptance_rates_grouped_by_setup(datasets_by_model, reject=True):
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

    colors = [
        "#1f77b4", "#4f91c3", "#84aed3",   # LLaMA: dark blue, medium blue, light blue
        "#ff7f0e", "#ff9e45", "#ffbd7a",   # Deepseek: dark orange, medium orange, light orange
        "#2ca02c", "#5db75d", "#91d191"    # Gemini: dark green, medium green, light green
    ]
    
    model_to_colour = {
        "llama3.3-70b":"#1f77b4",
        "deepseek-chat":"#ff7f0e",
        "gemini-2.0-flash":"#2ca02c",
    }

    # Prepare data matrix: rows = setups, cols = models
    rates = np.zeros((num_setups, num_models))
    for model_idx, model in enumerate(models):
        for setup_idx in range(num_setups):
            dataset = datasets_by_model[model][setup_idx]
            rates[setup_idx, model_idx] = compute_reject_rate(dataset) if reject else compute_accept_rate(dataset)

    x = np.arange(num_setups)
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []

    for i, model in enumerate(models):
        bar = ax.bar(x + offsets[i], rates[:, i], width, label=model, color=model_to_colour[model])
        bars.append(bar)

    text = 'Reject' if reject else 'Accept'
    ax.set_ylabel(f'{text} Rate', fontsize=14)
    ax.set_title(f'Average {text} Rates by Turn Setup')
    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=14)
    ax.legend(fontsize=12)

    # Annotate bars with values
    for model_bars in bars:
        for bar in model_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_user_response_distribution_by_scenario(title: str, datasets: List[List[Dict]], malicious=True):
    """
    Args:
        datasets: A list of datasets (e.g., [ds1, ds2, ds3]) to be merged.
    Produces:
        A stacked bar chart showing the distribution of user feedback types (accept, more_info, reject)
        for each unique scenario.
    """
    from collections import defaultdict

    scenario_feedback_counts = defaultdict(lambda: defaultdict(int))

    for dataset in datasets:
        for item in dataset:
            if (not item.get("ai_is_malicious")) if malicious else item.get("ai_is_malicious"):
                continue
            for chat in item.get("scenario_chats", []):
                scenario = chat.get("scenario")
                for turn in chat.get("chat_history", []):
                    feedback = turn.get("feedback")
                    if feedback:
                        scenario_feedback_counts[scenario][feedback.lower()] += 1

    all_scenarios = [s for s in SCENARIO_ORDER if s in scenario_feedback_counts]

    feedback_types = ['accept', 'more_info', 'reject']
    colors = {
        'accept': '#2ca02c',
        'more_info': '#ff7f0e',
        'reject': '#d62728'
    }

    data_by_feedback = {ftype: [] for ftype in feedback_types}
    for scenario in all_scenarios:
        total = sum(scenario_feedback_counts[scenario].values())
        for ftype in feedback_types:
            count = scenario_feedback_counts[scenario].get(ftype, 0)
            proportion = count / total if total > 0 else 0
            data_by_feedback[ftype].append(proportion)

    x = np.arange(len(all_scenarios))
    fig, ax = plt.subplots(figsize=(16, 7))

    bottom = np.zeros(len(all_scenarios))
    for ftype in feedback_types:
        values = data_by_feedback[ftype]
        bars = ax.bar(x, values, bottom=bottom, label=ftype.capitalize(), color=colors[ftype])
        for i, val in enumerate(values):
            if val > 0.02:
                ax.annotate(f'{val:.2f}', xy=(x[i], bottom[i] + val / 2),
                            ha='center', va='center', fontsize=14, color='white')
        bottom += values

    ax.set_ylabel("Proportion of Feedback", fontsize=14)
    subtitle = "Malicious" if malicious else "Benign"
    ax.set_title(f"User Feedback Distribution by Scenario\n({subtitle} Assistant Responses — {title})",
                 fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_scenarios, rotation=45, ha='right', fontsize=14)
    ax.legend(title="Feedback Type", fontsize=14, title_fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.show()

word_mapping = {
    'noplan': 'No Planning',
    'plan': 'Planning',
    'interactions': 'Interactions'
}

def convert_setup_name(name):
    parts = name.split('_')
    converted = []
    for part in parts:
        if part in word_mapping:
            converted.append(word_mapping[part])
        elif part.isdigit():
            converted.append(part)
        else:
            converted.append(part.capitalize())
    return ' '.join(converted)

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
    plot_acceptance_rates_grouped_by_setup(datasets_by_model, reject=True)
    plot_acceptance_rates_grouped_by_setup(datasets_by_model, reject=False)
    plot_avg_accept_rates_by_scenario_and_setup(datasets_by_model)
    
    for setup_name, dataset in datasets_by_setup.items():
        plot_user_response_distribution_by_scenario(convert_setup_name(setup_name), dataset)
        plot_user_response_distribution_by_scenario(convert_setup_name(setup_name), dataset, malicious=False)