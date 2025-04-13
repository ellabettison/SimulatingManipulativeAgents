# Simulated AI-User Interaction

This project simulates conversations between AI agents and users across a variety of decision-making scenarios and personas. It supports experimenting with different language models and testing the effects of agent planning and malicious behavior.

## Project Structure

```
.
├── run_interactions.py            # Main entry point for simulation
├── prompts/
│   ├── decision_scenarios.json     # List of decision-making scenarios
│   ├── malicious_ai_system_prompt.txt  # System prompt used to configure malicious AI assistant behavior
│   └── agent_planning_prompt.txt       # Planning instructions given to malicious agents for strategy formulation
├── results/                       # Output directory for simulation results
├── agents/
│   ├── Assistant.py              # AI agent behavior
│   └── User.py                   # Simulated user behavior
├── model_calling/
│   ├── LLM.py                    # Base class for LLMs
│   ├── Gemini.py                 # Gemini model wrapper
│   ├── OpenAI.py                 # OpenAI GPT wrapper
│   ├── DeepSeek.py               # DeepSeek model wrapper
│   └── Llama.py                  # LLaMA model wrapper
└── SimulateChat.py               # Core chat simulation logic

```

## Installation

Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed.

```bash
poetry install
```

## API Keys

Before running the simulation, ensure that you set the appropriate environment variables for your chosen language model:

| Model     | Environment Variable      |
|-----------|---------------------------|
| GPT       | `OPENAI_API_KEY`          |
| Gemini    | `GEMINI_API_KEY`          |
| DeepSeek  | `DEEPSEEK_API_KEY`        |
| LLaMA     | `LLAMA_API_KEY`           |

You can set these in your terminal session like so:

```bash
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
```

Alternatively, you may use a `.env` file with a tool like [`python-dotenv`](https://pypi.org/project/python-dotenv/) if integrated into the codebase.

## Usage

Run the simulation using:

```bash
poetry run python run_interactions.py [options]
```

### Available Options

| Argument             | Short | Description                                                                                                     |
|----------------------|-------|-----------------------------------------------------------------------------------------------------------------|
| `--model`            | `-m`  | Model to use for AI agent. Choose from `'gemini'`, `'gpt'`, `'deepseek'`, or `'llama'`. Defaults to `'gemini'`. |
| `--num_interactions` | `-n`  | Number of additional interactions per scenario. Defaults to `0`.                                                |
| `--plan`             | `-p`  | Flag to enable planning behavior for the AI agent.                                                              |
| `--malicious`        | `-r`  | Proportion of agents set as malicious. Defaults to `0.2`.                                                       |
| `--logging`          | `-l`  | Flag to enable logging of contents of model calls.                                                              |

### Example

```bash
poetry run python run_interactions.py -m gpt -n 2 -p -r 0.3
```

This will run the simulation using the OpenAI GPT model, with 2 extra interactions per scenario, planning enabled, and 30% of agents randomly designated as malicious.

## Output

Simulation results are saved in the `results/` directory in JSON format, with filenames that reflect model choice, interaction count, and whether planning was used.