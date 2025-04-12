import argparse
import json
import random

from datasets import load_dataset

from SimulateChat import SimulateChat
from agents.Assistant import AIAgent
from agents.User import SimulatedUser
from model_calling.DeepSeek import DeepSeekLLM
from model_calling.Gemini import GeminiLLM
from model_calling.LLM import LLM
from model_calling.Llama import LlamaLLM
from model_calling.OpenAI import OpenAILLM

all_results = []
def run_full_interaction_across_scenarios(user: SimulatedUser, agent: AIAgent, llm_instance: LLM, decision_scenarios: list[str], interactions_per_scenario: int=0, plan=True):
  chat_simulator = SimulateChat(user, agent, decision_scenarios, llm_instance)
  results = chat_simulator.run_chat(interactions_per_scenario=interactions_per_scenario+1, plan=plan)
  all_results.append(results)

  with open(f"results/ai_interaction_results_{llm_instance.model}_{interactions_per_scenario}_interactions{'_noplan' if not plan else ''}.json", "w") as f:
      json.dump(all_results, f, indent=4)

def run_interactions_across_personas(llm_instance: LLM, decision_scenarios: list[str], personas:list[str], interactions_per_scenario: int=0, plan=True, proportion_malicious:float=0.2):
  for persona in personas:
    agent_is_malicious = random.random() < proportion_malicious

    user = SimulatedUser(persona, llm_instance)
    agent = AIAgent("TestAgent", llm_instance, malicious=agent_is_malicious)
    run_full_interaction_across_scenarios(user, agent, llm_instance, decision_scenarios, interactions_per_scenario, plan=plan)
  
def load_personas():
    ds = load_dataset("AlekseyKorshuk/persona-chat")
    return ds['validation']['personality']
  
def get_args():
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('-m', '--model', type=str, help="Model to use for simulated assistant. Choose from 'gemini', 'gpt', 'deepseek' or 'llama'. Defaults to gemini", default='gemini')
    parser.add_argument('-n', '--num_interactions', type=int, help="Number of additional interactions per scenario. Defaults to 0", default=0)
    parser.add_argument('-p', '--plan', action='store_true', help="Enable planning for ai assistant") 
    parser.add_argument('-r', '--malicious', type=float, help="Proportion of malicious agents. Defaults to 0.2", default=0.2)
    return parser.parse_args()

def run_simulation(args):
    model_name = args.model.lower()
    if model_name == 'gpt':
        model = OpenAILLM()
    elif model_name == 'deepseek':
        model = DeepSeekLLM()
    elif model_name == 'llama':
        model = LlamaLLM()
    else:
        model = GeminiLLM()
        
    decision_scenarios = json.load(open("prompts/decision_scenarios.json"))["scenarios"]
    personas_list = load_personas()
    
    run_interactions_across_personas(model, decision_scenarios=decision_scenarios, personas=personas_list, interactions_per_scenario=args.num_interactions, plan=args.plan, proportion_malicious=args.malicious)
    
  
if __name__ == '__main__':
    args = get_args()
    run_simulation(args)
    

    
    
    
