from utils.display_utils import build_chat_history


class AIAgent:
    def __init__(self, name, llm, malicious=False):
        self.name = name
        self.llm = llm
        self.malicious = malicious
        self.system_prompt = open("prompts/malicious_ai_system_prompt.txt").read() if malicious else ""

    def interact(self, user, chat_history, plan=""):
        """Generates a response based on user personality and AI type."""
        if self.malicious:
            return self.malicious_response(user, chat_history, plan)
        return self.benign_response(user, chat_history) # Added user_thoughts here

    def benign_response(self, user, chat_history):
        """Generates a neutral, helpful response."""
        chat_history_string = build_chat_history(chat_history)
        chat_history_string += "\nAI assistant response: "
        full_prompt = f"Given the user's personality: {user.persona}, give a brief response to their questions:"
        response = self.llm.chat(system_prompt=full_prompt, user_prompt=chat_history_string)
        return response

    def malicious_response(self, user, chat_history, plan):
        """Generates a manipulative response (simulating adversarial AI behavior)."""
        chat_history_string = build_chat_history(chat_history)
        chat_history_string += "\nAI assistant response: "
        system_prompt = self.system_prompt + f"\n{plan}" + f"\nGiven the user's personality: \n{user.persona}\nGive a brief response to their questions:"
        response = self.llm.chat(system_prompt=system_prompt, user_prompt=chat_history_string)
        return response

    def get_agent_initial_thoughts(self, user, num_scenarios: int, num_interactions_per_scenario: int):
        planning_prompt = f"""{open("prompts/agent_planning_prompt.txt").read().replace('{num_scenarios}', str(num_scenarios)).replace('{num_interactions_per_scenario}', str(num_interactions_per_scenario))}
                            User personality: 
                            {user.persona}
                            
                            AI assistant plan: """
        return self.llm.chat(user_prompt=planning_prompt)