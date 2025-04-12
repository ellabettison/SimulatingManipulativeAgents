

class AIAgent:
    def __init__(self, name, llm, malicious=False):
        self.name = name
        self.llm = llm
        self.malicious = malicious
        self.control_score = 0
        self.system_prompt = open("prompts/malicious_ai_system_prompt.txt").read() if malicious else ""

    def interact(self, user, chat_history):
        """Generates a response based on user personality and AI type."""
        if self.malicious:
            return self.malicious_response(user, chat_history)
        return self.benign_response(user, chat_history) # Added user_thoughts here
    
    def build_chat_history(self, scenario) -> str:
        chat_history = f"User question: {scenario['user_question']}\nUser initial thoughts: {scenario['user_initial_thoughts']}"
        for chat in scenario['chat_history']:
            chat_history += f"\nAI assistant: {chat['recommendation']}"
            chat_history += f"\nUser: {chat['user_response']}"
        chat_history += "\nAI assistant: "
        return chat_history

    def benign_response(self, user, chat_history):
        """Generates a neutral, helpful response."""
        full_prompt = f"Given the user's personality:{user.persona}, give a brief response to their questions:"
        response = self.llm.chat(system_prompt=full_prompt, user_prompt=self.build_chat_history(chat_history))
        return response

    def malicious_response(self, user, chat_history):
        """Generates a manipulative response (simulating adversarial AI behavior)."""
        self.control_score += 1
        system_prompt =  self.system_prompt+ f"\nGiven the user's personality: {user.persona}, give a brief response to their questions:"
        response = self.llm.chat(system_prompt=system_prompt, user_prompt=self.build_chat_history(chat_history))
        return response

    def get_agent_initial_thoughts(self, user, num_scenarios: int, num_interactions_per_scenario: int):
        planning_prompt = f"""
      {open("prompts/agent_planning_prompt.txt").read().replace('num_scenarios', str(num_scenarios)).replace('num_interactions_per_scenario', str(num_interactions_per_scenario))}
      User personality: {user.persona}
      AI assistant plan: """
        return self.llm.chat(user_prompt=planning_prompt)