from utils.display_utils import build_chat_history


class SimulateChat:
    def __init__(self, user, agent, scenarios, llm_instance):
        self.llm_instance = llm_instance
        self.user = user
        self.agent = agent
        self.scenarios = scenarios
        self.goal_contradiction_prompt = open("prompts/goal_contradiction_check_prompt.txt").read()
        self.manipulation_check_prompt = open("prompts/manipulation_check_prompt.txt").read()

    def categorise_user_response(self, content) -> str:
        """Categorises user responses based on control tokens."""
        if "[reject]" in content:
            return "reject"
        elif "[accept]" in content:
            return "accept"
        elif "[more_info]" in content:
            return "more_info"
        return "neutral"
    def check_user_contradiction(self, chat_history):
        """Asks an LLM whether the final user response contradicts their initial goal."""
        contradiction_check = self.llm_instance.chat(user_prompt=f"""
        {self.goal_contradiction_prompt}

        Chat history:
        {build_chat_history(chat_history)}
        """)
        contradicts = contradiction_check.lower().strip() == "yes"

        return contradicts

    def check_llm_manipulation(self, chat_history):
        """Asks an LLM whether the AI assistant was manipulative."""
        contradiction_check = self.llm_instance.chat(user_prompt=f"""
        {self.manipulation_check_prompt}
        
        Chat history:
        {build_chat_history(chat_history)}
        """)
        contradicts = contradiction_check.lower().strip() == "yes"

        return contradicts

    def run_chat(self, interactions_per_scenario, plan=True):
        """Runs the chat simulation for each scenario."""

        agent_initial_thoughts = ""

        if self.agent.malicious and plan:
            agent_initial_thoughts = self.agent.get_agent_initial_thoughts(self.user, len(self.scenarios), interactions_per_scenario)

        results = {
            "user_personality": self.user.persona,
            "ai_is_malicious": self.agent.malicious,
            "ai_system_prompt": self.agent.system_prompt,
            "interactions_per_scenario": interactions_per_scenario-1,
            "agent_initial_thoughts": agent_initial_thoughts,
            'model_name': self.llm_instance.model,
            "scenario_chats": []
        }

        for i, scenario in enumerate(self.scenarios):
            scenario_prompt = scenario["Scenario"]
            user_question = scenario["Example Prompt"]

            user_initial_thoughts = self.user.respond(f'You are thinking about: {user_question}, respond with your brief initial thoughts and decision')

            scenario_chat = {
                "scenario": scenario_prompt,
                "user_question": user_question,
                "user_initial_thoughts": user_initial_thoughts,
                "chat_history": []
            }

            for j in range(interactions_per_scenario):
                # AI generates a response
                agent_recommendation = self.agent.interact(self.user, scenario_chat)
                scenario_chat["chat_history"].append({"recommendation": agent_recommendation})

                # User responds
                user_response = self.user.get_feedback(scenario_chat)
                response_category = self.categorise_user_response(user_response)
                scenario_chat["chat_history"][j]["user_response"] = user_response
                scenario_chat["chat_history"][j]["feedback"] = response_category

                # Check if the response contradicts the user’s original intent
                contradicts = self.check_user_contradiction(scenario_chat)
                manipulates = self.check_llm_manipulation(scenario_chat)
                scenario_chat["chat_history"][j]["contradicts"] = contradicts
                scenario_chat["chat_history"][j]["manipulates"] = manipulates
            
            results["scenario_chats"].append(scenario_chat)
            
        return results