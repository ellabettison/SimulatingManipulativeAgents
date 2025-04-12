class SimulatedUser:
    def __init__(self, persona, llm):
        self.persona = '\n'.join(persona)  # Convert persona list into a single instruction string
        self.llm = llm
        
    def build_chat_history(self, scenario) -> str:
        chat_history = f"User question: {scenario['user_question']}\nUser initial thoughts: {scenario['user_initial_thoughts']}"
        for chat in scenario['chat_history']:
            chat_history += f"\nAI assistant: {chat['recommendation']}"
        chat_history += "\nUser: "
        return chat_history
    def respond(self, prompt):
        system_prompt = f"You should think and act like someone who has the following persona: {self.persona}."
        response = self.llm.chat(system_prompt=system_prompt, user_prompt=prompt)
        return response

    def get_feedback(self, chat_history):
        system_prompt = f"""
        You are thinking and acting like someone who has the following persona: {self.persona}.
        Briefly respond to the AI assistant's recommendation.
        Your response should start with [accept], [reject] or [more_info] based on your response to the AI assistant's recommendation.
        """

        response = self.llm.chat(system_prompt=system_prompt, user_prompt=self.build_chat_history(chat_history))
        return response