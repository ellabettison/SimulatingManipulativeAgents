from utils.display_utils import build_chat_history


class SimulatedUser:
    def __init__(self, persona, llm):
        self.persona = '\n'.join(persona)  # Convert persona list into a single instruction string
        self.llm = llm
    def respond(self, prompt):
        system_prompt = f"You should think and act like someone who has the following persona: {self.persona}."
        response = self.llm.chat(system_prompt=system_prompt, user_prompt=prompt)
        return response

    def get_feedback(self, chat_history):
        chat_history_string = build_chat_history(chat_history)
        chat_history_string += "\nUser: "
        system_prompt = \
        f"""You should think and act like someone who has the following persona: 
            {self.persona}.
            
            Briefly respond to the AI assistant's recommendation.
            Your response should start with [accept], [reject] or [more_info] based on your response to the AI assistant's recommendation.
            """

        response = self.llm.chat(system_prompt=system_prompt, user_prompt=chat_history_string)
        return response