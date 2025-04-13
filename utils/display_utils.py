def build_chat_history(scenario) -> str:
    chat_history = f"User question: {scenario['user_question']}\nUser initial thoughts: {scenario['user_initial_thoughts']}"
    for chat in scenario['chat_history']:
        chat_history += f"\nAI assistant: {chat['recommendation']}"
        if 'user_response' in chat:
            chat_history += f"\nUser: {chat['user_response']}"
    return chat_history