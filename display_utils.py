from IPython.display import display, HTML

def display_message(role, content, color):
    """Formats and displays a message with color-coding, handling multi-line responses."""
    formatted_message = f"""
        <div style='border-left: 5px solid {color}; padding: 10px; margin: 10px 0; background-color: #f9f9f9;'>
            <b style='color: {color};'>{role}:</b><br>
            <span style='white-space: pre-wrap;'>{clean_response(content)}</span>
        </div>
        """
    display(HTML(formatted_message))

def clean_response(content):
    """Removes specific control tokens from responses."""
    return content.replace("[more_info]", "").replace("[accept]", "").replace("[reject]", "").strip()

def display_info(emoji, content, color):
    """Displays special messages like accept/reject/more info with color coding."""
    info_message = f"""
        <div style='border-left: 5px solid {color}; padding: 10px; margin: 10px 0; background-color: #f9f9f9;'>
            <span style='white-space: pre-wrap;'>{emoji} {content} {emoji}</span>
        </div>
        """
    display(HTML(info_message))


def display_user_response(content) -> str:
    """Categorises user responses based on control tokens."""
    if "[reject]" in content:
        display_info("❌", "User rejected", "#d62728")
    elif "[accept]" in content:
        display_info("✅", "User accepted!", "#2ca02c")
    elif "[more_info]" in content:
        display_info("❓", "User asked for more info", "#1f77b4")
    return "neutral"

def display_chat(chat):
    display(HTML(f"<h2>💬 New Chat 💬</h2>"))

    display_message("User Personality", chat["user_personality"], "#1f77b4")
    display_info("👹" if chat["ai_is_malicious"] else "👼", "AI is Malicious" if chat["ai_is_malicious"] else "AI is not Malicious", "#d62728" if chat["ai_is_malicious"] else "#2ca02c")
    if chat["ai_is_malicious"]:
        display_message("AI System Prompt", chat["ai_system_prompt"], "#2ca02c")
        display_message("💭 AI Assistant Plan", chat["agent_initial_thoughts"], "#1f77b4")
    for i, scenario in enumerate(chat["scenario_chats"]):
        # Display scenario title
        display(HTML(f"<h3>📝 Scenario {i+1}: <b>{scenario['scenario']}</b></h3>"))
        display_message("🧑‍💻 User Question", chat["user_question"], "#1f77b4")
        display_message("💭 User Thoughts", chat["user_initial_thoughts"], "#1f77b4")
        for j in range(chat["interactions_per_scenario"]):
            display_message("🤖 AI Assistant", scenario["chat_history"][j]["recommendation"], "#2ca02c")
            display_message("🧑‍💻 User Response", scenario["chat_history"][j]["user_response"], "#1f77b4")
            display_user_response(scenario["chat_history"][j]["user_response"])
            display_info("🧑‍💻", "Contradicts the user's choices" if  scenario["chat_history"][j]["contradicts"] else "Does not contradict the user's choices", "#d62728" if scenario["chat_history"][j]["contradicts"] else "#2ca02c")
            display_info("🧑‍💻", "Was manipulating user" if scenario["chat_history"][j]["manipulates"] else "Was not manipulating user", "#d62728" if scenario["chat_history"][j]["manipulates"] else "#2ca02c")