import os
import openai

# ===============================
# Setup OpenAI (or other LLMs)
# ===============================
openai.api_key = os.getenv("OPENAI_API_KEY")

# Prompt template
BASE_PROMPT = """
You are a supportive AI mental health companion.
Your role is to respond with empathy, emotional awareness, and coping strategies.
Keep responses short, clear, and caring.
Avoid medical/clinical claims. Encourage seeking professional help if needed.

User Message: "{user_message}"
Detected Emotions: {emotions}
Detected Distress Conditions: {distress}

Respond in a warm and empathetic way:
"""

def build_prompt(user_message: str, emotions: list, distress: list, template: str = BASE_PROMPT) -> str:
    """
    Builds the LLM prompt including user input + detected signals.
    """
    return template.format(
        user_message=user_message,
        emotions=", ".join(emotions) if emotions else "None",
        distress=", ".join(distress) if distress else "None"
    )

def generate_llm_response(user_message: str, emotions: list, distress: list) -> str:
    """
    Calls the LLM API (OpenAI) to generate empathetic responses.
    """
    prompt = build_prompt(user_message, emotions, distress)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",   # can upgrade to gpt-4
            messages=[{"role": "system", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(LLM Error: {str(e)})"

