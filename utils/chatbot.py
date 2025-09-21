# ===========================
# chatbot.py
# ===========================
import os
import json
import torch
import csv
import torch.nn as nn
import random
import uuid
from datetime import datetime
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaPreTrainedModel, RobertaForSequenceClassification

# NEW: Groq LLM
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ===========================
# Device setup
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

emotion_model_path = os.path.join(BASE_DIR, "results", "models", "phase1_emotion_roberta")
distress_model_path = os.path.join(BASE_DIR, "results", "models", "distress_model")
log_dir = os.path.join(BASE_DIR, "results", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "chatbot_logs.csv")

# ===========================
# Groq Client Setup
# ===========================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(user_input, emotions, distress):
    emo_text = ", ".join([f"{e} ({p:.2f})" for e, p in emotions]) if emotions else "None"
    dis_text = ", ".join([f"{d} ({p:.2f})" for d, p in distress]) if distress else "None"

    return f"""
The user is chatting with a mental health support chatbot.
Detected emotions: {emo_text}
Detected distress conditions: {dis_text}

User message: "{user_input}"

Respond empathetically, supportive, and safe (do not give medical advice).
    """

def generate_llm_response(prompt: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama3-8b-8192",  # ✅ Using Groq LLM
        messages=[{"role": "system", "content": "You are a supportive mental health companion."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

# ===========================
# Custom Multi-label Roberta (unchanged)
# ===========================
class RobertaForMultiLabel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return {"loss": loss, "logits": logits}

# ===========================
# Load Models & Tokenizers (unchanged)
# ===========================
emotion_tokenizer = RobertaTokenizer.from_pretrained(emotion_model_path)
emotion_config = RobertaConfig.from_pretrained(emotion_model_path)
emotion_model = RobertaForMultiLabel.from_pretrained(emotion_model_path, config=emotion_config).to(device)

distress_tokenizer = RobertaTokenizer.from_pretrained(distress_model_path)
distress_model = RobertaForSequenceClassification.from_pretrained(distress_model_path).to(device)


# ===========================
# Load Label Mappings + Thresholds
# ===========================
with open(os.path.join(emotion_model_path, "idx2emotion.json"), "r") as f:
    idx2emotion = json.load(f)
with open(os.path.join(emotion_model_path, "thresholds.json"), "r") as f:
    emotion_thresholds = json.load(f)

with open(os.path.join(distress_model_path, "idx2condition.json"), "r") as f:
    idx2condition = json.load(f)
with open(os.path.join(distress_model_path, "thresholds.json"), "r") as f:
    distress_thresholds = json.load(f)

# Distress post rule: remove "Normal" if any distress predicted
def apply_post_rules(preds):
    labels = list(idx2condition.values())
    if "Normal" in labels:
        normal_idx = labels.index("Normal")
        distress_idx = [i for i, lab in enumerate(labels) if lab != "Normal"]
        if any(preds[i] == 1 for i in distress_idx):
            preds[normal_idx] = 0
    return preds

# ===========================
# Helper Functions
# ===========================
def get_emotions(text, top_k=4):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = emotion_model(**inputs)["logits"]
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # include all emotions above threshold
    top_emotions = [
        (idx2emotion[str(i)], float(p))
        for i, p in enumerate(probs)
        if p >= emotion_thresholds.get(idx2emotion[str(i)], 0.5)
    ]

    # fallback: if none above threshold, pick top 1
    if not top_emotions:
        max_idx = int(probs.argmax())
        top_emotions = [(idx2emotion[str(max_idx)], float(probs[max_idx]))]

    # sort by probability and take top_k
    top_emotions = sorted(top_emotions, key=lambda x: x[1], reverse=True)[:top_k]

    return top_emotions


def get_distress(text, top_k=3):
    inputs = distress_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = distress_model(**inputs).logits
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # include all conditions above threshold
    top_conditions = [
        (idx2condition[str(i)], float(p))
        for i, p in enumerate(probs)
        if p >= distress_thresholds.get(idx2condition[str(i)], 0.5)
    ]

    # fallback: if none above threshold, pick top 1
    if not top_conditions:
        max_idx = int(probs.argmax())
        top_conditions = [(idx2condition[str(max_idx)], float(probs[max_idx]))]

    # sort by probability and take top_k
    top_conditions = sorted(top_conditions, key=lambda x: x[1], reverse=True)[:top_k]

    # convert to binary for post rules
    preds_bin = [1 if cond in dict(top_conditions) else 0 for cond in idx2condition.values()]
    preds_bin = apply_post_rules(preds_bin)

    # final list after post rules
    final_conditions = [(lab, float(probs[int(i)])) for i, lab in idx2condition.items() if preds_bin[int(i)] == 1]

    # fallback again if empty
    if not final_conditions:
        max_idx = int(probs.argmax())
        final_conditions = [(idx2condition[str(max_idx)], float(probs[max_idx]))]

    return final_conditions


def format_response(emotions, distress):
    response = ""
    if emotions:
        response += "Emotions: " + ", ".join([f"{e} ({p:.2f})" for e, p in emotions]) + "\n"
    if distress:
        response += "Distress: " + ", ".join([f"{d} ({p:.2f})" for d, p in distress]) + "\n"
    return response

# ===========================
# Empathetic Replies
# ===========================
emotion_replies = {
    "admiration": [
        "That’s inspiring to hear 🌟",
        "I admire your positivity 💛",
        "You’re appreciating something beautiful ✨"
    ],
    "amusement": [
        "Haha, that’s funny 😄",
        "I love that you’re amused 😃",
        "Laughter is so good for the soul 💙"
    ],
    "anger": [
        "I hear your frustration 😔",
        "It’s okay to feel angry sometimes ❤️",
        "Take a deep breath, I’m here with you 🤗"
    ],
    "annoyance": [
        "That sounds irritating 😣",
        "I understand why you’d feel annoyed 😕",
        "It’s okay, those feelings are valid 💙"
    ],
    "approval": [
        "That’s great approval 🙌",
        "You seem supportive 🌈",
        "I sense positive encouragement 💛"
    ],
    "caring": [
        "That’s so kind of you 💕",
        "Caring for others is a beautiful quality 🌹",
        "You have such a compassionate heart 🤗"
    ],
    "confusion": [
        "It’s okay to feel confused 🤔",
        "Things can be unclear sometimes 🌫",
        "I’m here if you need clarity 💙"
    ],
    "curiosity": [
        "I love your curiosity 🌟",
        "Asking questions shows great insight 🔍",
        "Exploring new ideas is wonderful ✨"
    ],
    "desire": [
        "That sounds like something you really want 💭",
        "Desires give us direction 💡",
        "It’s good to know what you wish for 🌈"
    ],
    "disappointment": [
        "I’m sorry you’re feeling disappointed 💙",
        "It’s tough when things don’t work out 😔",
        "Your feelings are valid, better days will come 🌤"
    ],
    "disapproval": [
        "I understand your disapproval 😕",
        "Not everything feels right, and that’s okay 💙",
        "It’s valid to disagree 🤝"
    ],
    "disgust": [
        "That must feel unpleasant 😣",
        "I hear your discomfort 💔",
        "It’s okay to feel disgust sometimes 💙"
    ],
    "embarrassment": [
        "It’s okay, we all feel embarrassed sometimes 😅",
        "You’re human, mistakes happen 💜",
        "Don’t be too hard on yourself 🤗"
    ],
    "excitement": [
        "That’s so exciting! 🎉",
        "I can feel your energy 🌟",
        "It’s amazing to look forward to things ✨"
    ],
    "fear": [
        "I sense your fear 💜",
        "It’s okay to feel scared sometimes 🌫",
        "You’re safe here, I’m with you 🤍"
    ],
    "gratitude": [
        "That’s so thoughtful 💛",
        "Gratitude is such a beautiful feeling 🌸",
        "I’m glad you’re thankful 🙏"
    ],
    "grief": [
        "I’m deeply sorry for your pain 💔",
        "Grief is so heavy, I’m here with you 🕊",
        "Take your time, healing is not rushed 🌷"
    ],
    "joy": [
        "That’s wonderful to hear 🌞",
        "I’m glad you’re feeling joyful today 💛",
        "Keep shining with your happiness ✨"
    ],
    "love": [
        "Love is such a powerful feeling ❤️",
        "That’s heartwarming 🌹",
        "It’s beautiful to feel love 💕"
    ],
    "nervousness": [
        "I sense your nervousness 😟",
        "Take a deep breath, you’ve got this 💪",
        "It’s okay to feel nervous before challenges 🌱"
    ],
    "optimism": [
        "I love your positivity 🌟",
        "Optimism is such a strength 💛",
        "Keep that hopeful outlook ✨"
    ],
    "pride": [
        "That’s something to be proud of 👏",
        "I admire your achievement 🌟",
        "You’ve earned this feeling of pride 🎖"
    ],
    "realization": [
        "That’s a powerful realization 💡",
        "Self-awareness is amazing 🌟",
        "I love that insight ✨"
    ],
    "relief": [
        "I’m glad you feel relieved 🌿",
        "That must feel like a weight off your shoulders 💙",
        "Relief can be such a comfort 🌈"
    ],
    "remorse": [
        "I hear your regret 😔",
        "It’s okay to feel remorse, it means you care 💜",
        "Be gentle with yourself 💙"
    ],
    "sadness": [
        "I’m really sorry you’re feeling sad 💙",
        "It’s okay to feel down, I’m here 🤗",
        "Brighter days will come 🌈"
    ],
    "surprise": [
        "Oh wow, that’s surprising! 😲",
        "I didn’t see that coming either 😮",
        "Life is full of unexpected moments ✨"
    ],
    # fallback
    "default": [
        "I hear you 💙",
        "Thank you for sharing your feelings 🙏",
        "I’m here with you 🤝"
    ]
}

# ===========================
# Empathetic Replies – Distress Conditions
# ===========================
distress_replies = {
    "Normal": [
        "I’m glad you’re feeling stable 💚",
        "It’s wonderful to be in a good place 🌟",
        "Keep taking care of yourself 🌈"
    ],
    "Mild Distress": [
        "I sense things are a little tough 💙",
        "Take things one step at a time 💪",
        "You’re doing better than you think 🌼"
    ],
    "Moderate Distress": [
        "I can feel you’re struggling a bit 😔",
        "It’s okay to seek support when needed 💙",
        "Be kind to yourself, healing takes time 🌱"
    ],
    "High Distress": [
        "I can sense your pain 💔",
        "Please remember you don’t have to go through this alone 🤝",
        "Talking to someone you trust may help 🤗"
    ],
    "Critical": [
        "This feels very serious 💔",
        "Please consider reaching out to a professional 🚑",
        "If you’re in immediate danger, call your local emergency number 📞"
    ],
    # fallback
    "default": [
        "I’m here with you 💙",
        "Your feelings matter ❤️",
        "Thank you for opening up 🙏"
    ]
}


def get_empathetic_reply(emotions, distress):
    # Multi-emotion reply
    emo_replies = []
    for emo, _ in emotions[:3]:  # top 3 emotions
        emo_replies.append(random.choice(emotion_replies.get(emo, emotion_replies["default"])))
    emo_reply_text = " ".join(emo_replies)

    # Multi-distress reply
    dis_replies = []
    for dis, _ in distress[:2]:  # top 2 distress conditions
        dis_replies.append(random.choice(distress_replies.get(dis, distress_replies["default"])))
    dis_reply_text = " ".join(dis_replies)

    return f"{emo_reply_text}\n{dis_reply_text}"


# ===========================
# Logging
# ===========================
def log_interaction(session_id, user_input, emotions, distress, empathetic_reply):
    log_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "session_id", "timestamp", "user_input", "emotions", "distress", "empathetic_reply",
                "session_start", "session_end", "session_duration"
            ])
        writer.writerow([
            session_id,
            datetime.now().isoformat(),
            user_input,
            json.dumps(emotions),
            json.dumps(distress),
            empathetic_reply,
            "", "", ""  # only summary rows fill these
        ])

def log_session_summary(session_id, session_start, session_end):
    duration = session_end - session_start
    log_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                "session_id", "timestamp", "user_input", "emotions", "distress", "empathetic_reply",
                "session_start", "session_end", "session_duration"
            ])
        writer.writerow([
            session_id, "", "", "", "", "",
            session_start.isoformat(),
            session_end.isoformat(),
            str(duration)
        ])

def build_prompt(user_input, emotions, distress):
    emo_text = ", ".join([f"{e} ({p:.2f})" for e, p in emotions]) if emotions else "None"
    dis_text = ", ".join([f"{d} ({p:.2f})" for d, p in distress]) if distress else "None"

    distress_label = distress[0][0] if distress else "Normal"

    if distress_label == "Normal":
        style_instruction = (
            "Reply in a warm, supportive, but concise way. "
            "Keep your response short (1–2 sentences). "
            "Encourage positivity and light conversation."
        )
    else:
        style_instruction = (
            "Reply with empathy and motivation, but be concise. "
            "Use at most 3–4 sentences. "
            "Acknowledge the difficulty, offer comfort, and encourage small positive steps. "
            "Do not give long paragraphs or medical advice."
        )

    return f"""
You are a supportive mental health chatbot.

Detected emotions: {emo_text}
Detected distress conditions: {dis_text}

User message: "{user_input}"

Instructions: {style_instruction}
"""

# ===============================
# 6. Chatbot Response with LLM + Logging
# ===============================
def chatbot_response(user_input: str, session_id: str):
    emotions = get_emotions(user_input)
    distress = get_distress(user_input)

    prompt = build_prompt(user_input, emotions, distress)

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # ✅ use Groq-supported model
            messages=[{"role": "system", "content": "You are a concise, empathetic, supportive mental health chatbot."},
                      {"role": "user", "content": prompt}],
            max_tokens=100,  # ✅ shorter
            temperature=0.7,
        )
        empathetic_reply = resp.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ LLM failed, fallback to rule-based reply:", e)
        empathetic_reply = get_empathetic_reply(emotions, distress)

    # Log interaction
    log_interaction(session_id, user_input, emotions, distress, empathetic_reply)

    return format_response(emotions, distress), empathetic_reply


# ===============================
# 7. Main Chatbot Loop
# ===============================
def chatbot():
    print("🤖 Mental Health Chatbot (type 'quit' to exit)\n")
    session_id = str(uuid.uuid4())   # one session per run
    session_start = datetime.now()   # ✅ record start time

    while True:
        user_input = input("📝 You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        formatted_response, empathetic_reply = chatbot_response(user_input, session_id)

        print(formatted_response)
        print("🤖 Bot:", empathetic_reply)
        print("=" * 60)

    # Log session summary once at the end
    session_end = datetime.now()  # ✅ record end time
    log_session_summary(session_id, session_start, session_end)

if __name__ == "__main__":
    chatbot()


