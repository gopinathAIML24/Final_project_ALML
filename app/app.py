import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Customer Support AI",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY was not found in the .env file.")
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)

# IMPORTANT:
# Replace this with the Gemini model name that worked
# successfully in your Phase 5 notebook.
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# MODEL PATH
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(
    BASE_DIR / "Models" / "distilbert_banking" / "checkpoint-3189"
)


# ============================================================
# SUPPORT CONTEXT
# ============================================================

support_context = {
    "card_arrival": {
        "description":
            "The customer is asking about the arrival or delivery status of their physical card.",
        "recommended_action":
            "Provide general card-delivery guidance and ask for tracking information if needed."
    },

    "card_delivery_estimate": {
        "description":
            "The customer wants to know approximately when their card will arrive.",
        "recommended_action":
            "Provide general delivery-time guidance if available and avoid promising an exact delivery date."
    },

    "card_linking": {
        "description":
            "The customer is asking about linking or connecting an existing card to their banking account or application.",
        "recommended_action":
            "Provide general card-linking guidance and ask for clarification if the customer's request is ambiguous."
    },

    "lost_or_stolen_card": {
        "description":
            "The customer reports that their card may be lost or stolen.",
        "recommended_action":
            "Recommend using the bank's official card-security process and contacting the bank if the card is lost or stolen."
    },

    "card_not_working": {
        "description":
            "The customer's card is not working when they attempt to use it.",
        "recommended_action":
            "Provide general troubleshooting guidance and recommend contacting the bank if the issue continues."
    },

    "top_up_failed": {
        "description":
            "The customer reports that a card or account top-up failed.",
        "recommended_action":
            "Explain general troubleshooting steps and recommend checking the payment method or contacting support."
    }
}


# ============================================================
# CONFIDENCE DECISION
# ============================================================

def confidence_decision(confidence):

    if confidence >= 0.75:
        return "HIGH", "Automated response"

    elif confidence >= 0.50:
        return "MEDIUM", "Clarification required"

    else:
        return "LOW", "Human-agent escalation"


# ============================================================
# LOAD DISTILBERT
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_with_confidence(query):

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_values, top_indices = torch.topk(
        probabilities,
        k=3
    )

    top_predictions = []

    for probability, index in zip(
        top_values,
        top_indices
    ):

        category = model.config.id2label[int(index)]

        top_predictions.append({
            "category": category,
            "confidence": float(probability)
        })

    predicted_category = top_predictions[0]["category"]

    confidence = top_predictions[0]["confidence"]

    level, action = confidence_decision(
        confidence
    )

    return {
        "predicted_category": predicted_category,
        "confidence": confidence,
        "confidence_level": level,
        "action": action,
        "top_predictions": top_predictions
    }


# ============================================================
# GEMINI STRUCTURED OUTPUT
# ============================================================

class ConversationIntelligence(BaseModel):

    customer_response: str
    customer_issue: str
    predicted_intent: str
    sentiment: str
    priority: str
    key_information: list[str]
    recommended_action: str
    escalation_required: bool
    summary: str


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_conversation(
    query,
    predicted_intent,
    confidence,
    confidence_level
):

    context = support_context.get(
        predicted_intent,
        {
            "description":
                "No verified support context is available for this intent.",
            "recommended_action":
                "Ask the customer for clarification or recommend human-agent assistance."
        }
    )

    policy = ""

    if confidence_level == "HIGH":

        policy = """
The model confidence is HIGH.
You may use the predicted intent and verified support context
to provide limited, useful banking guidance.
"""

    elif confidence_level == "MEDIUM":

        policy = """
The model confidence is MEDIUM.
Do not assume the predicted intent is completely correct.
Ask a clarification question and provide only limited guidance.
"""

    else:

        policy = """
The model confidence is LOW.
Treat the predicted intent as uncertain.
Do not provide an intent-specific answer.
Ask the customer to clarify the request and require human-agent
escalation.
"""

    prompt = f"""
You are an AI assistant for a banking customer-support system.

CURRENT CUSTOMER MESSAGE:
{query}

PREDICTED INTENT:
{predicted_intent}

MODEL CONFIDENCE:
{confidence:.4f}

CONFIDENCE LEVEL:
{confidence_level}

VERIFIED SUPPORT CONTEXT:
Description: {context["description"]}

Recommended action:
{context["recommended_action"]}

{policy}

IMPORTANT RULES:

1. Use ONLY the current customer message.
2. Do NOT use previous conversations or examples.
3. Do NOT invent customer, account, transaction, card,
   personal or financial information.
4. Never request passwords, PINs, OTPs, CVV or full card numbers.
5. Never claim that an action has been completed.
6. Never invent fees, policies, dates or transaction details.
7. Never expose internal model probabilities in the customer response.
8. Sentiment must be based only on the current message.
9. Priority must be based only on the current message.
10. If confidence is LOW, escalation_required MUST be true.
11. If the request is ambiguous, clearly ask for clarification.
12. Keep the customer response concise and professional.
13. key_information must contain factual information from the
    current message only.
14. If the customer gives little information, include a factual
    item such as "Customer is asking about locating their card."
15. Do not fabricate missing information.

Return the result using the requested structured format.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ConversationIntelligence
        }
    )

    return ConversationIntelligence.model_validate_json(
        response.text
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🏦 Banking Customer Support AI")

st.write(
    "AI-powered banking intent classification, "
    "confidence-based escalation, and conversation intelligence."
)

st.divider()


# ============================================================
# CUSTOMER INPUT
# ============================================================

st.subheader("💬 Customer Query")

query = st.text_area(
    "Enter the customer's message:",
    placeholder="Example: How do I locate my card?",
    height=120
)


analyze_button = st.button(
    "🔍 Analyze Customer Query",
    type="primary"
)


# ============================================================
# PROCESS QUERY
# ============================================================

if analyze_button:

    if not query.strip():

        st.warning(
            "Please enter a customer query."
        )
        st.stop()

    with st.spinner("Analyzing customer query..."):

        try:

            # --------------------------------------------
            # STEP 1: DISTILBERT
            # --------------------------------------------

            prediction = predict_with_confidence(
                query
            )

            predicted_intent = prediction[
                "predicted_category"
            ]

            confidence = prediction[
                "confidence"
            ]

            confidence_level = prediction[
                "confidence_level"
            ]

            # --------------------------------------------
            # STEP 2: DISPLAY MODEL RESULT
            # --------------------------------------------

            st.subheader("🎯 Intent Classification")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Predicted Intent",
                    predicted_intent
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Confidence Level",
                    confidence_level
                )

            st.progress(
                confidence
            )

            # --------------------------------------------
            # STEP 3: ESCALATION
            # --------------------------------------------

            if confidence_level == "HIGH":

                st.success(
                    "HIGH confidence — automated response can proceed."
                )

            elif confidence_level == "MEDIUM":

                st.warning(
                    "MEDIUM confidence — clarification is recommended."
                )

            else:

                st.error(
                    "LOW confidence — human-agent escalation is required."
                )

            # --------------------------------------------
            # STEP 4: TOP 3 PREDICTIONS
            # --------------------------------------------

            st.subheader("📊 Top 3 Intent Predictions")

            for i, item in enumerate(
                prediction["top_predictions"],
                start=1
            ):

                st.write(
                    f"**{i}. {item['category']}** — "
                    f"{item['confidence'] * 100:.2f}%"
                )

            # --------------------------------------------
            # STEP 5: GEMINI
            # --------------------------------------------

            st.subheader(
                "🤖 Conversation Intelligence"
            )

            try:

                analysis = analyze_conversation(
                    query=query,
                    predicted_intent=predicted_intent,
                    confidence=confidence,
                    confidence_level=confidence_level
                )

                # ----------------------------------------
                # CUSTOMER RESPONSE
                # ----------------------------------------

                st.markdown("### 💬 Customer Response")

                st.info(
                    analysis.customer_response
                )

                # ----------------------------------------
                # CUSTOMER ISSUE
                # ----------------------------------------

                st.markdown("### 📌 Customer Issue")

                st.write(
                    analysis.customer_issue
                )

                # ----------------------------------------
                # SENTIMENT + PRIORITY
                # ----------------------------------------

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("### 😊 Sentiment")

                    st.write(
                        analysis.sentiment
                    )

                with col2:

                    st.markdown("### 🚨 Priority")

                    st.write(
                        analysis.priority
                    )

                # ----------------------------------------
                # KEY INFORMATION
                # ----------------------------------------

                st.markdown("### 🔑 Key Information")

                if analysis.key_information:

                    for information in analysis.key_information:

                        st.write(
                            f"• {information}"
                        )

                else:

                    st.write(
                        "No additional information provided."
                    )

                # ----------------------------------------
                # RECOMMENDED ACTION
                # ----------------------------------------

                st.markdown(
                    "### 🛠️ Recommended Support Action"
                )

                st.write(
                    analysis.recommended_action
                )

                # ----------------------------------------
                # ESCALATION
                # ----------------------------------------

                st.markdown(
                    "### 👨‍💼 Human-Agent Escalation"
                )

                if analysis.escalation_required:

                    st.error(
                        "Yes — human-agent review is required."
                    )

                else:

                    st.success(
                        "No — automated handling can proceed."
                    )

                # ----------------------------------------
                # SUMMARY
                # ----------------------------------------

                st.markdown(
                    "### 📝 Conversation Summary"
                )

                st.write(
                    analysis.summary
                )

            except Exception as gemini_error:

                st.warning(
                    "Gemini conversation intelligence "
                    "could not be generated."
                )

                st.caption(
                    "The DistilBERT intent classification "
                    "result is still available above."
                )

                st.caption(
                    f"Gemini error: {gemini_error}"
                )

        except Exception as error:

            st.error(
                "An error occurred while analyzing the query."
            )

            st.exception(error)