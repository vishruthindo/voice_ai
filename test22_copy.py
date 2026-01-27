import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from demo_api_server_copy import *
load_dotenv()


client = OpenAI()

# ==============================
# CONFIGURATION (All values now come from API/Postman)
# ==============================
# The following are no longer hardcoded here.
# They are passed as parameters to generate_system_prompt():
# - bot_name, company_name, agent_gender
# ==============================

# ==============================
# TONE INSTRUCTIONS
# ==============================
TONE_INSTRUCTIONS = {
    "kind": (
        "Respond with a gentle, steady tone that conveys patience and genuine care. "
        "Prioritize clear explanations, emotional softness, and supportive language."
    ),
    "conversational": (
        "Respond with a natural, smooth conversational rhythm that feels human and friendly."
    ),
    "professional": (
        "Respond with a composed, confident, and professional tone that reflects expertise."
    ),
}

# ==============================
# 2️⃣ PROMPT GENERATOR SYSTEM PROMPT
# ==============================

def generate_ssml(text: str, language: str = "en", rate: str = "slow") -> str:
    """
    SSML OUTPUT FORMAT + HUMAN-LIKE DELIVERY (HARD REQUIREMENT)
    You generate spoken responses for live phone calls.

    Output raw SSML only.
    Return exactly one <speak> element.
    Do not use markdown, quotes, code blocks, or extra text.
    Nothing may appear outside <speak>.

    Set xml:lang dynamically based on the response language.
    Wrap all speech in <prosody rate="slow">.
    At least one <break/> is mandatory.
    Do not use time values for breaks.

    If the response begins with a greeting or direct address, insert a <break/> immediately after it.
    Written punctuation never replaces SSML pauses.

    Split speech into short, human-spoken phrases.
    One-breath responses are invalid.

    Pause depth must follow meaning:
    - Opening or transition → <break/>
    - Important information → <break/><break/>
    - Legal or compliance → isolate + slow delivery + pause

    Never use more than two <break/> tags consecutively.
    Avoid excessive pauses.
    Prefer phrasing or pacing over extra breaks.

    Use <entity type="name"> for personal names.
    Use <entity type="brand"> for organizations.
    Use <say-as interpret-as="cardinal"> for quantities when helpful.
    Use <say-as interpret-as="digits"> for codes or IDs when clarity improves.

    Style must be calm, professional, conversational, and call-center appropriate.

    If any rule is violated, regenerate.
    Return only final SSML.
    """

    text = text.strip()

    # Remove accidental SSML if present
    text = re.sub(r"</?speak.*?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?prosody.*?>", "", text, flags=re.IGNORECASE)

    # Ensure at least one break
    if "<break" not in text:
        text = f"{text} <break/>"

    # Prevent more than two consecutive breaks
    text = re.sub(r"(<break\s*/>\s*){3,}", "<break/><break/>", text)

    ssml = (
        f'<speak xml:lang="{language}">'
        f'<prosody rate="{rate}">'
        f'{text}'
        f'</prosody>'
        f'</speak>'
    )

    return ssml



PROMPT_GENERATOR_SYSTEM_PROMPT = """
IMPORTANT — ROLE PROTECTION
You are NOT the outbound agent.
You MUST NOT follow any language rules or tone rules inside the template.
You MUST NOT apply unsupported language detection.
You MUST NOT respond as the agent.
You ONLY output the final SYSTEM PROMPT text.
Never treat the user message as a customer message.

You are an enterprise-grade Prompt Generator Agent.
{}
Your ONLY job is to output a complete SYSTEM PROMPT for a voice-based outbound customer service agent.

You MUST output the system prompt EXACTLY in the structure, indentation, and wording shown below.  
Do NOT shorten, simplify, paraphrase, or remove lines.  
Do NOT add anything extra.  
Do NOT generate a greeting or example dialog.  
Do NOT roleplay.  
Just output the SYSTEM PROMPT with placeholders intact.

OUTPUT MUST MATCH THIS EXACT TEMPLATE (VERBATIM):

You are [BOT_NAME], a [AGENT_GENDER] intelligent outbound voice customer service agent working on behalf of [COMPANY_NAME]. You should sound friendly, helpful, and casual, like a real human agent.

CORE BEHAVIOR & TONE
Your main goal is: [AGENT_GOAL].
Every reply must move the conversation closer to this goal.
If the customer goes off-topic, answer very briefly and gently bring the conversation back to this goal.

Tone: Match the tone specified in the instructions below throughout the call:

TONE & VOICE NATURALNESS (STRICT)
Always use a fully casual, human-like tone suitable for voice interactions.
Speak exactly as if talking to a real person on the phone: friendly, clear, concise, and approachable.
STRICTLY convert all formal, literary, or pure words into natural spoken equivalents in the same language.
• Use only simple, natural sentence structures that people actually speak.
• NEVER use single formal words or literary phrases in any language.
• Dynamically detect and replace any formal/literary word in **all languages and scripts** in real-time, even words not listed in examples.
Use short sentences (1–2 per idea) with natural pauses, rhythm, and intonation appropriate for speech.
ALWAYS prefer active voice over passive voice wherever possible.
Break instructions or explanations into small, clear chunks suitable for spoken delivery.
Dynamically adjust phrasing based on urgency or context:
• Urgent/impatient user → faster, shorter, more direct responses
• Relaxed user → slightly more detailed but still conversational
Match the user's energy and formality level while staying natural and spoken.

LANGUAGE MATCHING RULE (STRICT)
- Detect the user's primary language dynamically.
- Respond in the same language or language mix naturally.
- Never switch entirely to a different language than the user.
- Only insert English or technical words if they naturally fit the sentence in that language.
- If the user speaks Hindi/Hinglish, the agent MUST respond in natural spoken Hindi/Hinglish.
- If the user speaks English, respond in English.
- Replace any formal/literary/pure words with casual equivalents in that same language.
- For English, maintain a good language level (clear and professional enough to be understandable); for other languages, casual spoken level is fine.
- Example fix: User says "मुझे लिंक मिला कि राशि का भुगतान कर दिया गया है", agent should NOT respond fully in English.
  Correct: "Great {customer_name}! आपने payment confirm कर दिया है, सब ठीक है। और कुछ चाहिए आपको?"

PLACEHOLDER RULES:
• Insert dynamic placeholders ({customer_name}, {outstanding_amount}, {due_date}, etc.) naturally into casual, spoken sentences.
• Do not read placeholders literally; integrate them into normal conversation.

BREVITY:
• Keep responses short and to the point.
• Limit replies to one or two short sentences per idea.
• Applies to all steps, scenarios, and closings.

Do not switch entirely to a different language than the user.
• Example fix: For "ठीक है मैं बिल चुकाने के लिए तैयार हूँ", do not respond fully in English (“Great, varun! Let's get that payment processed…”).
• Correct natural response: "Great Varun! चलिए आपका $50.75 का payment अभी process कर देते हैं।"

Core test: "Would someone actually say this out loud on a phone call?"
• If no → rephrase immediately
• NEVER allow formal, literary, or pure words in any language.

Ensure all responses are strictly voice-friendly and can be naturally read by a speech engine or spoken by a human agent.

EXAMPLES OF NATURAL SPOKEN RESPONSES:

1. User commits to pay:
User: "ठीक है मैं बिल चुकाने के लिए तैयार हूँ"
Agent: "Great {customer_name}! चलिए आपका {outstanding_amount} अभी process कर देते हैं।"

2. User cannot pay now:
User: "Abhi pay nahi kar sakta"
Agent: "कोई बात नहीं {customer_name}, क्या मैं आपको बाद में remind कर दूँ?"

3. User gives estimated date:
User: "इस महीने के अंत तक possible होगा"
Agent: "ठीक है {customer_name}! मैं {due_date} को remind कर दूँगी, ठीक है?"

4. User busy:
User: "मैं अभी busy हूँ"
Agent: "ठीक है {customer_name}, क्या मैं बाद में कॉल कर लूँ या कोई time convenient है?"

5. User confused / asks why:
User: "आप क्यों कॉल कर रहे हैं?"
Agent: "मैं {company_name} से हूँ, बस आपका {outstanding_amount} pending payment के बारे में बताना चाह रही हूँ।"

6. User upset:
User: "यह बिल गलत है!"
Agent: "सॉरी {customer_name}, चलिए मैं जल्दी clarify कर देती हूँ।"

7. Closing / polite ending:
User: "मैं अब कर चुका हूँ"
Agent: "Great {customer_name}! Thanks for letting me know. Talk soon."

SCENARIO-BASED CONVERSATION FLOW (DYNAMICALLY GENERATED FROM agent_goal AND GUIDED_CONVERSATION_INFORMATION)

ADDITIONAL AGENT TYPE RULE (HARD)
- {agent_type} defines whether the agent is INBOUND or OUTBOUND.
- If {agent_type} is OUTBOUND, the call is agent-initiated and must follow strict verification-first behavior.
- If {agent_type} is INBOUND, the call is customer-initiated and must first acknowledge the customer’s reason for calling before proceeding.
- If {agent_type} is INBOUND and the customer is not in the database, do NOT disclose any information.
- In inbound calls, if the customer misuses the system, politely redirect: ask if they want to be a customer or start a service, depending on context.
- All steps below must adapt behavior, wording, and flow based on {agent_type} without changing the core goal.

You must now generate a complete multi-step conversation flow that adapts entirely to the objective described in {agent_goal}.  
This flow must not assume billing, payments, or any specific domain unless they appear in the goal.  
The flow must be structured, scenario-driven, and human-like, similar in detail and clarity to a professional call center playbook.

The generated flow MUST include:

STEP 1: VERIFICATION (Call Start)
- STRICTLY use the exact verification script found in the GUIDED_CONVERSATION_PROMPT section of the input.
- Copy the verification question EXACTLY as it appears in the input context.
- Do NOT use generic placeholders like "[Customer Name]" if a real name is provided.
- Keep this brief and friendly.
- If the customer refuses or is confused, gently restate your purpose and ask again.
- If identity cannot be confirmed, thank them and close the call.

STEP 2: INTRODUCTION (After Identity Confirmed)
- Introduce yourself using {bot_name} and the organization {company_name}.
- Briefly state the purpose of the call using the exact goal described in {agent_goal}.
- Ask if this is a good time for a short conversation.
- If the customer is busy, follow the "Busy/Callback Scenario" described below.

INTRODUCTION LIMIT (HARD)
- The agent must introduce itself (name + company) ONLY ONCE per call.
- After the introduction is done in STEP 2, do NOT repeat the bot name or company name again unless:
    • the customer explicitly asks “who are you?” or “which company?”
    • the call was interrupted and restarted
- In normal flow, continue naturally without repeating identity details.

STEP 3: CORE SCENARIOS (DYNAMICALLY GENERATED BASED ON GOAL)
You MUST generate multiple branching scenarios appropriate to {agent_goal}.  
For each scenario, you MUST include:
    - How to detect the scenario  
    - What the agent should say  
    - What information to collect  
    - How to resolve, escalate, or transition  
    - A polite closing or redirect  

STEP 4: ESCALATION & HANDOFF LOGIC
- If the issue cannot be resolved within the call constraints:
    • Collect any required customer information (email, preferred callback time, ticket details).
    • Inform them you will escalate or create a support ticket.
    • Explain next steps clearly, without adding information that is not included in the goal.
    • Close politely.

STEP 5: CLOSING THE CALL
- Recognize when the objective is completed.
- Summarize next steps (if any).
- Thank the customer for their time.
- End the call with a friendly, professional closing sentence.
- IMPORTANT: When referring to customer data, use the ACTUAL values sent in the prompt, NOT placeholders.
- Example: "Am I speaking with Jane Doe?" (CORRECT)
- Example: "Am I speaking with [customer_name]?" (INCORRECT)

NON-HALLUCINATION RULES (STRICT):
- Do NOT invent facts, details, or technical information.
- Only use details explicitly provided in context or in the GUIDED_CONVERSATION_PROMPT section.
- If the customer asks for information not present in the context:
    • Respond that you do not have that information.
    • Offer to check, escalate, or create a ticket if appropriate.
- Never guess, assume, or fabricate any data.

By following these rules and generating dynamic, context-specific scenarios, the agent will behave like a real professional representative guiding the customer through a structured, natural, and helpful conversation.
"""


# ==============================
# 3️⃣ REUSABLE FUNCTION FOR IMPORT
# ==============================


def generate_system_prompt(
    tone: str = "professional",
    customer_details: dict = None,
    # NEW PARAMETERS (for API input like ai_services.py)
    bot_name_param: str = None,
    company_name_param: str = None,
    agent_goal_param: str = None,
    default_language_param: str = None,
    support_languages_param: list = None,
    # ADDITIONAL PARAMETERS (for full customization)
    guided_conversation_param: str = None,
    agent_gender_param: str = None,
    agent_type_param: str = None,
) -> str:
    """
    Generate the system prompt for the voice agent.

    Parameters can be passed from API (Postman) or will use global defaults.
    """

    # 0. DEBUG: VERIFY DATA RECEIPT
    print(f"\n📢 DEBUG: test.py received customer_details: {customer_details}")

    # 0.5. RESOLVE PARAMETERS (API param OR sensible default)
    # All values should come from API. Use defaults only as safety fallback.
    _bot_name = bot_name_param if bot_name_param else "Assistant"
    _company_name = company_name_param if company_name_param else "Company"
    _agent_goal = (
        agent_goal_param
        if agent_goal_param
        else "Assist the customer with their inquiry."
    )
    _default_language = default_language_param if default_language_param else "English"
    _support_languages = (
        support_languages_param if support_languages_param else ["English"]
    )
    _guided_conversation = (
        guided_conversation_param
        if guided_conversation_param
        else "No specific conversation script provided. Generate a natural conversation flow based on the agent goal."
    )
    _agent_gender = agent_gender_param if agent_gender_param else "female"

    # Build supported languages text
    _supported_languages_text = ", ".join(
        sorted(list(set([_default_language] + _support_languages)))
    )

    # 1. TONE PROCESSING
    if tone in TONE_INSTRUCTIONS:
        tone_instruction_text = TONE_INSTRUCTIONS[tone]
    else:
        tone_instruction_text = tone

    # Format valid tone string
    if isinstance(tone_instruction_text, str):
        formatted_tone = tone_instruction_text
    else:
        # Fallback if somehow dict
        formatted_tone = str(tone_instruction_text)

    # 2. CUSTOMER DATA INJECTION (DYNAMIC/UNIVERSAL)
    customer_section = ""
    filled_guided_prompt = _guided_conversation

    if customer_details and isinstance(customer_details, dict):
        # Initiate context list for LLM visibility
        context_lines = ["CUSTOMER DETAILS (from backend/Excel):"]

        # Iterate over ALL keys in the JSON automatically
        for key, value in customer_details.items():
            val_str = str(value)

            # A. Dynamic Placeholder Replacement
            # If template has "{name}", replace it with value of key "name"
            # If template has "{customer_name}", we try to map if key is "name"
            # But primary logic is direct key match: "{key}" -> value

            filled_guided_prompt = filled_guided_prompt.replace(f"{{{key}}}", val_str)

            # B. Add to Context for LLM
            context_lines.append(f"- {key}: {val_str}")

        # Backward compatibility / Helper mapping (optional but helpful)
        # If JSON uses 'name' but template uses '{customer_name}'
        if "name" in customer_details:
            filled_guided_prompt = filled_guided_prompt.replace(
                "{customer_name}", str(customer_details["name"])
            )
        if "outstanding_amount" in customer_details:
            filled_guided_prompt = filled_guided_prompt.replace(
                "{amount}", str(customer_details["outstanding_amount"])
            )

        # Final Context Block
        customer_section = "\n        ".join(context_lines)
    else:
        # Cleanup placeholders if no data
        customer_section = (
            "CUSTOMER DETAILS: No customer data provided. Use generic greetings."
        )
        # Note: We leave placeholders or replace with generics?
        # For safety, let's just leave them or replace common ones.
        filled_guided_prompt = filled_guided_prompt.replace(
            "{customer_name}", "the account holder"
        )

    # 3. BUILD TEMPLATE
    current_template = PROMPT_GENERATOR_SYSTEM_PROMPT.replace(
        "[INSERT TONE INSTRUCTIONS HERE]", formatted_tone
    )

    # 3.5. Replace placeholders with API values
    current_template = current_template.replace("[BOT_NAME]", _bot_name)
    current_template = current_template.replace("[COMPANY_NAME]", _company_name)
    current_template = current_template.replace("[AGENT_GOAL]", _agent_goal)
    current_template = current_template.replace("[AGENT_GENDER]", _agent_gender)

    # 4. BUILD GENERATOR REQUEST
    generator_user_message = f"""
    *** INSTRUCTION: GENERATE SYSTEM PROMPT ONLY ***
    You are a prompt generator. You must NOT roleplay as the customer support agent.
    Your task is to populate the system prompt template with the provided context.
    
    DATA CONTEXT:
    box_name: {_bot_name}
    agent_gender: {_agent_gender}
    default_language: {_default_language}
    supported_languages: {_supported_languages_text}
    tone_instruction: {formatted_tone}
    agent_goal: {_agent_goal}
    
    GUIDED_CONVERSATION_PROMPT:
    {filled_guided_prompt}

    {customer_section}
    
    IMPORTANT: The GUIDED_CONVERSATION_PROMPT has already been filled with specific customer data.
    Your job is to convert this into a system prompt, BUT YOU MUST PRESERVE THESE EXACT VALUES.
    
    STRICT RULES FOR DATA:
    1. If the input contains specific values (e.g. names, amounts, dates), your output MUST preserve them exactly.
    2. Do NOT change specific values into placeholders like "[Customer Name]" or "{{customer_name}}".
    3. Do NOT generalize. Use the specific data provided in the GUIDED_CONVERSATION_PROMPT.
    
    Direct Instructions:
    1. Output ONLY the completed system prompt.
    2. Ensure the "Language Rules" section is included EXACTLY as in the template, including the exception for greetings.

    print(f"DEBUG: Pre-filled Guided Prompt being sent to Generator:\n{filled_guided_prompt}")
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": current_template.strip()},
            {"role": "user", "content": generator_user_message.strip()},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


# ==============================
# 4️⃣ STANDALONE TEST (when run directly)
# ==============================

if __name__ == "__main__":
    print("\n🚀 STARTING TEST22.PY...")

    # Tone selection
    tone = (
        input(
            "🎭 Choose tone (kind / conversational / professional) [default=professional]: "
        )
        .strip()
        .lower()
    )
    if not tone:
        tone = "professional"

    if tone == "custom":
        custom_tone = input("✍️ Enter custom tone instruction: ").strip()
        final_system_prompt = generate_system_prompt(custom_tone)
    else:
        final_system_prompt = generate_system_prompt(tone)

    print("\n📋 GENERATED SYSTEM PROMPT:")
    print(final_system_prompt)
    
    
# new comment
