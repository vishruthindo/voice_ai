from collections import defaultdict
import json
import os
import sys
from pathlib import Path
import pickle
import re
from bs4 import BeautifulSoup

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import markdown
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
from django.conf import settings
from langchain_openai import ChatOpenAI
from openai import OpenAI
from paddleocr import PaddleOCR
from typing import List, Dict, Union, Optional, Any, Tuple, Generator

import pandas as pd
from langchain_community.document_loaders import (
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
    UnstructuredPowerPointLoader,
    PyPDFLoader,
    UnstructuredImageLoader,
)
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from mm1 import DocumentWithEmbedding, DocumentMetadata
from langdetect import detect
from test22_copy import generate_system_prompt

client = OpenAI()
from dotenv import load_dotenv

load_dotenv()
tesseract_path = os.getenv(
    "TESSERACT_FILE", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
pytesseract.pytesseract.tesseract_cmd = tesseract_path


def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns 'ja' for Japanese, 'en' for English, or 'auto' if undetermined.
    """
    try:
        lang = detect(text)
        if lang.startswith("ja"):
            return "ja"
        elif lang.startswith("en"):
            return "en"
        else:
            return "auto"
    except:
        return "auto"


# --- Retrieval System ---
class DocumentRetriever:
    def __init__(
        self,
        embedding_files: List[str],
        tenantids: Optional[List[str]] = None,
        departmentids: Optional[List[str]] = None,
        fileids: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        default: Optional[bool] = None,
    ):
        """Initialize with multiple embedding files and filtering criteria."""
        self.embedding_files = embedding_files
        self.tenantids = tenantids
        self.departmentids = departmentids
        self.fileids = fileids
        self.filenames = filenames
        self.default = default

        # Load all documents from all embedding files
        self.all_documents = self._load_all_documents()

        # Filter documents based on initialization criteria
        self.filtered_documents = self._filter_documents(
            self.all_documents, tenantids, departmentids, fileids, filenames, default
        )

        # Create FAISS index for filtered documents
        self.faiss_index = self._create_faiss_index(self.filtered_documents)

        print(
            f"🔍 Retriever initialized with {len(self.filtered_documents)} documents after filtering",
            flush=True,
        )

    def _load_all_documents(self) -> List[DocumentWithEmbedding]:
        """Load documents from all embedding files."""
        all_docs = []
        for file_path in self.embedding_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(data, dict) and "documents" in data:
                            all_docs.extend(data["documents"])
                        else:
                            print(f"⚠️ Unexpected format in {file_path}", flush=True)
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {e}", flush=True)
            else:
                print(f"⚠️ File not found: {file_path}", flush=True)
        return all_docs

    def _filter_documents(
        self,
        documents: List[DocumentWithEmbedding],
        tenantids: Optional[List[str]] = None,
        departmentids: Optional[List[str]] = None,
        fileids: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        default: Optional[bool] = None,
    ) -> List[DocumentWithEmbedding]:
        """Filter documents based on multiple criteria with fallback logic."""
        if not any([tenantids, departmentids, fileids, filenames, default is not None]):
            return documents  # Return all if no filters

        # Try different filter combinations with fallback
        filter_combinations = [
            {
                "tenantids": tenantids,
                "departmentids": departmentids,
                "fileids": fileids,
            },  # All filters
            {"tenantids": tenantids, "departmentids": departmentids},  # Tenant + Dept
            {"tenantids": tenantids},  # Tenant only
            {"filenames": filenames},  # Filename only
            {"default": True} if default is not None else None,  # Default docs
        ]

        # Remove None combinations
        filter_combinations = [fc for fc in filter_combinations if fc is not None]

        for combo in filter_combinations:
            filtered = []
            for doc in documents:
                match = True
                metadata = doc.get("metadata")

                if "tenantids" in combo and combo["tenantids"] is not None:
                    if metadata["tenantid"] not in combo["tenantids"]:
                        match = False

                if (
                    match
                    and "departmentids" in combo
                    and combo["departmentids"] is not None
                ):
                    if metadata["departmentid"] not in combo["departmentids"]:
                        match = False

                if match and "fileids" in combo and combo["fileids"] is not None:
                    if metadata["fileid"] not in combo["fileids"]:
                        match = False

                if match and "filenames" in combo and combo["filenames"] is not None:
                    if metadata["filename"] not in combo["filenames"]:
                        match = False

                if match and "default" in combo and combo["default"] is not None:
                    if metadata["default"] != combo["default"]:
                        match = False

                if match:
                    filtered.append(doc)

            if filtered:
                print(
                    f"🔍 Found {len(filtered)} documents with filter combo: {combo}",
                    flush=True,
                )
                return filtered

        return []  # No documents matched any filter combination

    def _create_faiss_index(
        self, documents: List[DocumentWithEmbedding]
    ) -> faiss.IndexFlatL2:
        """Create FAISS index from document embeddings."""
        if not documents:
            raise ValueError("No documents available to create index")

        embeddings = [doc["embedding"] for doc in documents]
        dimension = len(embeddings[0])

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))
        return index

    def get_relevant_documents(
        self, query: str, top_k: int = 5, threshold: float = 1.4
    ) -> List[DocumentWithEmbedding]:
        """Retrieve relevant documents based on pre-filtered index."""
        if not self.filtered_documents:
            return []

        # Get query embedding
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)

        # Search the index
        distances, indices = self.faiss_index.search(
            np.array([query_embedding]).astype("float32"),
            k=min(top_k, len(self.filtered_documents)),
        )

        # Get relevant documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.filtered_documents):
                if distances[0][list(indices[0]).index(idx)] < threshold:
                    results.append(self.filtered_documents[idx])

        return results


# --------------------------------------------------------------------------------
# Helper Functions for Prompt Safety
# --------------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def list_placeholders(text: str):
    """List all unique curly-brace placeholders in the text."""
    if not text:
        return []
    return list({m.group(1).strip() for m in re.finditer(_PLACEHOLDER_RE, text)})


def safe_render_template(template: str, vars_map: dict):
    """
    Render a template string safely, filling in known variables from vars_map.
    Missing variables are left as-is (with braces intact).
    Double braces {{ and }} are converted to single { and }.
    """
    # tolerant: convert {{ and }} back to single braces if double-escaped
    template = template.replace("{{", "{").replace("}}", "}")

    # simple safe .format_map with fallback to keep missing keys visible
    class _Missing(dict):
        def __missing__(self, key):
            return "{" + key + "}"  # leave placeholder in place

    # cast everything to string
    vars_str = {k: ("" if v is None else str(v)) for k, v in vars_map.items()}

    try:
        return template.format_map(_Missing(vars_str))
    except Exception:
        # fallback: return template unchanged if something goes wrong
        return template


# --------------------------------------------------------------------------------


# --- LLM Integration ---
class LLMAnswerGenerator:
    # In LLMAnswerGenerator.__init__ method in check_prompt.py:

    def __init__(
        self,
        model_name: str = "gpt-4o",
        customer_details: Optional[dict] = None,
        # NEW PARAMETERS (for API input like ai_services.py)
        bot_name: Optional[str] = None,
        company_name: Optional[str] = None,
        agent_goal: Optional[str] = None,
        default_language: Optional[str] = None,
        support_languages: Optional[List[str]] = None,
        # ADDITIONAL PARAMETERS (for full customization)
        guided_conversation: Optional[str] = None,
        agent_gender: Optional[str] = None,
        agent_type: Optional[str] = None,
        tone: str = "professional",  # kind, conversational, professional, or custom
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3, streaming=True)
        self.call_end_llm = ChatOpenAI(model=model_name, temperature=0.0)

        print("🔄 Generating system prompt from test.py (one-time)...", flush=True)

        # 1. Generate raw system prompt (string)
        # Note: 'test.py' might return text with placeholders roughly like {customer_name}
        raw_prompt = generate_system_prompt(
            tone=tone,  # Use API-provided tone
            customer_details=customer_details,
            # Pass API parameters to test22.py
            bot_name_param=bot_name,
            company_name_param=company_name,
            agent_goal_param=agent_goal,
            default_language_param=default_language,
            support_languages_param=support_languages,
            # Additional parameters
            guided_conversation_param=guided_conversation,
            agent_gender_param=agent_gender,
            agent_type_param=agent_type,
        )

        # 2. Build the exact substitution map for the final prompt
        # Ensure we have strings for everything

        # Access imported globals from test.py if possible, or redefine if needed.
        # Since 'test.py' was imported, we can access its variables if we imported them.
        # But 'generate_system_prompt' in test.py already injected TONE_INSTRUCTIONS via f-string?
        # If test.py injects a dict, we might see "{'kind': ...}" in raw_prompt.
        # We assume test.py will be fixed to not output those, OR we rely on safe_render not to choke.

        from test22 import TONE_INSTRUCTIONS

        if isinstance(TONE_INSTRUCTIONS, dict):
            tone_text = "\n".join([f"{k}: {v}" for k, v in TONE_INSTRUCTIONS.items()])
        else:
            tone_text = str(TONE_INSTRUCTIONS)

        vars_map = {
            "bot_name": bot_name,
            "company_name": company_name,
            "agent_gender": agent_gender,
            "tone_instruction": tone_text,
            "agent_goal": agent_goal,
            "agent_type": agent_type,
            # customer values:
            "customer_name": customer_details.get("name")
            if customer_details
            else "the account holder",
            "account_number": customer_details.get("account_number")
            if customer_details
            else "",
            "outstanding_amount": customer_details.get("outstanding_amount")
            if customer_details
            else "",
            "due_date": customer_details.get("due_date") if customer_details else "",
        }

        # 3. Render safely
        rendered_prompt = safe_render_template(raw_prompt, vars_map)

        # 4. CRITICAL: APPEND CUSTOMER DATA DIRECTLY TO THE SYSTEM PROMPT
        # This ensures the voice agent LLM always "sees" the actual values
        # regardless of what the Prompt Generator outputs.
        if customer_details and isinstance(customer_details, dict):
            customer_context = (
                "\n\n=== CUSTOMER DATA FOR THIS CALL (USE THESE EXACT VALUES) ===\n"
            )
            for key, value in customer_details.items():
                customer_context += f"- {key}: {value}\n"
            customer_context += "\nIMPORTANT: When the customer asks about their details, USE THE VALUES ABOVE.\n"
            customer_context += "Example: If they ask 'what is my name?', answer with the 'name' value above.\n"
            customer_context += (
                "Example: If they ask about account/invoice, use the values above.\n"
            )
            customer_context += "=== END CUSTOMER DATA ===\n"
            self._cached_system_prompt = rendered_prompt + customer_context
        else:
            self._cached_system_prompt = rendered_prompt

        # Check for remaining suspect placeholders
        remaining = list_placeholders(self._cached_system_prompt)
        if remaining:
            print(
                f"⚠️ System prompt contains curly placeholders (will be rendered as-is unless you supply vars): {remaining}"
            )

        # DEBUG output
        print(
            "DEBUG final system prompt first 400 chars:",
            self._cached_system_prompt[:400],
        )
        print(
            "DEBUG last 500 chars of system prompt:", self._cached_system_prompt[-500:]
        )
        print("✅ System prompt generated and cached.\n", flush=True)

    def _extract_used_sources(
        self, response: str, documents: List[DocumentWithEmbedding]
    ) -> List[str]:
        """Analyze the response to determine which sources were actually used"""
        used_sources = set()

        # Check for direct mentions of filenames in the response
        for doc in documents:
            if "metadata" in doc:
                if doc["metadata"]["filename"].lower() in response.lower():
                    fileid = doc["metadata"].get("fileid")
                    if fileid is not None:
                        used_sources.add(fileid)

        # If no direct mentions, use similarity to find most relevant
        if not used_sources:
            embeddings = OpenAIEmbeddings()
            response_embedding = embeddings.embed_query(response)

            # Compare with each document's content
            similarities = []
            for doc in documents:
                fileid = doc["metadata"].get("fileid")
                text = doc.get("text", "")

                if fileid not in ["None", None]:
                    doc_embedding = embeddings.embed_query(
                        text[:1000]
                    )  # First part for efficiency
                    similarity = np.dot(response_embedding, doc_embedding)
                    similarities.append((fileid, similarity))

            # Take top 2 most similar documents
            similarities.sort(key=lambda x: x[1], reverse=True)
            used_sources = {x[0] for x in similarities[:2]}

        return list(used_sources)

    def _decide_end_call(
        self,
        chat_history: List[Tuple[str, str]],
        last_user: str,
        last_assistant: str,
    ) -> bool:
        history_text = ""
        for human, ai in chat_history:
            history_text += f"User: {human}\nAssistant: {ai}\n"

        judge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                You are a call supervisor.
                Your only job is to look at a phone conversation between a customer and an AI collections agent
                and decide if the call has naturally ended.

                Respond with exactly one word:
                - END       → if the call should end now.
                - CONTINUE  → if the conversation should continue.

                Do not add anything else.

                ────────────────────────────────────────
                GLOBAL PRECONDITION — Conversation Activation Gate

                - conversation_active is FALSE by default.

                - conversation_active becomes TRUE only after the assistant has produced
                at least one meaningful response (excluding greetings, system prompts, or fillers).

                - While conversation_active is FALSE:
                    - User disengagement signals MUST be ignored for call-ending purposes
                    - The assistant MUST NOT produce any closing, farewell, or polite sign-off
                    - The assistant MUST respond with a neutral continuation or clarification
                    - call_ended MUST remain false
                
                IMPORTANT EXCEPTION:
                    - If the user gives an explicit command to stop contact (e.g., "cut the call", "hang up", "stop calling me")
                    OR the user becomes threatening/abusive, then END immediately even if conversation_active is FALSE.

                ────────────────────────────────────────
                Opening-Turn Disengagement Override (EXPLICIT)

                - If a disengagement phrase such as “bye”, “goodbye”, “thanks”, “thank you”, or similar
                appears in the first user message or before conversation_active becomes TRUE,
                it MUST NOT be interpreted as an intent to end the call.

                - Such messages are treated as non-terminal conversational input,
                and the call MUST be marked as CONTINUE.

                ────────────────────────────────────────
                Call Ending Conditions

                The call is considered ENDED only when ALL of the following are true:
                - conversation_active is TRUE
                - The assistant has given a clear and explicit closing
                (e.g., “thank you for your time, have a great day”)
                - There are no open questions, pending requests, or unfinished actions from either side
                - The user has acknowledged everything relevant
                - No follow-up, clarification, payment, scheduling, or future action is pending or implied
                - There is no ambiguity about whether the conversation has concluded

                ────────────────────────────────────────
                Call Continuation Conditions

                The call is NOT ended if ANY of the following are true:
                - The assistant has asked any question that has not been answered
                - The user has asked a question or made a request that has not been addressed
                - Any action, payment, scheduling, or follow-up has been proposed
                but not explicitly accepted, declined, or confirmed
                - The user expresses any future intent, deferred action, or plan to follow up later
                → In this case, any assistant closing MUST be ignored
                - The user provides only acknowledgements, fillers, pauses, silence,
                partial sentences, repeated words, or awkward phrasing
                - The assistant closes prematurely without explicit confirmation
                that all user needs have been addressed
                - There is any ambiguity about whether the call has concluded

                ────────────────────────────────────────
                Additional Rules (Evaluated ONLY when conversation_active == TRUE)

                - Short interjections such as “okay”, “thanks”, “thank you”, “got it”, or similar
                are acknowledgements and MUST NOT end the call by themselves.

                - Farewell phrases such as “bye”, “goodbye”, or similar
                MUST NOT be treated as call-ending signals on their own.

                - Call termination must never be inferred from user language alone.
                An explicit, fully resolved conversational state is required.
                
                Post-Activation Disengagement Handling

                - If conversation_active == TRUE and the user says a disengagement phrase
                such as “bye”, “goodbye”, “thanks”, “thank you”, or similar:
                
                1. The assistant must first check if the user is satisfied or if any open issues remain.
                2. If the user confirms all issues are resolved:
                    - The call should be considered END.
                3. If the user responds with something else:
                    - Analyze intent:
                    • If it indicates unresolved issues, questions, or requests → CONTINUE.
                    • If it indicates unavailability, busyness, or inability to continue → END.
                    • If ambiguous → CONTINUE.

                ────────────────────────────────────────
                Decision Invariant

                - If there is any doubt, ambiguity, or unresolved state,
                the correct decision is CONTINUE.

                """,
                ),
                (
                    "human",
                    """
                Full Conversation History:
                {history}
 
                Last user message:
                {last_user}
 
                Last assistant reply:
                {last_assistant}
 
                Should this call be considered finished now?
                Reply with ONLY END or CONTINUE.
                """,
                ),
            ]
        )

        chain = judge_prompt | self.call_end_llm | StrOutputParser()
        result = (
            chain.invoke(
                {
                    "history": history_text,
                    "last_user": last_user,
                    "last_assistant": last_assistant,
                }
            )
            .strip()
            .upper()
        )
        return result == "END"

    #     def _decide_end_call(
    #     self,
    #     chat_history: List[Tuple[str, str]],
    #     last_user: str,
    #     last_assistant: str,
    # ) -> bool:
    #         # Build a simple conversation text
    #         history_text = ""
    #         for human, ai in chat_history:
    #             history_text += f"User: {human}\nAssistant: {ai}\n"

    #         judge_prompt = ChatPromptTemplate.from_messages([
    #             (
    #                 "system",
    #                 """
    #                 You are a call supervisor.
    #                 Your only job is to look at a phone conversation between a customer and an AI collections agent
    #                 and decide if the call has naturally ended.

    #                 The call is considered ENDED when:
    #                 - The assistant has given a clear closing (for example: "thank you for your time, have a great day", "thanks for confirming, goodbye"), AND
    #                 - There are no open questions or pending confirmations from either side.

    #                 The call is NOT ended if:
    #                 - The assistant has just asked a question and is waiting for an answer.
    #                 - The customer has asked something that still needs a reply.
    #                 - Any next step still needs confirmation (like "shall I send the link?", "when can you pay?", etc.).

    #                 Respond with exactly one word:
    #                 - END       → if the call should end now.
    #                 - CONTINUE  → if the conversation should continue.

    #                 Do not add anything else.
    #                 """
    #             ),
    #             (
    #                 "human",
    #                 """
    #                 Full Conversation History:
    #                 {history}

    #                 Last user message:
    #                 {last_user}

    #                 Last assistant reply:
    #                 {last_assistant}

    #                 Should this call be considered finished now?
    #                 Reply with ONLY END or CONTINUE.
    #                 """
    #             )
    #         ])

    #         chain = judge_prompt | self.call_end_llm | StrOutputParser()
    #         result = chain.invoke({
    #             "history": history_text,
    #             "last_user": last_user,
    #             "last_assistant": last_assistant,
    #         }).strip().upper()

    #         # ✅ ADD DEBUG PRINT
    #         print(f"🔍 End call check: result = {result}")

    #         return result == "END"

    def voice_answer(
        self,
        query: str,
        documents: List[DocumentWithEmbedding],
        conversation_id: str,
        chat_history: List[Tuple[str, str]],
        lang: str = "auto",
    ) -> dict:
        """
        Generate a voice-optimized answer from documents (non-streaming, concise response).
        Designed for voice assistant interactions rather than chat.

        Args:
            query: The user's query string
            documents: List of relevant documents
            conversation_id: ID for conversation tracking
            lang: Language preference ("en", "ja", or "auto")

        Returns:
            dict: {
                "answer": str,  # Concise voice-optimized answer
                "source_files": List[str],  # Files actually used
                "conversation_id": str
            }
        """

        try:
            context = "\n\n".join(
                [
                    f"Document {i + 1} (Source: {doc['metadata']['filename']}):\n{doc['text']}"
                    for i, doc in enumerate(documents)
                ]
            )
            # 🟢 INSERT LANGUAGE LOGIC HERE
            if lang == "ja":
                language_instruction = "\nRespond in Japanese.\n"
            elif lang == "en":
                language_instruction = "\nRespond in English.\n"
            else:
                language_instruction = (
                    "\nRespond in the same language as the user's question.\n"
                )

            history_text = ""
            for human, ai in chat_history:
                history_text += f"User: {human}\nAssistant: {ai}\n"
            context = history_text + "\n\n" + context  # Prepend history
            # Use the cached system prompt (generated once at init)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self._cached_system_prompt),
                    (
                        "human",
                        """
                    Conversation History:
                    {history}
                    Customer just said: "{question}"
                    
                    

STRICT OUTPUT RULES:
- Output ONLY plain text
- Do NOT use SSML, XML, HTML, tags, markdown, or code fences
- Do NOT put words on separate lines
- Do NOT include line breaks, tabs, or escape characters like \n, \t, \r
- Remove all special characters except letters, numbers, and basic punctuation: . , ! ? ' -
- Use only standard ASCII characters; no Unicode or non-ASCII symbols
- Trim all leading and trailing whitespace
- Use exactly one space between words; no extra spaces
- Ensure the response ends with appropriate punctuation if it's a complete sentence
- Keep the response concise: 1-2 short sentences maximum
- No emojis or other symbols
- Normalize spaces (no double spaces or line breaks)
- Use ONE natural spoken sentence (maximum two)
- Write exactly as a human would speak aloud

Return ONLY the final cleaned sentence text.
                    """,
                    ),
                ]
            )

            chain = prompt | self.llm | StrOutputParser()
            full_response = ""

            # Stream the response
            for token in chain.stream(
                {"question": query, "context": context, "history": history_text}
            ):
                full_response += token
                yield token

            # Analyze which sources were actually used
            used_sources = self._extract_used_sources(full_response, documents)

            # Yield metadata after completion
            yield {
                "full_response": full_response,
                "source_files": used_sources,  # Only files actually used
                "conversation_id": str(conversation_id),
            }

        except Exception as e:
            yield json.dumps(
                {"status": "error", "message": f"Answer generation failed: {str(e)}"}
            )
            return


# ---------------------------------------------------------------------

if __name__ == "__main__":
    # 👇 Path to your embeddings pkl file
    embedding_file_path = "embeddings_data.pkl"

    # 👇 Example filtering - adjust these as per your metadata
    tenant_ids = ["sdfgdgdsfgsdfgdsgds"]
    department_ids = None
    file_ids = None
    filenames = None
    default = True

    # ✅ 1. Create the retriever once
    retriever = DocumentRetriever(
        embedding_files=[embedding_file_path],
        tenantids=tenant_ids,
        departmentids=department_ids,
        fileids=file_ids,
        filenames=filenames,
        default=default,
    )

    def get_customer_data():
        """
        Get customer data dynamically from backend/Excel.
        Backend will send JSON/dict of customer details.
        """
        # Method 1: Command line argument (backend calls: python check_prompt.py '{"name":"John","amount":"$100"}')
        if len(sys.argv) > 1:
            try:
                return json.loads(sys.argv[1])
            except:
                pass

        # Method 2: JSON file
        if os.path.exists("customer_data.json"):
            with open("customer_data.json", "r") as f:
                return json.load(f)

        # Method 3: Environment variable
        customer_json = os.getenv("CUSTOMER_DATA")
        if customer_json:
            return json.loads(customer_json)

        return None

    # Get dynamic customer data
    customer_details = get_customer_data()

    if customer_details:
        print(f"📋 Customer Details (from backend): {customer_details}")
    else:
        print("ℹ️ No customer data provided. Using generic prompt.")

    # ✅ Create LLM generator with dynamic data (or None)
    llm_generator = LLMAnswerGenerator(customer_details=customer_details)

    chat_history = []

    # Default first query for new conversation
    default_greeting_query = "Hello"
    while True:
        if not chat_history:
            # Use the default greeting as the first query
            query = default_greeting_query
            print(f"\n🤖 Sending default greeting:\n{query}")
        else:
            query = input(
                "\n❓ Enter your question (or type 'exit' to quit'): "
            ).strip()
            if query.lower() in ["exit", "quit"]:
                print("👋 Exiting.")
                break

        detected_lang = detect_language(query)
        print(f"🌐 Detected query language: {detected_lang}")

        relevant_docs = retriever.get_relevant_documents(query)
        print(f"\n🔎 Retrieved {len(relevant_docs)} relevant documents.")

        if not relevant_docs:
            print("⚠️ No documents matched the filters or query!")
            continue

        # ✅ TEXT-ONLY OUTPUT
        print("\n🧠 Answer from LLM:\n")
        conversation_id = "demo-convo-001"

        result_text = ""
        assistant_reply = ""
        metadata = None

        # Loop through streaming chunks (text only)
        for chunk in llm_generator.voice_answer(
            query,
            relevant_docs,
            conversation_id,
            chat_history=chat_history,
            lang=detected_lang,
        ):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                result_text += chunk
                assistant_reply += chunk
            else:
                # Final metadata chunk
                metadata = chunk
        print("\n\n✅ Full Answer Generation Complete.")
        # Print each word on its own line
        for word in result_text.split():
            print(word)
        chat_history.append((query, assistant_reply))

        # 🔍 Let the AI decide if the call has naturally ended
        should_end = llm_generator._decide_end_call(
            chat_history=chat_history,
            last_user=query,
            last_assistant=assistant_reply,
        )

        if should_end:
            print("\n\n📞 Call ended automatically based on conversation.\n")
            break
