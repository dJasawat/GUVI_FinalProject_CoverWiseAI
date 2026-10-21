
import streamlit as st
import tempfile

from sentence_transformers import SentenceTransformer
import numpy as np
import src.db as db
from pathlib import Path
import src.policy_analyzer as policyAnalyzer

from src.policy_session_manager import PolicySessionManager


st.set_page_config(
    page_title="CoverWiseAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


#@st.cache_resource
#def load_ocr():
#    return PaddleOCRVL(
#        pipeline_version="v1.6"
#    )


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


#ocr = load_ocr()
embedding_model = load_embedding_model()

if "policy_manager" not in st.session_state:

    st.session_state.policy_manager = (
        PolicySessionManager(embedding_model)
    )

manager = st.session_state.policy_manager
# ============================================================
# PAGE CONFIG
# ============================================================

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Overall page ---------- */

    .stApp {
        background-color: #ffffff;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ---------- App title ---------- */

    .app-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 25px;
    }

    /* ---------- Welcome screen ---------- */

    .welcome-container {
        text-align: center;
        padding-top: 12vh;
    }

    .welcome-icon {
        font-size: 52px;
        margin-bottom: 15px;
    }

    .welcome-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .welcome-text {
        color: #6b7280;
        font-size: 16px;
    }

    /* ---------- Uploaded PDF card ---------- */

    .pdf-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        font-size: 13px;
        word-break: break-word;
    }

    .pdf-icon {
        margin-right: 6px;
    }

    /* ---------- Chat area ---------- */

    .user-label {
        font-weight: 600;
        margin-bottom: 5px;
    }

    .assistant-label {
        font-weight: 600;
        margin-bottom: 5px;
    }

    /* ---------- Quick actions ---------- */

    .quick-action-title {
        text-align: center;
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 10px;
    }

    /* ---------- Hide Streamlit decoration ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = {}
if "ignored_upload_names" not in st.session_state:
    st.session_state.ignored_upload_names = set()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "policy_summary" not in st.session_state:
    st.session_state.policy_summary = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_message(role: str, content: str):
    """Add a message to chat history."""
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
        }
    )

def remove_pdf(file_name):
    """Remove a PDF and its extracted text/embeddings from this session."""
    st.session_state.uploaded_pdfs.pop(file_name, None)
    manager.remove_policy(file_name)
    st.session_state.ignored_upload_names.add(file_name)
    if st.session_state.get("policy_summary"):
        summary_meta = st.session_state.policy_summary.get("metadata", {})
        if summary_meta.get("file_name") == file_name:
            st.session_state.policy_summary = None


def clear_all_pdfs():
    """Remove all PDFs and their in-memory extracted data."""
    st.session_state.ignored_upload_names.update(st.session_state.uploaded_pdfs.keys())
    st.session_state.uploaded_pdfs = {}
    manager.clear()
    st.session_state.policy_summary = None

#============================================================
#  Collect user inputs for insurance requirements
#============================================================
def collect_user_inputs():
    """
    Collect multiple user inputs during a chat session.
    Return all answers as a dictionary on Submit.
    """

    # Initialize session state
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = None

    with st.form("user_requirements_form"):

        st.subheader("Tell me your insurance requirements")

        age = st.number_input(
            "What is your age?",
            min_value=18,
            max_value=100,
            value=30
        )

        family_members = st.multiselect(
            "Who needs insurance coverage?",
            ["Self", "Spouse", "Children", "Parents"]
        )

        sum_insured = st.selectbox(
            "What coverage amount do you prefer?",
            ["5 Lakhs", "10 Lakhs", "15 Lakhs", "25 Lakhs", "50 Lakhs"]
        )

        budget = st.number_input(
            "What is your annual premium budget (₹)?",
            min_value=0,
            value=20000,
            step=1000
        )

        pre_existing_disease = st.radio(
            "Does anyone have a pre-existing disease?",
            ["Yes", "No", "Not sure"]
        )

        additional_requirements = st.text_area(
            "Any additional requirements?",
            placeholder="For example, low waiting period, maternity cover..."
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        answers = {
            "age": age,
            "family_members": family_members,
            "sum_insured": sum_insured,
            "annual_budget": budget,
            "pre_existing_disease": pre_existing_disease,
            "additional_requirements": additional_requirements
        }

        st.session_state.user_answers = answers

    return st.session_state.user_answers


def generate_response(user_question: str) -> str:
    try:
        # Embed the user's question once.
        query_embedding = embedding_model.encode(
            user_question,
            normalize_embeddings=True,
            )
        
        query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32,
            )
        
        relevant_chunks = manager.relevant_chunks_semantic_search(user_question, top_k=5)
            
        #retrive chunks from qdrant and build the context for the chatbot
        policy_context = "\n\n".join(
                (
                    f"Policy: {chunk.get('policy_name', 'Unknown policy')}; "
                    f"Page: {chunk.get('page_number', 'unknown')}; "
                    f"Section: {chunk.get('section_title', 'Unknown section')};"
                    f"Clause: {chunk.get('clause_heading', 'Unknown clause')}\n"
                    f"{chunk.get('text', '')}"
                )
                for chunk in relevant_chunks
                if chunk.get("text")
                )
        
            #retrieve question and answer pairs from qdrant and build the context for the chatbot
        try:
            qa_Pair_chunks = db.search_policy_chunks(query_embedding, limit=5)
        except Exception as exc:
            # Policy Q&A should still work from uploaded PDFs if Qdrant is not configured.
            qa_Pair_chunks = []
            st.info(f"Optional Q&A knowledge base unavailable; using uploaded policy text only. ({exc})")
        qa_context = "\n\n".join(
            f"question: {chunk.get('question', 'Unknown question')}\n"
            f"answer: {chunk.get('answer', 'Unknown answer')}\n"
            f"category: {chunk.get('category', 'Unknown category')}\n"
            f"target_clause_or_feature: {chunk.get('target_clause_or_feature', 'Unknown feature')}"
            for chunk in qa_Pair_chunks
        )
        if qa_context:
            policy_context += "\n\n" + qa_context
        return policyAnalyzer.call_policy_chat(user_question, policy_context, st.session_state.messages[:-1])

    except Exception as e:
        st.error(f"Error retrieving relevant policy chunks: {e}")
        return "I'm sorry, I couldn't retrieve relevant information from the policies at this time."


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Application branding
    st.markdown(
        """
        <div class="app-title">🛡️ CoverWiseAI</div>
        <div class="app-subtitle">
            Health Insurance Policy Analyzer
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New chat
    if st.button(
        "＋ New Chat",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # UPLOAD POLICY
    # --------------------------------------------------------

    st.markdown("### 📄 Policy Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF policies",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            if (
                uploaded_file.name not in st.session_state.uploaded_pdfs
                and uploaded_file.name not in st.session_state.ignored_upload_names
            ):
                try:
                    policy_info = manager.add_policy(uploaded_file)
                    st.session_state.uploaded_pdfs[uploaded_file.name] = policy_info
                except Exception as exc:
                    st.error(f"Could not process {uploaded_file.name}: {exc}")

    # --------------------------------------------------------
    # DISPLAY UPLOADED PDFs
    # --------------------------------------------------------

    if st.session_state.uploaded_pdfs:

        st.markdown(
            f"**Uploaded Policies ({len(st.session_state.uploaded_pdfs)})**"
        )

        for file_name in list(
            st.session_state.uploaded_pdfs.keys()
        ):

            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class="pdf-card">
                        <span class="pdf-icon">📄</span>
                        {file_name}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                if st.button(
                    "×",
                    key=f"remove_{file_name}",
                    help=f"Remove {file_name}",
                ):
                    remove_pdf(file_name)
                    st.rerun()

        st.divider()

        if st.button(
            "🗑️ Clear all policies",
            use_container_width=True,
        ):
            clear_all_pdfs()
            st.rerun()

    else:

        st.info(
            "Upload one or more PDF policies to get started."
        )

    if st.session_state.ignored_upload_names and st.button(
        "Reset dismissed upload selection", use_container_width=True
    ):
        st.session_state.ignored_upload_names.clear()
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # SIDEBAR OPTIONS
    # --------------------------------------------------------

    st.markdown("### ⚙️ Options")

    show_sources = st.checkbox(
        "Show source clauses",
        value=True,
    )

    show_pages = st.checkbox(
        "Show page references",
        value=True,
    )

    st.divider()

    st.caption(
        "CoverWiseAI • AI-powered insurance policy analysis"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; margin-bottom:10px;">
        <h1 style="margin-bottom:0;">🛡️ CoverWiseAI</h1>
        <p style="color:#6b7280;">
            Understand your health insurance policy with AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EMPTY STATE / WELCOME
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">🛡️</div>
            <div class="welcome-title">
                How can I help with your policy?
            </div>
            <div class="welcome-text">
                Upload your health insurance policy and ask questions,
                summarize coverage, or compare policies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="quick-action-title">Try one of these</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "📋 Summarize Policy",
            use_container_width=True,
        ):
            if not st.session_state.uploaded_pdfs:
                st.warning("Please upload a policy PDF first.")
            else:
                add_message(
                    "user",
                    "Summarize my insurance policy",
                )
                if manager.policies:
                    policy = next(iter(manager.policies.values()))
                    
                    try:
                        with st.spinner("Analyzing policy clauses and checking source evidence..."):
                            st.session_state.policy_summary = policyAnalyzer.analyze_policy(
                                policy["chunks"],
                                policy["full_text"],
                                ocr_pages=policy.get("ocr_pages", []),
                            )
                        summary = st.session_state.policy_summary
                        extraction = summary.get("extraction", {})
                        summary_text = (
                            "### Policy summary\n"
                            f"**Policy:** {summary.get('metadata', {}).get('policy_name', 'Unnamed policy')}\n\n"
                            f"**Coverage items:** {len(extraction.get('coverage', []))}\n\n"
                            f"**Exclusions:** {len(extraction.get('exclusions', []))}\n\n"
                            f"**Waiting-period clauses:** {len(extraction.get('waiting_periods', []))}\n\n"
                            "See the Policy Summary section below for extracted clauses and evidence."
                        )
                        add_message("assistant", summary_text)
                    except Exception as exc:
                        st.error(f"Policy summary failed: {exc}")

                else:
                    st.warning("Please upload a policy PDF first.")
                st.rerun()

    with col2:
        if st.button(
            "🔍 Ask About Coverage",
            use_container_width=True,
        ):

            # Call the method
            user_inputs = collect_user_inputs()

            if user_inputs is not None:
                st.success("Your answers have been submitted!")
                st.write("Collected inputs:")
                st.json(user_inputs)
                add_message(
                "assistant",
                "Sure. Ask me anything about your insurance policy. "
                "For example: **Does this policy cover cataract surgery?**",
                )
                st.rerun()

    with col3:
        if st.button(
            "⚖️ Compare Policies",
            use_container_width=True,
        ):
            if len(st.session_state.uploaded_pdfs) < 2:
                st.warning(
                    "Upload at least two policies to compare them."
                )
            else:
                add_message(
                    "user",
                    "Compare my uploaded policies",
                )

                add_message(
                    "assistant",
                    "I can compare the uploaded policies across "
                    "coverage, waiting periods, exclusions, room-rent "
                    "limits, copay, deductibles and other important clauses.",
                )

                st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.markdown(message["content"])

    else:

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_prompt = st.chat_input(
    "Ask anything about your insurance policy..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if user_prompt:

    # Add user message
    add_message(
        "user",
        user_prompt,
    )

    # Display immediately
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate response
    if not st.session_state.uploaded_pdfs:
        response = (
            "Please upload at least one health insurance policy PDF "
            "before asking questions."
        )
    else:
        response = generate_response(user_prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save assistant response
    add_message(
        "assistant",
        response,
    )


#============================================================
#  Analyze policy summary
#============================================================


if st.session_state.policy_summary:
    summary_data = st.session_state.policy_summary
    metadata = summary_data.get("metadata", {})
    extraction = summary_data.get("extraction", {})
    st.subheader("Policy Summary")
    st.write(f"**Policy:** {metadata.get('policy_name') or summary_data.get('policy_name') or 'Unnamed policy'}")
    st.caption(f"Pages analyzed: {metadata.get('page_count', 'Unknown')}")
    if metadata.get("ocr_pages"):
        st.caption("OCR was used on pages: " + ", ".join(map(str, metadata["ocr_pages"])))
    for section_name, field_name, label_field in (
        ("Coverage", "coverage", "name"),
        ("Exclusions", "exclusions", "description"),
        ("Waiting periods", "waiting_periods", "condition"),
        ("Claims process", "claims_process", "step"),
        ("Renewal terms", "renewal_terms", "term"),
        ("Definitions", "definitions", "term"),
    ):
        items = extraction.get(field_name, [])
        with st.expander(f"{section_name} ({len(items)})", expanded=(section_name == "Coverage")):
            if not items:
                st.write("No supported items were extracted from the available text.")
            for item in items:
                st.markdown(f"**{item.get(label_field, 'Item')}**")
                detail = item.get("details") or item.get("duration") or item.get("meaning")
                if detail:
                    st.write(detail)
                if item.get("limit"):
                    st.write(f"Limit: {item['limit']}")
                for citation in item.get("evidence", []):
                    st.caption(f"Page {citation.get('page_number')}: “{citation.get('quote', '')}”")
    with st.expander("Scoring features (technical details)"):
        st.json(summary_data.get("ml_features", {}).get("values", {}))

