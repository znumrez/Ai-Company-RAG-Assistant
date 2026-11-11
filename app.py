import html
import streamlit as st

from pathlib import Path
import shutil

from src.r import ask_question
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import embed_documents
from src.vector_store import save_vectors, load_vectors

from database import create_table, save_chat, get_chat_history


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nexa AI | Company Knowledge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create the SQLite table if it does not exist
create_table()


# =========================================================
# 2. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None

if "page" not in st.session_state:
    st.session_state.page = "Chat"


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Urbanist:wght@400;500;600;700;800&display=swap');

/* Apply Urbanist to text only (NOT to every element), so Streamlit's
   icon font keeps working and icon names don't show up as text */
html, body, .stApp, .stMarkdown, p, label, button, input, textarea, h1, h2, h3, h4 {
    font-family: 'Urbanist', sans-serif;
}

/* Keep Streamlit's icon font so icons don't turn into text */
[data-testid="stIconMaterial"],
.material-icons,
.material-symbols-rounded,
span[class*="material"] {
    font-family: 'Material Symbols Rounded' !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #f1eaff 0%, transparent 30%),
        radial-gradient(circle at 90% 15%, #ffeaf5 0%, transparent 28%),
        linear-gradient(135deg, #f8f5ff, #fff7fb);
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.72);
    border-right: 1px solid rgba(160, 130, 190, 0.15);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero-card {
    background: linear-gradient(120deg, #8b6de9, #c18be9, #e5a6d2);
    border-radius: 25px;
    padding: 30px;
    color: white;
    box-shadow: 0 12px 35px rgba(140, 105, 200, 0.18);
    margin-bottom: 25px;
}

.hero-card h1 {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero-card p {
    font-size: 16px;
    opacity: 0.95;
}

.stat-card {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(180,150,210,0.2);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(130,100,170,0.06);
}

.stat-label {
    color: #82748f;
    font-size: 14px;
    font-weight: 600;
}

.stat-value {
    color: #554477;
    font-size: 27px;
    font-weight: 800;
    margin-top: 5px;
}

.section-heading {
    color: #554477;
    font-size: 22px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 12px;
}

.history-card {
    background: rgba(255,255,255,0.8);
    border: 1px solid rgba(180,150,210,0.2);
    border-radius: 17px;
    padding: 18px;
    margin-bottom: 12px;
}

.history-question {
    color: #594878;
    font-weight: 700;
    font-size: 16px;
}

.history-date {
    color: #9486a2;
    font-size: 12px;
    margin-top: 5px;
}

.answer-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid rgba(180,150,210,0.2);
    border-radius: 18px;
    padding: 22px;
    margin-top: 15px;
    color: #514366;
    line-height: 1.7;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #d9c9ed;
    background: white;
    color: #7658a4;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #f1eaff;
    border-color: #b99bdc;
    color: #65448e;
}

.stChatMessage {
    background: rgba(255,255,255,0.55);
    border-radius: 16px;
    border: 1px solid rgba(180,150,210,0.12);
    padding: 12px;
}

.stChatInput {
    border-radius: 15px;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6);
    border: 1px solid rgba(180,150,210,0.18);
    border-radius: 14px;
    padding: 10px;
}

/* Keep the upload area clean inside the expanded sidebar */

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
    width: 100%;
    overflow: hidden;
}

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] label {
    display: none !important;
}

section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small {
    overflow-wrap: anywhere;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def get_document_count():

    documents_folder = Path("data/documents")

    if not documents_folder.exists():
        return 0

    return len(list(documents_folder.glob("*.pdf")))


def get_chunk_count():

    return len(load_vectors())


def display_hero(title, subtitle):

    st.markdown(f"""
    <div class="hero-card">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 15px 5px 20px 5px;">
        <h2 style="color:#7658a4; font-weight:800;">
            ✦ Nexa AI
        </h2>
        <p style="color:#9586a5; font-size:13px;">
            Company Knowledge Assistant
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_page = st.radio(
        "NAVIGATION",
        ["Chat", "History"],
        index=0 if st.session_state.page == "Chat" else 1
    )

    st.session_state.page = selected_page

    st.markdown("---")

    if selected_page == "Chat":

        st.markdown("### 📄 Knowledge Base")

        uploaded_file = st.file_uploader(
            label="Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help="Upload an HR policy, employee handbook, or other company document."
        )

        if uploaded_file is not None:

            if st.button(
                "Process Document",
                use_container_width=True
            ):

                documents_folder = Path("data/documents")
                temp_folder = Path("data/.processing")

                documents_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                temp_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )

                temp_file_path = temp_folder / uploaded_file.name
                final_file_path = documents_folder / uploaded_file.name

                try:

                    with st.spinner("Reading and processing your document..."):

                        # Save the new PDF temporarily.
                        # The active document is not changed until processing succeeds.

                        with open(temp_file_path, "wb") as file:
                            file.write(uploaded_file.getbuffer())

                        # Read and split only the newly uploaded PDF

                        documents = load_pdf(temp_file_path)
                        chunks = split_documents(documents)

                        if not chunks:

                            st.warning(
                                "No readable text was found in this PDF. "
                                "Your previous active document has been kept."
                            )

                            temp_file_path.unlink(missing_ok=True)

                        else:

                            # Generate embeddings for the new PDF

                            embedded_chunks = embed_documents(chunks)

                            # Replace the old vector index with this PDF's vectors

                            save_vectors(embedded_chunks)

                            # Remove previous PDFs only after successful processing

                            for old_pdf in documents_folder.glob("*.pdf"):
                                old_pdf.unlink()

                            # Move successfully processed PDF into active folder

                            shutil.move(
                                str(temp_file_path),
                                str(final_file_path)
                            )

                            st.success(
                                f"{uploaded_file.name} is now the active knowledge document."
                            )

                            st.rerun()

                except Exception as error:

                    if temp_file_path.exists():
                        temp_file_path.unlink()

                    st.error(
                        f"Document processing failed: {error}"
                    )

        st.markdown("---")

        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):

            st.session_state.messages = []
            st.session_state.selected_history_id = None
            st.rerun()

        st.markdown("---")

        st.caption("Nexa AI · Powered by Gemini")


# =========================================================
# 6. CHAT PAGE
# =========================================================

if st.session_state.page == "Chat":

    display_hero(
        "Your Knowledge, One Conversation Away ✦",
        "Ask questions and get answers from your company's documents."
    )

    # Dashboard statistics

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">PDF DOCUMENTS</div>
            <div class="stat-value">{get_document_count()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">INDEXED CHUNKS</div>
            <div class="stat-value">{get_chunk_count()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">AI MODEL</div>
            <div class="stat-value">Gemini</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">💬 Chat Assistant</div>',
        unsafe_allow_html=True
    )

    # Welcome message

    if not st.session_state.messages:

        st.markdown("""
        <div class="answer-card">
            <h3 style="color:#7658a4;">Hello! 👋</h3>
            <p>
                I'm Nexa AI, your company knowledge assistant.
                Ask me about company policies, employees, departments,
                leave procedures, and other information in your PDFs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Display only the current conversation

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Chat input

    question = st.chat_input(
        "Ask something about your company documents..."
    )

    if question:

        # Display user's question

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)

        # Generate AI answer

        with st.chat_message("assistant"):

            with st.spinner("Searching company knowledge..."):

                try:

                    answer = ask_question(question)

                except Exception as error:

                    answer = (
                        "Something went wrong while generating the answer. "
                        "Please check your configuration and try again."
                    )

                    st.error(str(error))

                st.markdown(answer)

        # Save question and answer permanently in SQLite

        save_chat(question, answer)

        # Save in current session too

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()


# =========================================================
# 7. SEPARATE HISTORY PAGE
# =========================================================

elif st.session_state.page == "History":

    display_hero(
        "Conversation History 📚",
        "Your previously saved questions and AI responses."
    )

    history_records = get_chat_history()

    if not history_records:

        st.info(
            "No conversations saved yet. Start chatting with Nexa AI "
            "and your conversations will appear here."
        )

    else:

        st.markdown(
            '<div class="section-heading">Your Conversations</div>',
            unsafe_allow_html=True
        )

        st.caption(
            f"{len(history_records)} saved question-and-answer records"
        )

        # Display newest conversations first

        for record in history_records:

            chat_id, question, answer, created_at = record

            safe_question = html.escape(question)
            safe_date = html.escape(str(created_at))

            st.markdown(f"""
            <div class="history-card">
                <div class="history-question">
                    💬 {safe_question}
                </div>
                <div class="history-date">
                    {safe_date}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # View button for each conversation

            if st.button(
                "View Conversation",
                key=f"history_{chat_id}"
            ):

                st.session_state.selected_history_id = chat_id

            # Display selected conversation immediately below its button

            if st.session_state.selected_history_id == chat_id:

                st.markdown(
                    '<div class="section-heading">Selected Conversation</div>',
                    unsafe_allow_html=True
                )

                st.caption(f"Saved on: {created_at}")

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    st.markdown(answer)

                if st.button(
                    "Close Conversation",
                    key=f"close_{chat_id}"
                ):

                    st.session_state.selected_history_id = None
                    st.rerun()