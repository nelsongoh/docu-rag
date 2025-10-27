# app.py
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from src.rag_chain import DocumentRAG, RAGError
from src.document_manager import DocumentManager

st.set_page_config(
    page_title="Documentation Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize document manager
doc_manager = DocumentManager()

# Initialize RAG (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_rag(collection_name: str):
    """Load RAG instance for a specific collection."""
    return DocumentRAG(
        collection_name=collection_name,
        llm_client="anthropic",
        model="claude-sonnet-4",
        prompt_preset="payment_api"
    )

# Get available documentation options
@st.cache_data
def get_documentation_options():
    """Get available documentation options from config."""
    try:
        return doc_manager.get_documentation_options()
    except Exception as e:
        st.error(f"Error loading documentation options: {e}")
        return {}

# Title
st.title("📄 Documentation Assistant")
st.markdown("Ask me anything about the available documentation sources.")

# Sidebar with info and document selection
with st.sidebar:
    st.header("📚 Select Documentation")

    # Get available documentation options
    doc_options = get_documentation_options()

    if not doc_options:
        st.error("No documentation sources available. Please configure doc_sources.json and vector_store_config.json")
        st.stop()

    # Create a dropdown to select documentation
    selected_doc = st.selectbox(
        "Choose documentation to query:",
        options=list(doc_options.keys()),
        key="selected_documentation"
    )

    # Get the collection name for the selected documentation
    collection_name = doc_options[selected_doc]

    st.markdown(f"**Selected:** `{collection_name}`")

    st.divider()

    st.header("About")
    st.write("""
    This assistant uses Retrieval-Augmented Generation (RAG) to answer questions about the available documentation sources.

    **Features:**
    - 🔍 Semantic search across docs
    - 📚 Citation tracking
    - 💰 Cost estimation before queries
    - ⚡ Fast response generation
    """)

    st.header("Example Questions")
    example_questions = [
        "How do I handle failed payment retries?",
        "What's the difference between Payment Intent and Charge?",
        "How do webhooks work for subscriptions?",
        "What are common payment errors and how to handle them?",
        "How do I implement idempotency for payments?"
    ]

    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.current_question = q
            st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for confirmation
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "cost_estimate" not in st.session_state:
    st.session_state.cost_estimate = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📌 View Sources"):
                for source in message["sources"]:
                    st.write(f"{source['id']} **{source['title']}**")

# Handle question from sidebar
if "current_question" in st.session_state:
    query = st.session_state.current_question
    del st.session_state.current_question
    st.session_state.pending_query = query
    st.session_state.awaiting_confirmation = True
else:
    query = st.chat_input("Ask a question about the selected documentation...")
    if query:
        st.session_state.pending_query = query
        st.session_state.awaiting_confirmation = True
        st.rerun()

# If we have a pending query awaiting confirmation
if st.session_state.awaiting_confirmation and st.session_state.pending_query:
    query = st.session_state.pending_query

    # Add user message to chat history (only once)
    user_message_exists = any(
        msg.get("role") == "user" and msg.get("content") == query and msg.get("_pending") == True
        for msg in st.session_state.messages
    )

    if not user_message_exists:
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "_pending": True  # Mark this message as pending confirmation
        })

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    try:
        # Load RAG for the selected collection
        rag = load_rag(collection_name)

        # Generate response with cost estimation UI
        with st.chat_message("assistant"):
            # Step 1: Estimate cost (only once)
            if st.session_state.cost_estimate is None:
                with st.spinner("Estimating cost..."):
                    try:
                        cost_estimate = rag.estimate_cost(query)
                        st.session_state.cost_estimate = cost_estimate
                    except RAGError as e:
                        st.error(f"❌ Error: {e}")
                        st.session_state.awaiting_confirmation = False
                        st.session_state.pending_query = None
                        st.session_state.cost_estimate = None
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()
                        st.stop()

            cost_estimate = st.session_state.cost_estimate

            # Step 2: Display cost estimate and ask for confirmation
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Estimated Cost",
                    f"${cost_estimate['total_cost_estimate']:.6f}",
                    delta=None
                )

            with col2:
                st.metric(
                    "Model",
                    cost_estimate['model']
                )

            # Display token breakdown
            st.caption(
                f"📊 Tokens: ~{cost_estimate['input_tokens_estimate']} input, "
                f"~{cost_estimate['output_tokens_estimate']} output"
            )

            # Step 3: Ask for confirmation before proceeding
            # Only show buttons if user hasn't made a choice yet
            if not st.session_state.get("_confirmation_made", False):
                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("✅ Yes, generate response", use_container_width=True, key="confirm_yes"):
                        # Mark that a choice has been made (prevents button from showing again)
                        st.session_state._confirmation_made = True
                        st.rerun()

                with col_no:
                    if st.button("❌ No, cancel", use_container_width=True, key="confirm_no"):
                        st.info("Query cancelled. You can ask another question.")
                        # Clear the state
                        st.session_state.awaiting_confirmation = False
                        st.session_state.pending_query = None
                        st.session_state.cost_estimate = None
                        st.session_state._confirmation_made = False
                        # Remove the user message from history
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()
                        st.rerun()
            else:
                # User has made a choice - show processing/results
                if st.session_state.get("_user_confirmed", None) is None:
                    # Set confirmation flag on first run after button click
                    st.session_state._user_confirmed = True

                if st.session_state._user_confirmed:
                    # Proceed with generation
                    with st.spinner("Generating response..."):
                        try:
                            # Generate response without confirmation
                            result = rag.generate_response(query, stream=False, require_confirmation=False)
                            full_response = result['answer']
                            sources = result['sources']
                            usage = result['usage']

                            # Display response
                            st.markdown(full_response)

                            # Show sources
                            with st.expander("📌 View Sources"):
                                for source in sources:
                                    st.write(f"{source['id']} **{source['title']}** (chunk {source['chunk_id']})")

                            # Show actual usage vs estimate
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Input Tokens", usage['input_tokens'])
                            with col2:
                                st.metric("Output Tokens", usage['output_tokens'])
                            with col3:
                                # Calculate actual cost based on model pricing
                                from config.constants import CLAUDE_PRICING
                                model_key = rag.model_key
                                if model_key in CLAUDE_PRICING:
                                    pricing = CLAUDE_PRICING[model_key]
                                    actual_cost = (
                                        (usage['input_tokens'] / 1_000_000) * pricing["input"] +
                                        (usage['output_tokens'] / 1_000_000) * pricing["output"]
                                    )
                                    st.metric("Actual Cost", f"${actual_cost:.6f}")

                            # Update the pending user message to mark it as confirmed
                            for msg in st.session_state.messages:
                                if msg.get("role") == "user" and msg.get("content") == query:
                                    msg["_pending"] = False
                                    break

                            # Add assistant message to history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": full_response,
                                "sources": sources
                            })

                            # Clear the state for the next query
                            st.session_state.awaiting_confirmation = False
                            st.session_state.pending_query = None
                            st.session_state.cost_estimate = None
                            st.session_state._confirmation_made = False
                            st.session_state._user_confirmed = None

                        except RAGError as e:
                            st.error(f"❌ Error: {e}")
                            # Remove the user message if generation failed
                            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                                st.session_state.messages.pop()
                            # Clear the state
                            st.session_state.awaiting_confirmation = False
                            st.session_state.pending_query = None
                            st.session_state.cost_estimate = None
                            st.session_state._confirmation_made = False
                            st.session_state._user_confirmed = None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        # Remove the user message if an error occurred
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()
        st.session_state.awaiting_confirmation = False
        st.session_state.pending_query = None
        st.session_state.cost_estimate = None

# Footer
st.markdown("---")
st.caption("Built with Claude Sonnet 4, ChromaDB, and Streamlit | Production-ready RAG system")