"""
NOC Intelligence — Conversational AI Assistant (Stage 9)
Independent Portfolio Demo

A conversational chatbot interface for the Hybrid Relational-Semantic Question
Answering System with Deterministic Routing, Grounded Synthesis, Active Guardrail
Verification, and Operational Identifier Masking.

STRICT CONTRACT:
- Only renders allowlisted SafePresentationView fields.
- Fails closed on unverified or suppressed answers.
- 100% synthetic demonstration data; zero proprietary records.
"""

import streamlit as st
from typing import Optional, Dict, Any, List

from src.ui.presentation_adapter import (
    PresentationAdapter,
    SafePresentationView,
    SafeCitationView,
)
from src.ui.pipeline_service import (
    execute_safe_query,
    check_ollama_availability,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NOC Intelligence — AI Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Clean typography and professional styling */
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.15rem;
}
.sub-title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #0284C7;
    margin-bottom: 0.5rem;
}
.synthetic-banner {
    background-color: #F8FAFC;
    border-left: 4px solid #0EA5E9;
    padding: 0.55rem 0.9rem;
    font-size: 0.85rem;
    color: #475569;
    margin-bottom: 1rem;
    border-radius: 0 4px 4px 0;
}
.status-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}
.status-verified {
    background-color: #DEF7EC;
    color: #03543F;
    border: 1px solid #BCF0DA;
}
.status-clarification {
    background-color: #FEF08A;
    color: #713F12;
    border: 1px solid #FDE047;
}
.status-suppressed {
    background-color: #FDE8E8;
    color: #9B1C1C;
    border: 1px solid #FBD5D5;
}
.status-unsupported {
    background-color: #F3F4F6;
    color: #374151;
    border: 1px solid #E5E7EB;
}
.citation-box {
    background-color: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 0.4rem 0.65rem;
    margin-top: 0.35rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
}
.metric-pill {
    font-family: monospace;
    font-size: 0.8rem;
    background-color: #F1F5F9;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    color: #475569;
}
.starter-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Example Query Starters (Verified against synthetic dataset)
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_PROMPTS = [
    {
        "label": "🎫 Ticket Status",
        "query": "Ticket INC-2025-00001 สถานะเป็นอะไร",
        "desc": "SQL direct lookup"
    },
    {
        "label": "🔧 Work Order Detail",
        "query": "Show resolution for work order WO-700165",
        "desc": "SQL work order lookup"
    },
    {
        "label": "🔗 Bridge Relationship",
        "query": "ตั๋ว INC-2025-00042 เชื่อมโยงกับใบงานช่างใบไหนบ้าง",
        "desc": "SQL bridge traversal"
    },
    {
        "label": "🌐 Circuit Ownership",
        "query": "วงจร CKT-880001 เป็นของบริษัทอะไร",
        "desc": "SQL circuit lookup"
    },
    {
        "label": "📊 Ticket Aggregation",
        "query": "Apex Logistics มีตั๋วแจ้งปัญหาทั้งหมดกี่รายการ",
        "desc": "SQL customer ticket count"
    },
    {
        "label": "🔍 Semantic Troubleshooting",
        "query": "ไฟฟ้าดับเป็นวงกว้าง การไฟฟ้า PEA ดับไฟซ่อมหม้อแปลง",
        "desc": "ChromaDB dense search"
    },
    {
        "label": "⚡ Hybrid Investigation",
        "query": "ค้นหาปัญหา fiber degradation ของบริษัท Apex Logistics",
        "desc": "SQL + Vector hybrid"
    },
    {
        "label": "❓ Ambiguous Query",
        "query": "ตั๋วสถานะเป็นอะไร",
        "desc": "Clarification trigger"
    },
    {
        "label": "🚫 Out-of-Domain",
        "query": "พยากรณ์อากาศที่กรุงเทพวันนี้ฝนจะตกไหม",
        "desc": "Unsupported rejection"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sidebar Configuration & Architecture Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Assistant Settings")

    provider_choice = st.radio(
        "Synthesis LLM Provider:",
        options=["Deterministic Fake Provider", "Local Ollama (typhoon2:8b)"],
        index=0,
        help="Deterministic Fake Provider runs completely offline with 0 network calls. Local Ollama connects to http://127.0.0.1:11434."
    )

    provider_type = "fake" if "Fake" in provider_choice else "ollama"

    if provider_type == "ollama":
        with st.spinner("Checking local Ollama status..."):
            ollama_status = check_ollama_availability()
            if ollama_status["available"]:
                st.success(f"🟢 {ollama_status['message']}")
            else:
                st.warning(f"🟡 {ollama_status['message']}")
    else:
        st.info("🔒 Offline Fake Mode: Zero external/cloud network calls.")

    st.markdown("---")
    st.markdown("### 🛡️ Privacy & Guardrails")

    masking_enabled = st.checkbox(
        "Identifier Masking Active",
        value=True,
        help="Pseudonymizes tickets [TICKET-***], work orders [WO-***], circuits [CIRCUIT-***], and customer names."
    )

    st.markdown("---")
    st.markdown("### 🗑️ Conversation Actions")

    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_chat_button"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Dataset Specifications")
    st.markdown(
        """
        - **Relational Tables:** 6 SQLite tables
        - **Total Records:** 15,050 records
        - **Vector Embeddings:** 5,500 documents (ChromaDB)
        - **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2`
        - **Privacy Guarantee:** 100% Synthetic Fictional Data
        """
    )

    st.markdown("---")
    st.markdown("### ⚠️ Known Masking Gaps")
    st.caption(
        "Regex masking covers Tickets, Work Orders, Circuits, and Customers. "
        "Technician aliases, contact persons, phone numbers, site names, and IP addresses "
        "are not masked by regex; therefore, raw operational database rows are strictly "
        "prohibited from being exposed to the presentation layer."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Header & Synthetic Data Disclaimer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🛡️ NOC Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Network Operations Assistant</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="synthetic-banner">
        <strong>⚠️ Synthetic Demonstration Data Disclaimer:</strong>
        All trouble tickets, work orders, circuit identifiers, enterprise corporate names,
        and operational activities displayed in this system are 100% synthetically generated.
        Zero company, customer, or proprietary network data is utilized.
    </div>
    """,
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assistant Message Renderer (Strict Presentation Boundary)
# ─────────────────────────────────────────────────────────────────────────────

def render_assistant_view(view: SafePresentationView):
    """
    Renders an assistant response card strictly adhering to the Safe Presentation Contract.
    Never exposes internal diagnostic structures, unmasked keys, or raw SQL rows.
    """
    # Header: Status badge and single-query execution latency
    col_badge, col_lat = st.columns([3, 1])

    with col_badge:
        if not view.is_safe_to_present:
            badge_class = "status-suppressed"
            badge_text = f"🛡️ SAFE SUPPRESSION ({view.status})"
        elif view.status == "VERIFIED":
            badge_class = "status-verified"
            badge_text = "✅ VERIFIED EVIDENCE"
        elif "CLARIFICATION" in view.status:
            badge_class = "status-clarification"
            badge_text = "❓ CLARIFICATION REQUIRED"
        elif "UNSUPPORTED" in view.status:
            badge_class = "status-unsupported"
            badge_text = "🚫 OUT-OF-DOMAIN REJECTION"
        else:
            badge_class = "status-verified"
            badge_text = f"ℹ️ {view.status}"

        st.markdown(f'<span class="status-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    with col_lat:
        st.markdown(
            f'<div style="text-align: right;"><span class="metric-pill">⏱️ {view.execution_time_ms:.2f} ms</span></div>',
            unsafe_allow_html=True
        )

    # Sanitized Final Answer (safe fallback rendered if is_safe_to_present is False)
    st.markdown(view.final_answer)

    # Citations Panel (Only sanitized citations rendered)
    if view.citations:
        st.markdown("<div style='font-size:0.85rem; font-weight:600; color:#475569; margin-top:0.4rem;'>📚 Verified Sources:</div>", unsafe_allow_html=True)
        for i, c in enumerate(view.citations, start=1):
            ver_icon = "✅" if c.is_verified else "⚠️"
            st.markdown(
                f"""
                <div class="citation-box">
                    <strong>[{i}] {c.identifier}</strong> &nbsp;
                    <span style="color: #64748B;">({c.entity_type})</span> &nbsp;
                    <span style="color: #059669;">{ver_icon} Verified</span><br/>
                    <span style="color: #475569; font-size: 0.8rem;">{c.label}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Safe Warnings
    if view.warnings:
        for w in view.warnings:
            st.caption(f"⚠️ {w}")

    # Restrained Safe Telemetry (Strictly Allowlisted)
    with st.expander("🔍 Safe Telemetry", expanded=False):
        diag = view.get_safe_diagnostics()
        d_cols = st.columns(4)
        keys = list(diag.keys())
        for idx, k in enumerate(keys):
            d_cols[idx % 4].metric(label=k, value=str(diag[k]))
        st.caption(
            "Telemetry strictly filtered by Safe Presentation Boundary. "
            "Raw database rows, reverse pseudonymization maps, and internal triples are blocked."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Conversation State & Chat History
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# If conversation is empty, display welcome guide and quick starter prompts
starter_clicked: Optional[str] = None
if len(st.session_state["messages"]) == 0:
    st.markdown(
        """
        <div class="starter-card">
            <strong>👋 Welcome to NOC Intelligence!</strong><br/>
            I am your AI-powered Network Operations Center assistant. Ask me questions about
            trouble ticket statuses, work order resolutions, leased circuits, or technical network symptoms.
            <div style="margin-top: 0.4rem; font-size: 0.85rem; color: #64748B;">
                Try asking a question below, or select a starter prompt:
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display starter prompts as responsive buttons in a 3-column grid
    cols = st.columns(3)
    for idx, prompt_info in enumerate(EXAMPLE_PROMPTS):
        target_col = cols[idx % 3]
        if target_col.button(
            f"{prompt_info['label']}",
            help=prompt_info['query'],
            key=f"starter_btn_{idx}",
            use_container_width=True
        ):
            starter_clicked = prompt_info["query"]

# Render previous chat messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🛡️"):
            render_assistant_view(msg["view"])


# ─────────────────────────────────────────────────────────────────────────────
# 7. Chat Input & Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

chat_input_text = st.chat_input("Ask a question about tickets, work orders, circuits, or technical symptoms...")
active_query = chat_input_text or starter_clicked

if active_query:
    clean_query = active_query.strip()
    if clean_query:
        # 1. Add user message to conversation history & render
        st.session_state["messages"].append({"role": "user", "content": clean_query})
        with st.chat_message("user"):
            st.markdown(clean_query)

        # 2. Execute query through backend pipeline with spinner
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Analyzing operational records and running active guardrails..."):
                view: SafePresentationView = execute_safe_query(
                    query=clean_query,
                    provider_type=provider_type,
                    masking_enabled=masking_enabled
                )
                render_assistant_view(view)

        # 3. Add assistant response to conversation history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": view.final_answer,
            "view": view
        })
