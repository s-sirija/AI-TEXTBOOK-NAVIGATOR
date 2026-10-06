import streamlit as st

from app.config import APP_TITLE, DEFAULT_TOP_K
from app.document_loader import load_document
from app.chunker import chunk_documents
from app.embeddings import get_embedding_model
from app.vector_store import TextbookVectorStore
from app.qa_engine import answer_question

from pathlib import Path
import streamlit as st

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "style.css"
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

st.set_page_config(page_title=APP_TITLE, page_icon='📚', layout='wide')
from pathlib import Path

def load_css():
    css_file = Path(__file__).parent / "style.css"

    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
st.markdown("""
<div class="hero-icon">📚</div>

<div class="main-title">
    <h1>AI Textbook Navigator</h1>
</div>

<div class="main-subtitle">
    Upload your textbook • Ask anything • Get answers from the right sections instantly
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.header('Settings')
    top_k = st.slider('Relevant sections', 2, 10, DEFAULT_TOP_K)
    chunk_size = st.slider('Chunk size', 400, 1600, 900, step=100)
    overlap = st.slider('Chunk overlap', 50, 300, 150, step=25)
    if st.button('🗑️ Clear indexed textbook', use_container_width=True):
        try:
            TextbookVectorStore().reset()
            for key in ('indexed_name', 'indexed_chunks'):
                st.session_state.pop(key, None)
            st.success('Index cleared.')
        except Exception as exc:
            st.error(f'Could not clear index: {exc}')

uploaded = st.file_uploader('Upload your textbook', type=['pdf', 'docx', 'txt'])

if uploaded is not None:
    if st.session_state.get('indexed_name') != uploaded.name:
        with st.spinner('Reading textbook and building the search index...'):
            try:
                documents = load_document(uploaded)
                if not documents:
                    st.error('No readable text was found in this file.')
                    st.stop()
                chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
                embedding_model = get_embedding_model()
                store = TextbookVectorStore()
                store.reset()
                with st.spinner('Creating semantic embeddings...'):
                    store.add_chunks(chunks, embedding_model)
                st.session_state['indexed_name'] = uploaded.name
                st.session_state['indexed_chunks'] = len(chunks)
                st.success(f'Indexed {uploaded.name} — {len(chunks)} searchable sections.')
            except Exception as exc:
                st.exception(exc)
                st.stop()

if st.session_state.get('indexed_name'):
    st.info(f"Current textbook: **{st.session_state['indexed_name']}** | Indexed sections: **{st.session_state.get('indexed_chunks', 0)}**")
    question = st.text_area('Ask a question', placeholder='Example: Explain the difference between supervised and unsupervised learning.', height=110)
    if st.button('🔎 Find & Explain', type='primary', use_container_width=True):
        if not question.strip():
            st.warning('Please enter a question.')
            st.stop()
        with st.spinner('Finding the most relevant textbook sections...'):
            store = TextbookVectorStore()
            embedding_model = get_embedding_model()
            results = store.search(question, embedding_model, top_k=top_k)
        if not results:
            st.warning('No relevant textbook sections were found.')
            st.stop()
        with st.spinner('Generating a textbook-grounded explanation...'):
            answer = answer_question(question, results)
        st.markdown("""
<div class="answer-card">
    <div class="answer-title">✨ AI Explanation</div>
""", unsafe_allow_html=True)

st.markdown(answer)

st.markdown("</div>", unsafe_allow_html=True)
st.subheader('📖 Relevant textbook sections')
for i, result in enumerate(results, start=1):
            metadata = result['metadata']
            page = metadata.get('page', 'Unknown')
            source = metadata.get('source', st.session_state['indexed_name'])
            distance = result.get('distance')
            label = f'Section {i} — {source} — Page {page}'
            if distance is not None:
                label += f' — distance: {distance:.4f}'
            with st.expander(label, expanded=(i <= 2)):
                st.write(result['text'])
else:
    st.markdown('''### How to use\n1. Upload a textbook.\n2. Wait for indexing to finish.\n3. Ask a natural-language question.\n4. The app retrieves the most relevant sections.\n5. Gemini explains the answer using those sections.\n\n**Tip:** For scanned/image-only PDFs, install Tesseract OCR separately.''')
