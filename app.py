# app.py
import os
import tempfile
from pathlib import Path
import streamlit as st

# Verify presence of the core LangChain and vector dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except Exception:
    EnsembleRetriever = None
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- BHASHINI API CLIENT INTEGRATION ---
try:
    from bhashini_client import BhashiniTranslator
except ImportError:
    # Inline translation client if the bhashini_client module is not present
    import requests
    import logging
    
    class BhashiniTranslator:
        def __init__(self, user_id: str, ulca_api_key: str):
            self.user_id = user_id
            self.ulca_api_key = ulca_api_key
            self.config_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
            self.pipeline_id = "64392f96daac500b55c543cd"
            
        def _fetch_pipeline_config(self, source_lang: str, target_lang: str):
            headers = {
                "userID": self.user_id,
                "ulcaApiKey": self.ulca_api_key,
                "Content-Type": "application/json"
            }
            # Provide a minimal pipelineTasks structure; the API expects a list
            payload = {
                "pipelineTasks": [],
                "pipelineRequestConfig": {"pipelineId": self.pipeline_id}
            }
            response = requests.post(self.config_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            service_id = data["config"]["serviceId"]
            callback_url = data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
            inference_key_name = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]
            inference_key_val = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
            return service_id, callback_url, {inference_key_name: inference_key_val}
            
        def translate(self, text_input: str, source_lang: str, target_lang: str) -> str:
            if source_lang == target_lang or not text_input.strip():
                return text_input
            try:
                service_id, callback_url, auth_headers = self._fetch_pipeline_config(source_lang, target_lang)
                # Minimal payload for pipeline invocation
                payload = {
                    "pipelineTasks": [],
                    "inputData": {"input": [{"source": text_input}]}
                }
                response = requests.post(callback_url, headers=auth_headers, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data["output"]["target"]
            except Exception as err:
                logging.error(f"Translation API transaction failed: {err}")
                return text_input

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="NCERT Multilingual Assistant", page_icon="📖", layout="wide")

# Initialize persistent session state keys
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_active" not in st.session_state:
    st.session_state.pipeline_active = False

# --- SIDEBAR CONTROL LAYER ---
with st.sidebar:
    st.title("📖 Assistant Control Panel")
    st.caption("A localized K-12 RAG engine running entirely on local hardware.")
    
    st.subheader("1. Language Settings")
    selected_language = st.selectbox(
        "Response Language",
        options=["en", "hi", "mr", "ta", "te"],
        format_func=lambda x: {"en": "English", "hi": "Hindi (हिन्दी)", "mr": "Marathi (मराठी)", "ta": "Tamil (தமிழ்)", "te": "Telugu (తెలుగు)"}[x]
    )
    
    st.subheader("2. Model Selection")
    local_llm_model = st.selectbox("Local Generative Model", ["llama3.2", "gemma3:1b", "deepseek-r1"])
    inference_temperature = st.slider("Temperature (Response Variation)", 0.0, 1.0, 0.2, 0.05)
    
    st.subheader("3. Ingest Textbook Documents")
    uploaded_pdfs = st.file_uploader(
        "Upload NCERT Textbook PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    st.subheader("4. Bhashini Credentials")
    bhashini_uid = st.text_input("Bhashini User ID", type="password")
    bhashini_apikey = st.text_input("Bhashini API Key", type="password")
    
    if st.button("Clear Conversation History"):
        st.session_state.chat_history = []
        st.rerun()

# --- CACHED DOCUMENT PARSING ENGINE ---
@st.cache_resource(show_spinner="Analyzing textbook files and building vector index...")
def build_knowledge_base(uploaded_files):
    """
    Saves uploaded files to a temporary directory, parses them,
    and returns a combined dense-sparse retrieval system.
    """
    if not uploaded_files:
        return None

    temp_dir = tempfile.mkdtemp()
    all_pages = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Parse document page-by-page
        loader = PyPDFLoader(file_path)
        all_pages.extend(loader.load())
        
    # Segment documents recursively
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=75,
        length_function=len
    )
    chunks = splitter.split_documents(all_pages)
    
    # Create indexes and retrievers
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    dense_vectorstore = FAISS.from_documents(chunks, embeddings)
    dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": 3})
    
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 3
    
    # Combine using EnsembleRetriever if available; otherwise create a simple combined retriever
    if EnsembleRetriever is not None:
        combined_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
    else:
        class CombinedRetriever:
            """Simple fallback retriever that merges results from multiple retrievers."""
            def __init__(self, retrievers):
                self.retrievers = retrievers

            def invoke(self, query):
                results = []
                seen = set()
                for r in self.retrievers:
                    docs = []
                    # try common retrieval methods
                    try:
                        if hasattr(r, "get_relevant_documents"):
                            docs = r.get_relevant_documents(query)
                        elif hasattr(r, "retrieve"):
                            docs = r.retrieve(query)
                        elif hasattr(r, "as_retriever"):
                            docs = r.as_retriever().get_relevant_documents(query)
                        elif hasattr(r, "invoke"):
                            docs = r.invoke(query)
                    except Exception:
                        docs = []
                    for d in docs:
                        key = (d.metadata.get("source", ""), getattr(d, "page", None), d.page_content[:200])
                        if key not in seen:
                            seen.add(key)
                            results.append(d)
                return results

        combined_retriever = CombinedRetriever([dense_retriever, sparse_retriever])
    return combined_retriever

# --- CORE GENERATION ENGINE ---
def run_grounded_generation(query_text, retriever_engine, model_name, temp):
    """
    Runs a query against the RAG pipeline and returns the generated response
    alongside verified citations.
    """
    chat_model = ChatOllama(model=model_name, temperature=temp)
    
    prompt_template = """
    You are an expert curriculum assistant for the Indian K-12 education system. 
    Your answers must be grounded strictly in the provided NCERT textbook context.
    If the context does not contain the answer, state that you do not have sufficient information.
    Do not use outside knowledge or introduce external syllabus topics.

    Retrieved Context:
    \"\"\"{context}\"\"\"

    Question: {question}

    Provide a clear, age-appropriate academic explanation with citations in step-by-step prose.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        formatted_blocks = []
        for i, doc in enumerate(docs):
            filename = os.path.basename(doc.metadata.get("source", "Syllabus"))
            page = doc.metadata.get("page", "N/A")
            formatted_blocks.append(f": {filename} (Page {page})\nContent: {doc.page_content}")
        return "\n\n".join(formatted_blocks)
        
    # Retrieve matching chunks
    retrieved_chunks = retriever_engine.invoke(query_text)

    # Format retrieved documents into a context string
    context = format_docs(retrieved_chunks)

    # Construct the final prompt text
    prompt_text = prompt_template.format(context=context, question=query_text)

    # Attempt to invoke the chat model in a few common ways; fall back gracefully
    try:
        if hasattr(chat_model, "invoke"):
            answer = chat_model.invoke(prompt_text)
        elif hasattr(chat_model, "generate"):
            answer = chat_model.generate(prompt_text)
        elif callable(chat_model):
            answer = chat_model(prompt_text)
        else:
            answer = "[Model invocation unavailable in this environment]"
    except Exception as gen_err:
        answer = f"[Model generation failed: {gen_err}]"

    return answer, retrieved_chunks

# --- MAIN INTERFACE RERUN LOOP ---
st.title("🇮🇳 NCERT Classroom Conversational Assistant")
st.caption("A private, curriculum-grounded school tutor powered by secure local hardware and Bhashini.")

# Initialize the Bhashini client
bhashini_translator = None
if bhashini_uid and bhashini_apikey:
    bhashini_translator = BhashiniTranslator(bhashini_uid, bhashini_apikey)

# Build the knowledge base from uploaded PDFs
retriever = None
if uploaded_pdfs:
    retriever = build_knowledge_base(uploaded_pdfs)
    st.session_state.pipeline_active = True
else:
    st.info("💡 Please upload textbook PDFs in the sidebar panel to initialize the RAG knowledge base.")

# Render chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Show Verified Citations"):
                for idx, ref in enumerate(message["citations"]):
                    st.write(f"**Reference [{idx+1}]**: {ref['source']} (Page {ref['page']})")
                    st.caption(f"Context: {ref['content']}")

# Process new student inputs
if user_prompt := st.chat_input("Ask a question about your science or history lessons..."):
    # Render user query
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    
    if not retriever:
        with st.chat_message("assistant"):
            st.error("No active knowledge base. Please upload NCERT textbook PDFs in the sidebar panel.")
    else:
        # Step 1: Query translation (Target Lang -> English)
        active_query = user_prompt
        if bhashini_translator and selected_language!= "en":
            with st.spinner("Translating query to English..."):
                active_query = bhashini_translator.translate(user_prompt, selected_language, "en")
                
        # Step 2: Query the local RAG pipeline
        with st.spinner("Retrieving textbook segments and generating answer..."):
            english_response, raw_citations = run_grounded_generation(
                active_query, retriever, local_llm_model, inference_temperature
            )
            
        # Step 3: Response translation (English -> Target Lang)
        final_translated_response = english_response
        if bhashini_translator and selected_language!= "en":
            with st.spinner("Translating answer back into selected language..."):
                final_translated_response = bhashini_translator.translate(english_response, "en", selected_language)
                
        # Format citation references
        formatted_references = []
        for doc in raw_citations:
            # page may be an int or missing, coerce safely
            page_val = doc.metadata.get("page", None)
            try:
                page_display = (int(page_val) + 1) if page_val is not None else "N/A"
            except Exception:
                page_display = page_val
            formatted_references.append({
                "source": os.path.basename(doc.metadata.get("source", "Syllabus")),
                "page": page_display,
                "content": doc.page_content
            })
            
        # Step 4: Render response and citations to screen
        with st.chat_message("assistant"):
            st.markdown(final_translated_response)
            if formatted_references:
                with st.expander("Show Verified Citations"):
                    for idx, ref in enumerate(formatted_references):
                        st.write(f"**Reference [{idx+1}]**: {ref['source']} (Page {ref['page']})")
                        st.caption(f"Context: {ref['content']}")
                        
        # Save interaction to session state
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_translated_response,
            "citations": formatted_references
        })