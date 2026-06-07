"""
RAG-LLM Engine — Core infrastructure for building RAG chatbots.
This file should not need modification for most use cases.
Import and use from your application-specific script.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from uuid import uuid4
import warnings
from pathlib import Path
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import shutil
import panel as pn

warnings.filterwarnings("ignore")

# === Available Models ===

MODELS = {
    "phi": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "deepseek": {
        "repo": "QuantFactory/deepseek-math-7b-rl-GGUF",
        "file": "deepseek-math-7b-rl.Q4_K_M.gguf",
    },
    "mistral": {
        "repo": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "file": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    },
}


def download_model(model_name):
    """Download a GGUF model and return the local path."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
    
    model_info = MODELS[model_name]
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    print(f"\n[GGUF] Downloading {model_info['repo']}/{model_info['file']}...")
    model_path = hf_hub_download(
        repo_id=model_info["repo"],
        filename=model_info["file"],
        cache_dir=str(cache_dir),
    )
    return model_path


def load_pdfs(pdf_folder_path):
    """Load all PDFs from a folder and return a list of LangChain documents."""
    documents = []
    pdf_count = 0
    for file in sorted(os.listdir(pdf_folder_path)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder_path, file)
            try:
                loader = PyMuPDFLoader(pdf_path)
                documents.extend(loader.load())
                pdf_count += 1
                print(f"  Loaded: {file}")
            except Exception as e:
                print(f"Failed to load {file}: {e}")
    print(f"\n Loaded {pdf_count} PDFs with {len(documents)} pages total")
    return documents


def build_qdrant_index(documents, collection_name, embedding, chunk_size=500, chunk_overlap=50):
    """Build or load a Qdrant vector store from documents."""
    qdrant_path = Path.home() / ".cache" / "qdrant" / collection_name
    
    REBUILD_INDEX = os.environ.get("REBUILD_INDEX", "false").lower() == "true"
    
    if REBUILD_INDEX and qdrant_path.exists():
        print("REBUILD_INDEX=true — Removing cached data")
        shutil.rmtree(qdrant_path)
    
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))
    
    existing_collections = [c.name for c in client.get_collections().collections]
    
    if collection_name in existing_collections:
        print(f"Loading existing Qdrant collection '{collection_name}' (skipping indexing)")
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
        )
        collection_info = client.get_collection(collection_name)
        print(f"Collection has {collection_info.points_count} vectors")
    else:
        print(f"Creating new Qdrant collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_documents = text_splitter.split_documents(documents)
        print(f"Split into {len(split_documents)} chunks (chunk_size={chunk_size})")
        
        texts = [doc.page_content for doc in split_documents]
        metadatas = [doc.metadata for doc in split_documents]
        
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            qdrant.add_texts(texts=batch_texts, metadatas=batch_meta)
            print(f"Indexed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
        
        print(f"Indexed {len(texts)} chunks into Qdrant")
    
    return qdrant


class RetrieverCallbackHandler(BaseCallbackHandler):
    """Callback handler that displays retrieved documents in the chat."""
    
    def __init__(self, instance):
        self.instance = instance
        self.retrieved_docs = []
    
    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self.retrieved_docs = documents
        if documents:
            docs_text = "\n\n---\n\n".join(
                [f"**Source ({doc.metadata.get('source', '?')}, page {doc.metadata.get('page', '?')}):**\n{doc.page_content[:200]}..."
                 for doc in documents]
            )
            self.instance.send(
                f"**Retrieved Context:**\n\n{docs_text}",
                user="Retriever",
                respond=False,
            )


def build_chain(qdrant, model_path, retriever_callbacks, input_prompt_template,
                retriever_k=3, temperature=0.3, max_tokens=300, stop_tokens=None):
    """Build a LangChain RAG chain."""
    
    retriever = qdrant.as_retriever(
        callbacks=retriever_callbacks,
        search_type="mmr",
        search_kwargs={"k": retriever_k},
    )
    
    default_stops = ["<|end|>", "<|user|>", "<|assistant|>"]
    stop = (stop_tokens or []) + default_stops
    
    llm_callback_manager = CallbackManager([])
    llm = LlamaCpp(
        model_path=str(model_path),
        callback_manager=llm_callback_manager,
        temperature=temperature,
        n_ctx=4096,
        max_tokens=max_tokens,
        verbose=False,
        echo=False,
        stop=stop,
    )
    
    transformed_prompt_template = PromptTemplate.from_template(input_prompt_template)
    
    def format_docs(docs):
        blocks = []
        for d in docs:
            src = Path(d.metadata.get("source", "?")).name
            page = d.metadata.get("page", "?")
            blocks.append(f"[Source: {src}, page {page}]\n{d.page_content}")
        return "\n\n".join(blocks)
    
    def show_docs(docs):
        for cb in retriever_callbacks:
            cb.on_retriever_end(docs, run_id=uuid4())
        return docs
    
    return (
        {
            "context": retriever | show_docs | format_docs,
            "question": RunnablePassthrough(),
        }
        | transformed_prompt_template
        | llm
    )


def serve_chat(callback_fn, title="RAG-LLM Chatbot", placeholder="Type your question...",
               port=5006, theme="dark"):
    """Launch the Panel chat interface."""
    
    pn.extension(design="material", theme=theme)
    
    chat_interface = pn.chat.ChatInterface(
        callback=callback_fn,
        placeholder_text=placeholder,
        styles={"background": "#1a1a2e"} if theme == "dark" else {},
    )
    
    template = pn.template.FastListTemplate(
        title=title,
        main=[chat_interface],
        accent_base_color="#6C63FF",
        header_background="#1a1a2e" if theme == "dark" else "#ffffff",
        theme=theme,
    )
    
    pn.serve({"/": template}, port=port, websocket_origin="*", show=False)