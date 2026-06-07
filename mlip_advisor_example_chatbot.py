"""
Example RAG chatbot configuration
    Current config: Recommends which MLIP to use for a given chemical system and property

This code sets the RAG parameters (the model, chunk size, retrival vector size (k), etc), 
defines the prompt template (the chatbots task definition) and passes them to rag_engine.py.

Modify this section for your own desired RAG-LLM chat bot application.
"""
import textwrap
import asyncio

# Simple setup where rag_engine.py must be in the same directory as present script
from rag_engine import (
    download_model, load_pdfs, build_qdrant_index,
    build_chain, serve_chat, RetrieverCallbackHandler,
)
from langchain_huggingface import HuggingFaceEmbeddings

#### Chat bot model and paramter configuration ####

MODEL_NAME = "phi"                  # Select which local model from MODELS dict in rag_engine.py to use
PDF_FOLDER = "mlip_pdfs"            # PDF folder for RAG documents (determined by download_mlip_papers.py)
COLLECTION_NAME = "mlip_literature" # name of cached Qdrant index
RETRIEVER_K = 6                     # number of relevant chunks to retrieve for each question
CHUNK_SIZE = 1500                   # Characters per text chunk
CHUNK_OVERLAP = 150                 # Character overlap between neigboring chunks
TEMPERATURE = 0.3                   # lower value for more deterministic results
MAX_TOKENS = 500                    # max length of generated answer
PORT = 5006                         # port for chatbot web UI
STOP_TOKENS = ["<|end|>"]           # Add strings that will stop the chatbots text generation

APP_TITLE = "MLIP Advisor"          # Name of chatbot

# Promt template for wrapping retrived context and user question:
# <|user|> Instructions ensure samll local models can provide more reliable responses
# {context} determined by retrival and augmentation at run time
# {question} input by user at runtime and determines {context}
PROMPT_TEMPLATE = textwrap.dedent(
    """\
<|user|>
You are a Machine Learning Interatomic potential (MLIP) advisor. 
You are to use the MLIP literature to advise on which MLIP should be used for a given chemical system and property to model.
You should provide a brief and concise explanation for your recommendation, with pros and cons. 
You can advise up to three models and they should be given in order of preference.
When you cite a source, refer to it ONLY by the exact PDF filename shown in
brackets in the context, e.g. [Source: GRACE_2508.17936.pdf]. Do NOT produce a
"References" or "Citations" list, and never invent author names, journal names,
years, volumes, or page numbers. If a filename is not in the context, do not cite it.
Use only facts present in the context. 
Do not state what a model's name or acronym stands for unless the context defines it. 
If the context doesn't support a claim, say so rather than inventing detail.
Each cited claim must come from the named source; do not mix details across sources.

Context:
{context}

Question:
{question}
<|end|>
<|assistant|>
"""
)

#### Main ####

if __name__ == "__main__":
    # Setup model, embedding space and document store
    model_path = download_model(MODEL_NAME)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    documents = load_pdfs(PDF_FOLDER)
    qdrant = build_qdrant_index(documents, COLLECTION_NAME, embedding, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # callback defines the per message execution: Build the chain and generate answer
    async def callback(contents, user, instance):
        handler = RetrieverCallbackHandler(instance)
        chain = build_chain(
            qdrant=qdrant,
            model_path=model_path,
            retriever_callbacks=[handler],
            input_prompt_template=PROMPT_TEMPLATE,
            retriever_k=RETRIEVER_K,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop_tokens=STOP_TOKENS,
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, chain.invoke, contents)
        instance.send(result, user=MODEL_NAME, avatar="🌳", respond=False)

    # Loading web UI for input questions
    serve_chat(callback_fn=callback, title=APP_TITLE, port=PORT)