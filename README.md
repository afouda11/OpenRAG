# OpenRAG

Lightweight code for Retrieval-Augmented Generation (RAG) chatbots over a PDF document catalogue.  
OpenRAG pairs a local GGUF language model with a Qdrant vector store and a Panel web interface.

By defining a set of intructions in the prompt template, small locally deployed models can become reliable domain specific experts.

This project was inspired by following the following workshop tutorial:
https://uw-ssec-tutorials.readthedocs.io/en/latest/AI_Postdoc_Workshop/module2/index.html

Project files:

- `rag_engine.py` -- core library handling the model downloading, PDF ingestion, vector indexing, chain construction, and the chat UI. For basic chatbot applications, no modifications are required.
- `mlip_advisor_example_chatbot.py` -- An example usage script, which is a ready to go chatbot for advising which MLIP to use, given a chemical system and property type prompt from the user. Modify this code to make you own application.
- `download_mlip_papers.py` -- code for downloading the MLIP literature from arxiv for the MLIP advisor example chatbot.

## Requirements

- conda 
- Roughly 8 GB of free disk space for the model weights in present code

## Available models

`rag_engine.py` will download quantized GGUF models from Hugging Face on first run. 
Three models are included out of the box, the model is selected by `MODEL_NAME` in the chatbot script.
More model options can be included to the `MODEL` dict in `rag_engine.py`. 

| Name       | Model                                          | Size   |
|------------|------------------------------------------------|--------|
| phi        | Microsoft Phi-3-mini-4k-instruct (Q4)          | ~2 GB  |
| deepseek   | DeepSeek-Math-7B-RL (Q4_K_M)                   | ~4 GB  |
| mistral    | Mistral-7B-Instruct-v0.2 (Q4_K_M)              | ~4 GB  |

## Quick Start for MLIP Advisor Example

0. Clone repository, create and activate conda environment

```
git clone https://github.com/afouda11/OpenRAG.git
cd OpenRAG
conda env create -f environment.yml
conda activate openrag
```
The environment includes all required Python packages (LangChain, llama-cpp-python,
Qdrant, Panel, PyMuPDF, sentence-transformers, and others). 

1. Fetch the PDF catalouge of the MLIP literauture selected within `download_mlip_papers.py`:

```
python download_mlip_papers.py
```

2. Run the present configurations set in `mlip_advisor_example_chatbot.py`:

```
python mlip_advisor_example_chatbot.py
```

3. Follow the link to the chatbot server, ask which MLIP to use for any chemical system and property type. See what the model suggests and verify its suggestions.

## Making a new RAG-LLM chat bot application

1. Either modify the `download_mlip_papers.py` to download new set of papers of your choice, or move a collection of pdfs to a new folder catalogue. 

2. Modify `mlip_advisor_example_chatbot.py` for a new `PDF_FOLDER` location and modify the RAG parameters and prompt template for your desired application. 
