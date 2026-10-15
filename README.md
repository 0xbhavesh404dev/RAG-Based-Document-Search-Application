# RAG-Based Document Search Application

A local, command-line Retrieval-Augmented Generation (RAG) application for asking questions about your own PDF and text documents.

The application loads documents from a folder, splits them into smaller passages, converts those passages into vector embeddings, retrieves the most relevant passages for each question, and uses a locally running Ollama model to generate an answer grounded in the retrieved context.

> **Privacy note:** Documents are processed locally. The application uses Hugging Face embeddings, a local Chroma vector database, and Ollama for local language-model inference.

## Features

- Supports PDF and plain-text (`.txt`) documents.
- Uses recursive text splitting to create searchable chunks.
- Generates embeddings with `all-MiniLM-L6-v2`.
- Stores embeddings in a local Chroma database.
- Retrieves the four most relevant chunks for each question.
- Generates answers with the local Ollama `llama3` model.
- Displays the source files used for each answer.
- Provides an interactive command-line interface.

## How the application works

The application follows a standard RAG pipeline:

```text
Documents
   │
   ▼
Load PDF and TXT files
   │
   ▼
Split documents into chunks
   │
   ▼
Create vector embeddings
   │
   ▼
Store embeddings in Chroma
   │
   ▼
Retrieve relevant chunks for a question
   │
   ▼
Send retrieved context to Ollama
   │
   ▼
Display the grounded answer and source files
```

### 1. Document loading

By default, the application reads files from the `./docs` directory. It loads:

- PDF files using `PyPDFDirectoryLoader`.
- Text files matching `**/*.txt` using `DirectoryLoader` and `TextLoader`.

The folder can also be supplied as a command-line argument.

### 2. Text chunking

Documents are divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`:

- **Chunk size:** 500 characters
- **Chunk overlap:** 50 characters

The overlap helps preserve context when a relevant passage crosses a chunk boundary.

### 3. Embedding generation

Each chunk is converted into a numerical vector using the Hugging Face model:

```text
all-MiniLM-L6-v2
```

Chunks with similar meanings produce similar vectors, allowing the application to search by semantic meaning rather than exact keyword matches.

### 4. Vector storage and retrieval

The vectors are stored in the local `./chroma_db` directory using Chroma. For every question, the application retrieves the four most relevant chunks.

The database is rebuilt each time the application starts. This ensures that changes to the documents are reflected, but it also means that startup may take longer for large document collections.

### 5. Answer generation

The retrieved chunks are inserted into a prompt and sent to the local Ollama model configured in `main.py`:

```text
llama3
```

The prompt instructs the model to answer using the retrieved context, avoid inventing information, and say that it does not know when the answer is not supported by the documents.

### 6. Source reporting

After generating an answer, the application prints the source file paths for the retrieved documents. Duplicate source paths are removed from the displayed list.

## Project structure

```text
RAG-Based-Document-Search-Application/
├── main.py                 # Main RAG application and interactive CLI
├── requirements.txt        # Python dependencies
├── docs/                   # Input PDF and TXT documents
│   ├── rag_test_corpus.txt
│   └── rag_test_document.pdf
├── chroma_db/              # Generated local vector database
└── README.md               # Project documentation
```

## Requirements

Before starting, install the following software:

- Python 3.9 or newer
- [Ollama](https://ollama.com/)
- The Ollama `llama3` model
- A system capable of running the embedding model locally

The first time the embedding model is used, Hugging Face may download model files to the local cache.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yusufshereef/RAG-Based-Document-Search-Application.git
cd RAG-Based-Document-Search-Application
```

### 2. Create and activate a virtual environment

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The current application imports the following integration packages in addition to the packages listed in `requirements.txt`. If your installation reports missing-module errors, install them with:

```bash
pip install langchain-text-splitters langchain-chroma langchain-ollama
```

### 4. Install and prepare Ollama

Install Ollama for your operating system, then download the model used by the application:

```bash
ollama pull llama3
```

Start Ollama if it is not already running. On many systems, Ollama runs as a background service after installation. You can also verify that the model is available with:

```bash
ollama list
```

## Add your data

Place the documents you want to search in the `docs/` directory.

Supported formats:

- `.pdf`
- `.txt`

For example:

```text
docs/
├── company-handbook.pdf
├── product-notes.txt
└── research-paper.pdf
```

You can use the sample files already included in this repository to test the application. The program exits with an error if the selected directory does not exist or contains no supported PDF or TXT files.

### Data recommendations

For better retrieval quality:

- Use text-based PDFs when possible. Scanned image-only PDFs may require OCR before they can be searched.
- Keep unrelated document collections in separate folders.
- Use clear, complete documents with consistent text encoding.
- Remove duplicate or obsolete documents before indexing.
- Avoid placing secrets or sensitive information in files unless you understand and accept the local storage and model-processing implications.

## Run the application

### Use the default `docs/` directory

```bash
python main.py
```

### Use a custom document directory

```bash
python main.py path/to/your/documents
```

For example:

```bash
python main.py ./my_documents
```

After startup, enter a question at the prompt:

```text
Ask a question: What is this document about?
```

The application prints:

1. The generated answer.
2. The source files used during retrieval.

To stop the application, type either:

```text
exit
```

or:

```text
quit
```

You can also press `Ctrl+C`.

## Example session

```text
$ python main.py
Loading documents from ./docs...
Loaded 1 PDF pages/documents.
Loaded 1 text files.
Splitting documents into chunks...
Created 12 text chunks.
Initializing embedding model...
Creating local vector database (Chroma)...
Connecting to local LLM via Ollama...

--- Setup Complete ---
Type 'exit' or 'quit' to stop the application.

Ask a question: What topics are covered in the documents?
Thinking...

Answer:
The documents cover ...

Sources:
  - ./docs/rag_test_document.pdf
  - ./docs/rag_test_corpus.txt
```

The exact number of pages, chunks, answer text, and source files depends on the documents in the selected folder.

## Configuration

The main configuration values are currently defined directly in `main.py`:

| Setting | Current value | Purpose |
| --- | --- | --- |
| Default input folder | `./docs` | Used when no folder argument is supplied |
| Chunk size | `500` | Maximum size of each text chunk |
| Chunk overlap | `50` | Repeated context between adjacent chunks |
| Embedding model | `all-MiniLM-L6-v2` | Converts text chunks into vectors |
| Chroma directory | `./chroma_db` | Stores the local vector database |
| Retrieved chunks | `4` | Number of relevant chunks per question |
| Ollama model | `llama3` | Generates the final answer |

To use a different Ollama model, update the `model` value in `main.py` and pull that model with Ollama first.

## Troubleshooting

### `No supported documents ... found`

Make sure the input directory contains at least one `.pdf` or `.txt` file and that the path is correct.

### `Path does not exist` or `Path is not a directory`

Check the folder path passed to the command. Relative paths are resolved from the directory where you run `python main.py`.

### `Could not connect to Ollama`

Make sure Ollama is installed and running, and verify that the model is available:

```bash
ollama list
ollama pull llama3
```

### `ModuleNotFoundError`

Activate the virtual environment and install the dependencies again:

```bash
pip install -r requirements.txt
pip install langchain-text-splitters langchain-chroma langchain-ollama
```

### Poor or incomplete answers

Try asking more specific questions, adding higher-quality source documents, or adjusting the chunk size, chunk overlap, and number of retrieved chunks in `main.py`.

### Scanned PDFs are not searchable

The PDF loader extracts text; it does not perform OCR. Convert scanned PDFs to searchable PDFs or extract their text before adding them to `docs/`.

## Limitations

- The application currently provides an interactive CLI rather than a web interface.
- The Chroma database is deleted and rebuilt on every run.
- Only PDF and TXT files are loaded.
- Answer quality depends on the quality of the documents, retrieved context, embedding model, and Ollama model.
- The application does not provide authentication, multi-user access, or document-level permissions.

## License

No license file is currently included in this repository. Add a license if you plan to distribute or reuse the project under specific terms.
