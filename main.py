import os
import argparse
import sys
import shutil

# Modern Document Loaders & Splitters
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Modern Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Modern LLM and LCEL Imports
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_documents(folder_path):
    """
    Loads PDF and Text files from the specified folder.
    Returns a list of LangChain Document objects.
    """
    print(f"Loading documents from {folder_path}...")
    documents = []

    # 1. Load PDFs
    pdf_loader = PyPDFDirectoryLoader(folder_path)
    try:
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} PDF pages/documents.")
    except Exception as e:
        print(f"Warning: Failed to load PDFs. Error: {e}")

    # 2. Load Text Files
    txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    try:
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} text files.")
    except Exception as e:
        print(f"Warning: Failed to load text files. Error: {e}")

    return documents


def format_docs(docs):
    """Formats retrieved document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    # 1. Parse Arguments with default fallback to ./docs
    parser = argparse.ArgumentParser(description="Local RAG Document Search CLI")
    parser.add_argument(
        "folder_path",
        type=str,
        nargs="?",        # Makes folder_path optional
        default="./docs", # Defaults to ./docs folder automatically
        help="Path to the local folder containing PDF and TXT files"
    )
    args = parser.parse_args()
    folder_path = args.folder_path

    # 2. Validate directory
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(folder_path):
        print(f"Error: The path '{folder_path}' is not a directory.")
        sys.exit(1)

    # 3. Load Documents
    documents = load_documents(folder_path)

    if not documents:
        print("Error: No supported documents (PDF or TXT) found in the provided folder.")
        sys.exit(1)

    # 4. Split text into smaller, more granular chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 5. Initialize Embeddings
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 6. Initialize Vector Database (Chroma)
    # Clear stale vector cache to force a clean indexing of all files
    db_path = "./chroma_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print("Creating local vector database (Chroma)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )

    # 7. Retriever - Increase 'k' to 4 to fetch more context across files
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 8. Ollama LLM Connection
    print("Connecting to local LLM via Ollama...")
    llm = OllamaLLM(model="llama3")

    # 9. Prompt Template
    system_prompt = (
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer based on the context, say that you don't know.\n"
        "Do not make up an answer. Keep the answer concise and relevant.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 10. Modern LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- Setup Complete ---")
    print("Type 'exit' or 'quit' to stop the application.")

    # 11. Interactive CLI Loop
    while True:
        try:
            user_input = input("\nAsk a question: ")

            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting RAG CLI. Goodbye!")
                break

            if not user_input.strip():
                continue

            print("Thinking...")

            # Run query through retriever to get source documents
            source_docs = retriever.invoke(user_input)

            # Generate answer using the LCEL chain
            answer = rag_chain.invoke(user_input)

            print(f"\nAnswer:\n{answer}")

            print("\nSources:")
            # Use set to avoid printing duplicate source files if multiple chunks came from the same file
            seen_sources = set()
            for doc in source_docs:
                source_file = doc.metadata.get("source", "Unknown file")
                if source_file not in seen_sources:
                    seen_sources.add(source_file)
                    print(f"  - {source_file}")

        except KeyboardInterrupt:
            print("\nExiting RAG CLI. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during retrieval/generation: {e}")


if __name__ == "__main__":
    main()