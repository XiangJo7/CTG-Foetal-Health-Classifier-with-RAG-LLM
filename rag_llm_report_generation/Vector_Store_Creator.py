import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def process_pdf_folder_langchain(folder_path, index_save_path="faiss_index"):
    all_docs = []

    print("🔄 Loading and chunking PDFs...")
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            try:
                loader = PyMuPDFLoader(os.path.join(folder_path, file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file  # Add filename as metadata
                all_docs.extend(docs)
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    print(f"✅ Loaded {len(all_docs)} pages. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    print(f"📄 Total chunks created: {len(chunks)}")

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Persist the index
    vectorstore.save_local(index_save_path)
    print(f"✅ Vector store saved to '{index_save_path}'")

    return vectorstore

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    pdf_papers_path = project_root / "Academic Paper Storage" / "Academic Papers"
    vector_space_path = project_root / "Academic Paper Storage"
    ctg_vectorstore = process_pdf_folder_langchain(pdf_papers_path, str(vector_space_path))