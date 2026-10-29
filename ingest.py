from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path("base_conhecimento")
DB_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def carregar_documentos():
    docs = []
    for caminho in DOCS_DIR.rglob("*"):
        ext = caminho.suffix.lower()
        if ext == ".pdf":
            docs += PyPDFLoader(str(caminho)).load()      # 1 Document por página
        elif ext in {".md", ".txt"}:
            docs += TextLoader(str(caminho), encoding="utf-8").load()
    return docs


def main():
    docs = carregar_documentos()

    # Divide em chunks com sobreposição, para não cortar ideias no meio
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Cria e persiste o índice em disco
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print(f"{len(docs)} documentos → {len(chunks)} chunks indexados.")


if __name__ == "__main__":
    main()