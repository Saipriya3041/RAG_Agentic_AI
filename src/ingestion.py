import time
from pathlib import Path
from urllib.request import Request, urlopen

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"


def ensure_pdf(pdf_path: str) -> Path:
    path = Path(pdf_path)
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(PDF_URL, headers={"User-Agent": "AgenticAIRAG/1.0"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Downloaded file is not a PDF: {PDF_URL}")
    path.write_bytes(content)
    return path


def _get_compatible_index(index_name: str, dimension: int):
    client = Pinecone()
    index_names = client.list_indexes().names()
    if index_name not in index_names:
        raise ValueError(f"Pinecone index '{index_name}' does not exist.")

    base_description = client.describe_index(index_name)
    if base_description.dimension == dimension:
        return index_name, client.Index(index_name)

    compatible_name = f"{index_name}-{dimension}-v2"
    if compatible_name not in index_names:
        serverless_spec = base_description.spec.get("serverless")
        if serverless_spec is None:
            raise ValueError(
                f"Index '{index_name}' has dimension {base_description.dimension}, "
                f"but {EMBEDDING_MODEL} outputs {dimension}. Create a {dimension}-dimension index."
            )
        client.create_index(
            name=compatible_name,
            dimension=dimension,
            metric=base_description.metric,
            spec=ServerlessSpec(
                cloud=serverless_spec.cloud,
                region=serverless_spec.region,
            ),
        )

    deadline = time.monotonic() + 120
    while True:
        description = client.describe_index(compatible_name)
        if description.status["ready"]:
            if description.dimension != dimension:
                raise ValueError(
                    f"Pinecone index '{compatible_name}' has dimension "
                    f"{description.dimension}; expected {dimension}."
                )
            return compatible_name, client.Index(compatible_name)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Pinecone index '{compatible_name}' did not become ready.")
        time.sleep(2)


def run_ingestion(pdf_path: str, index_name: str):
    pdf_path = str(ensure_pdf(pdf_path))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    dimension = len(embeddings.embed_query("dimension check"))
    index_name, index = _get_compatible_index(index_name, dimension)
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    if index.describe_index_stats().get("total_vector_count", 0) > 0:
        return vector_store

    docs = PyPDFLoader(pdf_path).load()
    docs = [doc for doc in docs if doc.page_content.strip()]
    for doc in docs:
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["source"] = Path(pdf_path).name

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")

    vector_store.add_documents(chunks)
    expected_count = len(chunks)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if index.describe_index_stats().get("total_vector_count", 0) >= expected_count:
            break
        time.sleep(2)
    else:
        raise TimeoutError("PDF vectors were not all visible in Pinecone after ingestion.")

    return vector_store
