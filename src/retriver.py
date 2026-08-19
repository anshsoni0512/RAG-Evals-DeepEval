import os
import re
import glob

from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads PINECONE_API_KEY from .env

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INDEX_NAME = "deepevals"
EMBEDDING_MODEL = "llama-text-embed-v2"  # must match the model attached to the Pinecone index


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it in Pinecone so we don't re-embed
def load_store():
    embeddings = PineconeEmbeddings(model=EMBEDDING_MODEL)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    stats = pc.Index(INDEX_NAME).describe_index_stats()

    if stats.total_vector_count > 0:
        return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
    ).split_documents(docs)

    return PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriver.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")

    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")
