# vector_builder.py
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema.document import Document

class RemoteEmbeddingFunction:
    def __init__(self, embed_func):
        self.embed_func = embed_func

    def embed_documents(self, texts):
        return self.embed_func(texts)

    def embed_query(self, text):
        return self.embed_func([text])[0]


def build_vectorstore_from_pdf(file, get_embeddings):
    loader = UnstructuredPDFLoader(file_path=file.name)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    embedding = RemoteEmbeddingFunction(get_embeddings)

    vectorstore = Chroma.from_documents(split_docs, embedding=embedding, collection_name="demo")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    return retriever
