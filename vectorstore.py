from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="./local_embedding_model") # all-MiniLM-L6-v2
    vectorstore=FAISS.from_documents(chunks,embeddings)
    return vectorstore