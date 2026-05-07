import streamlit as st
import os

from ingestor import load_pdf
from chunking import load_chunks
from vectorstore import create_vectorstore

st.title("Document Processor")

# 2. Create the file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@st.cache_resource
def process_pdf(file_path):
    with st.spinner("Processing PDF.."):
        docs=load_pdf(file_path)
    with st.spinner("Chunking documents..."):
        chunks=load_chunks(docs)
    with st.spinner("Creating Vectorstore.."):
        vectorstore=create_vectorstore(chunks)
    return vectorstore

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


if uploaded_file is not None:
    # 3. Construct the local file path
    if uploaded_file.name!=st.session_state.current_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.vectorstore=process_pdf(file_path)

        st.session_state.current_file=uploaded_file.name




from typing import TypedDict,List
from langchain_core.documents import Document
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model

google_api=os.getenv("google_api")

@st.cache_resource
def load_llm():
    return init_chat_model("google_genai:gemini-2.5-flash",api_key=google_api)

llm=load_llm()

class State(TypedDict):
    question:str
    answer:str
    document:List[Document]
    evidence:str

def retrive_documents(state:State):
    """Retrieve relevant docuements based on the question"""
    question=state['question']
    documents=[]
    vectorstore=st.session_state.vectorstore
    results=vectorstore.similarity_search_with_score(question,k=8)

    min_score=float('inf')
    for doc, score in results:
        if score<min_score:
            min_score=score
    threshold=min_score*1.2
    if results:
        for doc, score in results:
            if score<=threshold:
                documents.append(doc)
    return {**state,"document":documents}

def evidence_evaluater(state:State):
    """Based on the retireved documents decide wheather the documents are sufficient to answer the question or not"""
    question=state["question"]
    documents=state.get("document",[])

    context="\n\n".join([doc.page_content for doc in documents])
    prompt=f"""Based on the Question, evaluated wheater the context has satisfactory data to answer the question to the fullest
    Question: {question}, Context: {context}. Return only 'FULL', 'PARTIAL', 'INSUFFICIENT' """

    response=llm.invoke(prompt)
    decision = response.content.strip().upper()
    if "FULL" in decision:
        evidence = "FULL"
    elif "PARTIAL" in decision:
        evidence = "PARTIAL"
    else:
        evidence = "INSUFFICIENT"
    return {**state,"evidence":evidence}

def generate_answer(state:State):
    """Generate answer using retrieved docuemtents or direct answer"""
    question=state["question"]

    documents=state.get("document",[])

    evidence_state=state["evidence"]

    context="\n\n".join([doc.page_content for doc in documents])

    if evidence_state=="FULL":
        prompt = f"""
        You are a document-grounded AI assistant.

        Answer the question ONLY using
        the provided context.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
    elif evidence_state=="PARTIAL":
        prompt = f"""
        You are a document-grounded AI assistant.

        The retrieved context partially answers
        the user's question.

        Use the context as the PRIMARY source.

        You may supplement missing information
        using general knowledge, but clearly
        distinguish supplemental information
        from document-supported information.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
    
    else:
        prompt = f"""
        The retrieved documents do not contain
        sufficient information to fully answer
        the question.

        Explain this limitation clearly.

        Then provide only a brief high-level
        answer if possible.

        QUESTION:
        {question}

        ANSWER:
        """
        
    response=llm.invoke(prompt)
    return {**state,"answer":response.content}

builder=StateGraph(State)

builder.add_node("retrieve",retrive_documents)
builder.add_node("evaluator",evidence_evaluater)
builder.add_node("generator",generate_answer)

builder.add_edge(START,"retrieve")
builder.add_edge("retrieve","evaluator")
builder.add_edge("evaluator","generator")
builder.add_edge("generator",END)

graph=builder.compile()

def graph_response(question):
    respond={"question":question}
    result=graph.invoke(respond)
    return result


question=st.text_input("Enter Question?")

if question:

    if st.session_state.vectorstore is None:
        st.warning("Please upload PDF first.")
    else:
        with st.spinner("Generating Answer.."):
            result=graph_response(question)
        if result['evidence']=="PARTIAL":
            st.warning(f"The query has partial relevance to the uploaded document, generated supplemental information")
        elif result['evidence']=="INSUFFICIENT":
            st.info(f"The query is not relevant to the uploaded document, using pre-built reasoning")
        else:
            st.success("Answer generated from uploaded document")
        st.write(result["answer"])