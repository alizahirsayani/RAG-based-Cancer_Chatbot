import os
import glob
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Setup & Environment
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

CANCER_KEYWORDS = [
    "cancer", "tumor", "tumour", "oncology", "chemotherapy", 
    "radiation", "radiotherapy", "metastasis", "carcinoma", 
    "leukemia", "lymphoma", "melanoma"
]

DISCLAIMER = (
    "Disclaimer:\n"
    "This application is provided for educational purposes only.\n"
    "It does not provide medical diagnoses or treatment advice.\n"
    "Please consult a qualified healthcare professional."
)

# 2. Document Processing
def initialize_rag():
    all_docs = []
    # Automatically find all documents in the data/ folder
    files = glob.glob("data/*")
    
    for f in files:
        if f.endswith(".txt"):
            loader = TextLoader(f)
        elif f.endswith(".pdf"):
            loader = PyPDFLoader(f)
        elif f.endswith(".docx"):
            loader = Docx2txtLoader(f)
        else:
            continue
        all_docs.extend(loader.load())

    # Text Splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)

    # Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = initialize_rag()

# 3. RAG Chain Setup
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True, 
    output_key="answer"
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# 4. Interface Logic
def is_cancer_related(question):
    question = question.lower()
    return any(keyword in question for keyword in CANCER_KEYWORDS)

def chat_fn(message, history):
    if not is_cancer_related(message):
        return "This system only answers cancer-related questions."

    res = rag_chain.invoke({"question": message})
    return res["answer"]

# 5. Launch
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="RAG Cancer Treatment Assistant",
        description=DISCLAIMER
    )
    demo.launch()
