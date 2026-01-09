import os
import gradio as gr
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Configuration & Environment Setup
# Ensure GROQ_API_KEY is set in your system environment variables
api_key = os.environ.get("GROQ_AF")

CANCER_KEYWORDS = [
    "cancer", "tumor", "tumour", "oncolgy",
    "chemotherapy", "radiation", "radiotherapy",
    "metastasis", "carcinoma", "luekemia",
    "lymphoma", "melanoma"
]

DISCLAIMER = (
    "Disclaimer:\n"
    "This application is provided for educational purposes only.\n"
    "It does not provide medical diagnoses or treatment advice.\n"
    "Please consult a qualified healthcare professional."
)

# 2. Document Processing Functions
def load_and_process_documents(folder_path="documents"):
    """Loads documents from a local folder and creates a retriever."""
    all_docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder '{folder_path}'. Please add your PDFs/docs there.")
        return None

    for f in os.listdir(folder_path):
        full_path = os.path.join(folder_path, f)
        if f.endswith(".txt"):
            loader = TextLoader(full_path)
        elif f.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif f.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        else:
            continue
        all_docs.extend(loader.load())

    if not all_docs:
        return None

    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)

    # Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Initialize RAG Components
retriever = load_and_process_documents()

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=api_key
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# We wrap the chain creation in a function to handle cases where docs aren't loaded yet
def get_rag_chain():
    if retriever is None:
        return None
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

rag_chain = get_rag_chain()

# 4. Logic Functions
def is_cancer_related(question: str) -> bool:
    question = question.lower()
    return any(keyword in question for keyword in CANCER_KEYWORDS)

def chat_fn(message, history):
    if not api_key:
        return "Error: GROQ_API_KEY not found in environment variables."
    
    if not is_cancer_related(message):
        return "This system only answers cancer-related questions."

    if rag_chain is None:
        return "Error: No documents found. Please add PDF files to the 'documents' folder."

    res = rag_chain.invoke({"question": message})
    return res["answer"]

# 5. Launch App
if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="RAG Cancer Treatment Assistant",
        description=DISCLAIMER
    )
    demo.launch()
