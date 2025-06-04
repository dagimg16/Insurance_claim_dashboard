import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

os.environ.pop("SSL_CERT_FILE", None)

load_dotenv() 
openai_key= os.getenv("openai_key")
PDF_FOLDER = "adjuster_manual"
VECTORSTORE_FOLDER = "adjuster_manual_vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_FOLDER, "index.faiss")

def load_text_from_pdfs(pdf_folder):
    all_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
    return all_text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_or_load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    if os.path.exists(INDEX_FILE):
        print("Loading existing vector database...")
        vector_store = FAISS.load_local(VECTORSTORE_FOLDER, embeddings, allow_dangerous_deserialization=True)
    else:
        print("No vector database found. Creating new one...")
        # Load and split text
        manual_text = load_text_from_pdfs(PDF_FOLDER)
        chunks = split_text_into_chunks(manual_text)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

        # Create and save vector store
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local(VECTORSTORE_FOLDER)

    return vector_store


vector_store = create_or_load_vectorstore()


# Initialize the LLM (the brain!)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",   # or "gpt-4" if you want
    openai_api_key= openai_key
)

# Set up the Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  
)

if __name__ == "__main__":
    while True:
        query = input("Ask something: ")
        print(qa_chain.run(query))