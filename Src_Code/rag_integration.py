# src_codes/rag_integration.py

import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

# ================== Initialize RAG once ==================
def init_rag():
    """Build or load vector DB from WHO/CDC health pages"""
    persist_dir = "rag_db"
    if os.path.exists(persist_dir):
        print("✅ Loading existing Chroma DB...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        return vectordb.as_retriever(search_kwargs={"k": 5})

    print("🌐 Building RAG knowledge base...")
    web_pages = [
        "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "https://www.who.int/news-room/fact-sheets/detail/obesity",
        "https://www.cdc.gov/cholesterol/facts.html",
        "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/index.htm",
    ]

    web_docs = []
    for url in web_pages:
        loader = WebBaseLoader(url)
        web_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(web_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("✅ Vector DB created successfully")

    return vectordb.as_retriever(search_kwargs={"k": 5})


# Initialize retriever globally (only once)
retriever = init_rag()
groq = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.3
)

qa_chain = RetrievalQA.from_chain_type(
    llm=groq,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)


# ================== Main RAG Query Function ==================
def query_rag(patient_data: dict, risk_level: str):
    """
    Takes structured patient data + predicted risk
    Returns causes and suggestions using the RAG knowledge base.
    """
    try:
        query = f"""
        Patient vitals:
        - Gender: {patient_data.get('Gender')}
        - Age: {patient_data.get('Age')}
        - BP: {patient_data.get('Systolic BP')}/{patient_data.get('Diastolic BP')}
        - Cholesterol: {patient_data.get('Cholesterol')}
        - BMI: {patient_data.get('BMI')}
        - Smoker: {patient_data.get('Smoker')}
        - Diabetic: {patient_data.get('Diabetes')}
        Predicted Risk: {risk_level}

        Provide:
        1. Explanation of possible causes.
        2. Recommended preventive actions or medical follow-up.
        3. Short actionable summary.
        """

        result = qa_chain(query)
        answer = result.get("result", "").strip()

        # Simple parsing (optional, can be made more robust)
        causes, suggestions = [], []
        if "cause" in answer.lower():
            causes.append(answer)
        if "suggest" in answer.lower() or "recommend" in answer.lower():
            suggestions.append(answer)

        return {
            "causes": causes or [answer],
            "suggestions": suggestions or ["Consult a healthcare provider for personalized advice."]
        }

    except Exception as e:
        print("⚠️ RAG query failed:", e)
        return {"causes": [], "suggestions": ["Unable to fetch RAG insights at the moment."]}
