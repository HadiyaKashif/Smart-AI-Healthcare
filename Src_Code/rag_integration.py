# src_codes/rag_integration.py

import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import re

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
        "https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings",
        "https://www.cdc.gov/heart-disease/about/index.html",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke",
        "https://www.cdc.gov/obesity/risk-factors/risk-factors.html",
        "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "https://www.mayoclinic.org/tests-procedures/blood-pressure-test/about/pac-20393098"
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
        Your Personal Health Report
        Hello {patient_data.get('Name')}, this report is designed to help you understand your recent vital signs and what they might mean for your health. Our goal is to give you clear information and practical next steps.

        1. Explanation of Your Results & Risk Level
        Based on the information we have, your results indicate a Moderate to High Risk that warrants attention.

        In simple terms, your body is showing signs of working harder than it should to pump blood throughout your body. We see this primarily in your elevated blood pressure. When this is consistently high, it can put extra strain on your heart and blood vessels over time. We have categorized your risk as moderate to high because addressing this now is important for protecting your long-term health.

        2. What This Could Mean (Possible Diagnosis)
        It's important to remember that this is not a formal diagnosis, but a assessment based on your current numbers. The pattern of your vitals is most commonly associated with Primary Hypertension (High Blood Pressure).

        This is a very common condition where the long-term force of blood against your artery walls is high enough that it may eventually cause health problems. The good news is that it is often manageable with lifestyle adjustments and, if needed, medication.

        3. Your Suggested Next Steps
        Your health is a partnership, and there are clear actions we can take together. Here is what we recommend:

        Schedule a Follow-Up Appointment: Please book an appointment with your primary care provider to discuss these findings in detail. This is the most important next step. They will likely want to check your blood pressure again to confirm the reading.

        Monitor at Home: If your provider agrees, you might consider monitoring your blood pressure at home. We can advise you on how to choose a reliable monitor and how to take accurate readings.

        Lifestyle Considerations: There are powerful steps you can take to support your heart health:

        Diet: Reducing sodium (salt) intake can have a significant positive impact.

        Activity: Incorporating gentle, regular exercise like brisk walking can help strengthen your heart.

        Stress: Exploring stress-reduction techniques such as deep breathing or meditation can be beneficial.

        We are here to support you. Please don't hesitate to reach out if you have any questions or need help scheduling your next appointment. Taking proactive steps now is a powerful way to invest in your future well-being.
        """

        result = qa_chain(query)
        answer = result.get("result", "").strip()
        print(f"✅ RAG response obtained. {answer}" )
         # --- Pattern-based extraction ---
        sections = {
            "explanation": "",
            "diagnosis": "",
            "nextSteps": "",
        }

        patterns = {
            "explanation": r"Explanation of Your Results.*?(?=\n\s*\*\*?What|2\. What|What This Could|$)",
            "diagnosis": r"What This Could Mean.*?(?=\n\s*\*\*?Your Suggested|3\. Your|Your Suggested|$)",
            "nextSteps": r"Your Suggested Next Steps.*?(?=\n\s*\*\*?Additional|4\. Additional|Additional Concerns|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, answer, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(0)
                # remove the heading itself
                text = re.sub(r"^.*?:?\s*", "", text.split("\n", 1)[-1]).strip()
                sections[key] = text

        # --- Convert next steps and concerns into list form ---
        # --- Convert next steps and concerns into list form ---
        def extract_bullets(text):
            items = re.findall(r"(?:\*|\+|-)\s*(.+)", text)
            # Filter out the "being." item and other unwanted short items
            filtered_items = []
            for item in items:
                clean_item = item.strip()
                # Skip items that are just "being." or other very short non-meaningful text
                if clean_item and clean_item not in ["being.", "being"] and len(clean_item) > 3:
                    filtered_items.append(clean_item)
            return filtered_items if filtered_items else [text] if text else []
        next_steps = extract_bullets(sections["nextSteps"])
        # --- Detect risk level from explanation ---
        risk_match = re.search(r"(High|Moderate|Low)\s*Risk", sections["explanation"], re.IGNORECASE)
        risk_label = risk_match.group(0).title() if risk_match else risk_level
    
        structured_response = {
            "risk": risk_label,
            "explanation": sections["explanation"] or "No explanation found.",
            "diagnosis": sections["diagnosis"] or "No diagnosis found.",
            "nextSteps": next_steps or ["No steps provided."],
        }

        print("✅ Parsed structured response:", structured_response)
        return structured_response  
    except Exception as e:
        print("⚠️ RAG query failed:", e)
        return {
            "risk": risk_level,
            "explanation": "RAG analysis failed.",
            "diagnosis": "Unable to retrieve diagnosis.",
            "nextSteps": ["Consult a doctor for further advice."],
        }
