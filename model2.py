from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import chainlit as cl  # type: ignore
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load BioBERT
def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModelForMaskedLM.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    return pipeline('fill-mask', model=model, tokenizer=tokenizer)

biobert_pipeline = load_biobert()  # Load BioBERT for use

# Define treatment suggestions based on symptoms
treatment_suggestions = {
    'cold': [
        "1. Stay hydrated with plenty of fluids.",
        "2. Rest as much as possible.",
        "3. Over-the-counter medications like decongestants or antihistamines can help relieve symptoms.",
        "4. Gargle with salt water for a sore throat."
    ],
    'fever': [
        "1. Stay hydrated with fluids.",
        "2. Take fever-reducing medications like acetaminophen or ibuprofen.",
        "3. Rest as much as possible.",
        "4. Avoid overheating by wearing light clothing."
    ],
    'cough': [
        "1. Drink warm liquids like tea or broth.",
        "2. Use a humidifier to keep the air moist.",
        "3. Over-the-counter cough medicines can help alleviate symptoms.",
        "4. If the cough persists for more than a week, consult a doctor."
    ],
    'nausea': [
        "1. Ginger tea or ginger ale may help.",
        "2. Eat small, bland meals to avoid an upset stomach.",
        "3. Stay hydrated by sipping water.",
        "4. If nausea persists, consult a healthcare provider."
    ],
    # Add more conditions as needed
}

# Define custom prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 1}),  # Fetch only one document
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Loading the LLaMA model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function (using LLaMA for retrieval-based QA)
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Medical-based question answering using BioBERT
def bioBERT_answer(user_input):
    # Identify the condition based on the user's input
    lower_input = user_input.lower()
    
    # Check for specific conditions and generate appropriate response
    if 'cold' in lower_input:
        treatment = "\n".join(treatment_suggestions['cold'])
    elif 'fever' in lower_input:
        treatment = "\n".join(treatment_suggestions['fever'])
    elif 'cough' in lower_input:
        treatment = "\n".join(treatment_suggestions['cough'])
    elif 'nausea' in lower_input:
        treatment = "\n".join(treatment_suggestions['nausea'])
    else:
        # Use BioBERT for terms not explicitly defined
        medical_query = f"Patient is asking: {user_input}. The potential medical term is [MASK]."
        predictions = biobert_pipeline(medical_query)
        top_prediction = predictions[0]['sequence']
        return f"BioBERT Answer: {top_prediction}. Please consult a healthcare provider for personalized advice."

    return f"BioBERT Answer: Based on your symptoms, here are some suggested treatments:\n{treatment}"

# Symptom Checker using BioBERT
def symptom_checker(user_input, user_history):
    common_symptoms = [
        # General symptoms
        'fever', 'cold', 'cough', 'sore throat', 'headache', 'runny nose', 'chills', 
        'body aches', 'fatigue', 'nausea', 'vomiting', 'diarrhea', 'dizziness', 
        'shortness of breath', 'muscle pain', 'joint pain', 'loss of appetite', 
        'weight loss', 'swelling', 'rash', 'itching', 'constipation',

        # Cancer-related symptoms
        'unexplained weight loss', 'persistent cough', 'difficulty swallowing', 
        'unusual lumps', 'chronic pain', 'night sweats', 'bleeding', 
        'fatigue that doesn’t go away', 'persistent fever', 'skin changes', 
        'persistent sores', 'changes in bowel or bladder habits', 
        'unexplained bleeding or bruising', 'bone pain', 'abdominal pain', 
        'hoarseness', 'blood in urine', 'blood in stool', 'swollen lymph nodes', 
        'persistent bloating', 'unusual bleeding', 'jaundice', 'persistent back pain', 
        'new mole or changes in existing mole', 'difficulty breathing'
    ]

    identified_symptoms = [symptom for symptom in common_symptoms if symptom in user_input.lower()]

    # Enhanced with BioBERT to detect specific terms
    bio_bert_predictions = biobert_pipeline(f"Patient is experiencing {user_input}. The symptom might include [MASK].")
    bio_bert_keywords = [prediction['token_str'] for prediction in bio_bert_predictions[:5]]  # Top 5 predictions
    identified_symptoms.extend(bio_bert_keywords)

    if 'pain' in identified_symptoms:
        return "Can you tell me more about the pain? Where is it located? Is it sharp or dull?"
    
    if not identified_symptoms:
        return "I noticed you're not mentioning any specific symptoms. Can you describe where you feel discomfort or pain?"

    return None  # No follow-up is needed

# Comforting response for emotional support
def comforting_response(user_input):
    if "tired" in user_input.lower():
        return "I'm really sorry you're feeling this way. It must be tough, but remember, you're strong and there are ways we can manage the fatigue."
    
    if "hopeless" in user_input.lower():
        return "It's understandable to feel overwhelmed, but I'm here to help. Let's take it step by step. Have you been able to rest or talk to someone about how you're feeling?"
    
    return None

# Chainlit code to handle the conversation flow
@cl.on_chat_start
async def start():
    chain = qa_bot()  # Ensure qa_bot is defined before calling
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I am your personal Medical Bot. I am here to assist you. How was your day?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):    
    chain = cl.user_session.get("chain")
    user_history = cl.user_session.get("user_history", [])
    
    # Append new message to history
    user_history.append(message.content)
    cl.user_session.set("user_history", user_history)
    
    # Check if the user needs a comforting response
    comforting_msg = comforting_response(message.content)
    if comforting_msg:
        await cl.Message(content=comforting_msg).send()
        return
    
    # Call the symptom checker first
    follow_up_question = symptom_checker(message.content, user_history)  # corrected variable name
    
    # If a follow-up question is triggered, ask that instead of querying the LLM
    if follow_up_question:
        await cl.Message(content=follow_up_question).send()
        return
    
    # BioBERT answers medical-specific questions
    bio_answer = bioBERT_answer(message.content)
    if bio_answer:
        await cl.Message(content=bio_answer).send()
        return

    # For other non-medical questions or general QA, use LLaMA
    res = await chain.acall(message.content)
    
    # Get the final answer from the result
    answer = res.get("result", "")
    
    # Ensure only one instance of sources (if any) is added
    sources = res.get("source_documents", [])
    if sources:
        top_source = sources[0]  # Fetch the first source document
        source_info = f"\nSource: {top_source.metadata.get('source', 'Unknown')}, Page {top_source.metadata.get('page', 'Unknown')}"
    else:
        source_info = "\nNo sources found"
    
    # Send only one message with the final result
    final_answer = f"{answer} {source_info}"
    await cl.Message(content=final_answer).send()

    # Store chainlit user session for later conversation continuity
    cl.user_session.set("chain", chain)
