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
    'chest pain': [
        "1. Apply heat to the affected area for comfort.",
        "2. Try deep breathing exercises to help with anxiety.",
        "3. Over-the-counter pain relievers can provide temporary relief.",
        "4. Consult your oncologist to rule out any serious issues."
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
    elif 'chest pain' in lower_input:
        treatment = "\n".join(treatment_suggestions['chest pain'])
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
    input_lower = user_input.lower()
    
    if "tired" in input_lower:
        follow_up_question = "What do you think could help you feel more energized?"
        response = "It's normal to feel tired, especially during treatment. Perhaps taking short walks or trying light stretching could help boost your energy levels. Remember to take it easy and listen to your body."
    
    elif "hopeless" in input_lower:
        follow_up_question = "Is there someone you trust that you can reach out to for support?"
        response = "It's tough to feel hopeless, but reaching out to a friend or family member can make a difference. You're not alone in this; support is always available when you need it."
    
    elif "anxious" in input_lower:
        follow_up_question = "Are there specific things that make you feel anxious right now?"
        response = "Anxiety is common during treatment. Identifying what triggers your anxiety can help manage it. Consider breathing exercises or mindfulness techniques; they can provide some relief."
    
    elif "sad" in input_lower:
        follow_up_question = "Have you found any activities that help lift your mood?"
        response = "It's okay to feel sad sometimes. Engaging in hobbies or activities that you enjoy can help bring some joy back into your day. Remember, it's perfectly fine to take time for yourself."
    
    elif "overwhelmed" in input_lower:
        follow_up_question = "What are the main sources of your overwhelm at the moment?"
        response = "Feeling overwhelmed can be a lot to handle. Taking things one day at a time and focusing on small tasks can make it easier. You’re doing the best you can, and that’s what matters."
    
    elif "lonely" in input_lower:
        follow_up_question = "Are there support groups or friends you could connect with?"
        response = "Loneliness can be difficult, but connecting with others who understand what you're going through can help. There are many support groups available; consider reaching out to one."
    
    elif "frustrated" in input_lower:
        follow_up_question = "What aspects of your treatment are causing the most frustration?"
        response = "It's understandable to feel frustrated at times. Sharing your feelings with your healthcare team can help; they might provide alternatives or adjustments that could improve your experience."
    
    elif "uncertain" in input_lower:
        response = "It's normal to feel uncertain about the future, especially during treatment. Gathering information and discussing your concerns can help you feel more in control. I'm here to help you find those answers."
    
    else:
        return None  # No specific emotional state detected

    return f"{follow_up_question} {response}"

# Chainlit code to handle the conversation flow
@cl.on_chat_start
async def start():
    chain = qa_bot()  # Ensure qa_bot is defined before calling
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I am your personal Medical Bot. I am here to assist you. How was your day?"
    await msg.update()

    cl.user_session.set("chain", chain)

# Adding a check for casual conversation and greetings
def is_medical_query(user_input):
    """
    Detects if the input is related to a medical condition or is general conversation.
    """
    medical_keywords = [
        'fever', 'cough', 'cold', 'pain', 'nausea', 'headache', 'sore throat', 
        'chills', 'cancer', 'treatment', 'fatigue', 'symptoms', 'medicine', 
        'diagnosis', 'doctor', 'hospital', 'surgery', 'disease', 'infection'
    ]
    
    # Check if any medical-related keywords are in the user input
    for keyword in medical_keywords:
        if keyword in user_input.lower():
            return True
    
    return False  # Assume it's general conversation if no keywords match

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
    
    # Check if the input is medical or general conversation
    if is_medical_query(message.content):
        # Medical-related query: Use BioBERT
        bio_answer = bioBERT_answer(message.content)
        if bio_answer:
            await cl.Message(content=bio_answer).send()
            return
    else:
        # General conversation: Respond with a casual message
        if message.content.lower() in ['hi', 'hello', 'hey']:
            await cl.Message(content="Hi! How can I assist you today?").send()
            return
        
    # If the input is neither medical nor a greeting, use LLaMA for general QA
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
    final_answer = f"{answer}"
    await cl.Message(content=final_answer).send()

    # Store chainlit user session for later conversation continuity
    cl.user_session.set("chain", chain)
