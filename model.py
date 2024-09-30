# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.prompts import PromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

import chainlit as cl # type: ignore

DB_FAISS_PATH = 'vectorstore/db_faiss'

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
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=['context', 'question'])
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

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device': 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def symptom_checker(user_input, user_history):
    # General and cancer-related symptoms
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

    # Expanded pain details
    pain_details = [
        # Types of pain
        'sharp', 'dull', 'burning', 'throbbing', 'stabbing', 'aching', 'cramping', 
        'radiating', 'shooting', 'pressure', 'tenderness', 'constant', 'intermittent',

        # Locations (general)
        'head', 'chest', 'back', 'stomach', 'abdomen', 'leg', 'arm', 'shoulder', 'neck', 
        'hip', 'pelvis', 'knee', 'foot', 'hand', 'jaw', 'side', 'rib', 'lower back',

        # Locations (cancer-specific)
        'breast', 'lung', 'liver', 'colon', 'pancreas', 'prostate', 'ovary', 
        'brain', 'lymph nodes', 'skin', 'throat', 'esophagus', 'bladder', 'kidney', 
        'testicle', 'uterus', 'rectum', 'bone',

        # Duration/Timing
        'chronic', 'sudden', 'long-lasting', 'persistent', 'acute', 'recurring', 
        'continuous', 'comes and goes', 'worse at night', 'worse in the morning',

        # Cancer-related details
        'lump', 'mass', 'tumor', 'swelling', 'nodule', 'growth', 'lesion', 'hardness', 
        'bloating', 'enlarged lymph nodes', 'abnormal bleeding', 'persistent cough', 
        'unexplained pain', 'unexplained fatigue', 'difficulty swallowing',

        # Other descriptors
        'location', 'localized', 'spread', 'generalized', 'migrating', 'on one side', 
        'on both sides', 'near the spine', 'deep', 'surface-level', 'inside the body',
        
        # Symptoms associated with pain
        'nausea', 'vomiting', 'fatigue', 'weakness', 'fever', 'chills', 
        'shortness of breath', 'dizziness', 'fainting', 'numbness', 'tingling', 
        'loss of function', 'difficulty moving', 'sensitivity to touch'
    ]

    # Identify symptoms from user input
    identified_symptoms = [symptom for symptom in common_symptoms if symptom in user_input.lower()]
    
    # If pain is mentioned
    if 'pain' in identified_symptoms:
        # Check if details are already provided in current input
        if any(detail in user_input.lower() for detail in pain_details):
            return None  # Details are sufficient
        
        # Check if user history has already provided pain details
        for past_message in user_history:
            if any(detail in past_message for detail in ['sharp', 'dull', 'location', 'pain']):
                return None  # Pain details have already been provided

        # Ask for more pain details
        return "Can you tell me more about the pain? Where is it located? Is it sharp or dull?"

    # If no specific symptoms were identified
    if not identified_symptoms:
        return "I noticed you're not mentioning any specific symptoms. Can you describe where you feel discomfort or pain?"
    
    return None  # No follow-up is needed

#comforting response
def comforting_response(user_input):
    if "tired" in user_input.lower():
        return "I'm really sorry you're feeling this way. It must be tough, but remember, you're strong and there are ways we can manage the fatigue."
    
    if "hopeless" in user_input.lower():
        return "It's understandable to feel overwhelmed, but I'm here to help. Let's take it step by step. Have you been able to rest or talk to someone about how you're feeling?"
    
    return None


#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, I am your personal Medical Bot. I am here to assist you, How was your day?"
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
    
    # Call the symptom checker with user history
    follow_up_question = symptom_checker(message.content, user_history)
    
    # If a follow-up question is triggered, ask that instead of querying the LLM
    if follow_up_question:
        await cl.Message(content=follow_up_question).send()
        return
    
    # Call the chain and capture result
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
    final_answer = f"{answer} "
    await cl.Message(content=final_answer).send()

    # Store chainlit user session for later conversation continuity
    cl.user_session.set("chain", chain)






