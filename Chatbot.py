import boto3
import helper

from langchain_aws import BedrockLLM
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate



def configure_model(bedrock_client, foundation_model="anthropic.claude-instant-v1") -> BedrockLLM:
    """ Configures Foundation Model """
    inference_modifiers = {
        "max_tokens_to_sample": 256, # Update this to 4096 for production
        "temperature": 0.1,
        "top_p": 0.5,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = BedrockLLM(
        client=bedrock_client,
        model_id=foundation_model,
        model_kwargs=inference_modifiers
    )

    return llm

def start_conversation(llm, patient_name) -> None:
    """ Starts the Conversation Chain with context"""
    
    context = f"""
        Context: You are a receptionist calling {patient_name} to schedule an appointment because it's been at least one year.
        To schedule the appointment, you need to confirm the patient's full name and their insurance, then find a day they are available to have an appointment.
        End the conversation once you've received all the information you need and explicitly say "bye".

        Greet them like a receptionist before retrieving any information. Only ask one question at a time. Use three sentences maximum and keep responses concise and friendly.
        If you don't know an answer, just say that you don't know but always check the patient information before saying you don't know. 
        
        You can use the doctor's schedule for reference but never share the doctor's schedule. The following is the doctor's schedule: 
        <doctor_schedule>
            Monday: Free,
            Tuesday: Busy,
            Wednesday: Free,
            Thurday: Free,
            Friday: Busy
            Weekend: Closed
        </doctor_schedule>
    """
    
    retriever = vectorstore.as_retriever()
    rag_documents = helper.format_docs(retriever.invoke(context))
    rag_context = f"""
        The following is information related to the patient:
        <patient_information>
            {rag_documents}
        <patient_information>
    """

    prompt = PromptTemplate(
        template="""
        Current conversation:
        <conversation_history>
            {history}
        </conversation_history>


        Human: {input}

        
        Assistant:"""
    )

    memory = ConversationBufferMemory(ai_prefix="Assistant")
    memory.chat_memory.add_user_message(context)
    memory.chat_memory.add_ai_message("I am a receptionist calling to schedule an appointment")
    if rag_documents:
        memory.chat_memory.add_user_message(rag_context)
        memory.chat_memory.add_ai_message("This is information relating to this patient")

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=False, # Shows full conversation logs when true
        memory=memory
    )
    
    # Conversation
    response = conversation.predict(input="Hello")
    print(f"Assistant:{response}")
    
    while "bye" not in response and "Bye" not in response:
        user_input = input("Human: ").strip()
        response = conversation.predict(input=user_input)
        print(f"Assistant:{response}")



bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

llm = configure_model(bedrock_client)
vectorstore = helper.generate_embeddings(bedrock_client)

patients_to_call = helper.check_appointment_needed()
for patient in patients_to_call:
    print("\n\n")
    start_conversation(llm, patient)