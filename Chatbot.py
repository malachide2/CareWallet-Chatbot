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
        To schedule the appointment, you need to receive the patient's full name, their insurance, and the day they are available to have an appointment.
        Greet them like a receptionist before retrieving any information. Only ask one question at a time. Keep responses concise but stay friendly.
        End the conversation once you've received all the information you need.
        
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

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=False, # Shows full conversation logs when true
        memory=memory
    )
    
    # Conversation
    response = conversation.predict(input="Hello")
    print(f"Assistant:{response}")
    
    user_input = input("Human: ").strip()
    while user_input.lower() != 'q':
        response = conversation.predict(input=user_input)
        print(f"Assistant:{response}")
        user_input = input("Human: ").strip()



bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

llm = configure_model(bedrock_client)

patients_to_call = helper.CheckAppointmentNeeded()
for patient in patients_to_call:
    start_conversation(llm, patient)