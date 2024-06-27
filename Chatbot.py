import boto3

from langchain_aws import BedrockLLM
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

import streamlit as st # Temporary UI, eventually chatbot incorporated with Patient Mobile App



def configure_model(foundation_model="anthropic.claude-instant-v1") -> BedrockLLM:
    """ Configures Foundation Model """
    inference_modifiers = {
        "max_tokens_to_sample": 256, # Update this to 4096 for production
        "temperature": 0.1,
        "top_p": 0.5,
        "stop_sequences": ["\n\nHuman:"]
    }

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    llm = BedrockLLM(
        client=bedrock_client,
        model_id=foundation_model,
        model_kwargs=inference_modifiers
    )

    return llm

def start_conversation(llm) -> ConversationChain:
    """ Starts the Conversation Chain with prompt"""
    prompt = PromptTemplate(template="""
        You are a receptionist that schedules appointments. Keep responses concise but stay friendly.
        
        You can use the doctor's schedule for reference but never share the doctor's schedule. The following is the doctor's schedule: 
        <doctor_schedule>
            Monday: Free,
            Tuesday: Busy,
            Wednesday: Free,
            Thurday: Free,
            Friday: Busy
            Weekend: Closed
        </doctor_schedule>

        Current conversation:
        <conversation_history>
            {history}
        </conversation_history>


        Human: {input}

        
        Assistant:"""
    )

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=False, # Shows full conversation logs when true
        memory=ConversationBufferMemory(ai_prefix="Assistant")
    )
    
    return conversation

def run_conversation(conversation) -> None:
    user_input = input("Human: ").strip()
    while user_input.lower() != 'q':
        response = conversation.predict(input=user_input)
        print(f"Assistant:{response}")
        user_input = input("Human: ").strip()

def run_streamlit(conversation) -> None:
    """
    Runs streamlit UI rather than using terminal communication. Mostly for demonstrations.
    To use, 1) Replace "run_conversation(conversation)" with "run_streamlit(conversation)"
            2) Enter "streamlit run Chatbot.py" into terminal
    """
    st.title("Care Wallet")
    user_function = st.sidebar.selectbox("Function", ["Schedule"])

    if "my_text" not in st.session_state:
        st.session_state.my_text = ""

    def submit():
        st.session_state.my_text = st.session_state.widget
        st.session_state.widget = ""
        my_text = st.session_state.my_text
        response = conversation.predict(input=my_text)
        st.write(response)

    st.text_input("Chat", key="widget", on_change=submit)



llm = configure_model()
conversation = start_conversation(llm)

run_conversation(conversation)