import json
import random
from datetime import datetime, timedelta

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_aws import ChatBedrockConverse



def setup_llm(foundation_model="anthropic.claude-3-sonnet-20240229-v1:0") -> ChatBedrockConverse:
    """ Configures Foundation Model """  
    inference_modifiers = {
        # "max_tokens_to_sample": 256, # Update this to 4096 for production
        "temperature": 0.1,
        "top_p": 0.5,
        "stop_sequences": ["\n\nHuman:"]
    }

    llm = ChatBedrockConverse(
        model=foundation_model,
        temperature=0.1,
        max_tokens=1024
    )

    return llm

def generate_embeddings(bedrock_client) -> VectorStoreRetriever:
    """ Converts JSON data into queryable vectorstore for RAG """
    with open('data.json', 'r') as data_file:
        data = json.load(data_file)

    splitter = RecursiveJsonSplitter(max_chunk_size=500)
    docs = splitter.create_documents(texts=[data])

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=BedrockEmbeddings(client=bedrock_client)
    )
    retriever = vectorstore.as_retriever()

    return retriever

def create_doctor_schedule() -> dict:
    """ Returns a dictionary filled with a random doctor schedule """
    doctor_schedule = {}

    for i in range(5):
        schedule = {}
        for hour in range(24):
            time = ""
            if hour == 0: time = "12am"
            elif hour == 12: time = "12pm"
            elif hour < 12: time = f"{hour}am"
            else: time = f"{hour % 12}pm"

            if hour < 9 or hour > 16: schedule[time] = "Closed"
            else:
                choices = ["Booked", "Open"]
                schedule[time] = random.choice(choices)
        
        date = (datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d')
        doctor_schedule[date] = schedule

    return doctor_schedule

def create_patient_data() -> dict:
    """ Returns a dictionary filled with random patient information """
    patient_data = {}
    patient_names = ["Laura Diaz", "Rose Jackson", "Julia Robinson", "Patrick Gray"]
    insurance_providers = ["United Healthcare", "Blue Cross Blue Shield", "Aetna", "Kaiser Permanente"]

    for patient_name in patient_names:
        patient = {}
        patient["name"] = patient_name
        patient["age"] = str(random.randrange(18, 91))
        patient["phone"] = f"{random.randrange(100, 1000)}-{random.randrange(0, 1000):03}-{random.randrange(0, 10000):04}"
        patient["insurance"] = random.choice(insurance_providers)
        patient["last_appointment"] = f"{random.randrange(2020, 2024)}-{random.randrange(1, 12):02}-{random.randrange(1, 30):02}"
        patient["next_appointment"] = "None"

        patient_data[patient_name] = patient

    return patient_data

def create_JSON() -> None:
    """ Creates a JSON file filled with key documents """
    data = {}
    data["doctor_schedule"] = create_doctor_schedule()
    data["patient_data"] = create_patient_data()

    with open('data.json', 'w') as data_file:
        json.dump(data, data_file)

def check_appointment_needed() -> list[str]:
    """ Returns a list of patient names that need a routine checkup """
    with open('data.json', 'r') as data_file:
        data = json.load(data_file)
        patient_data = data["patient_data"]

    output = []
    todays_date = datetime.today().strftime('%Y-%m-%d').split('-')
    for patient, patient_record in patient_data.items():
        if (int(todays_date[0]) - int(patient_record["last_appointment"].split('-')[0]) > 0 and
            int(todays_date[1]) >= int(patient_record["last_appointment"].split('-')[1])):
            output.append(patient)

    return output

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    create_JSON()