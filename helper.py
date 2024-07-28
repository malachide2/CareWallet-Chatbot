import json
import random
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter

def create_JSON() -> None:
    """ Creates a JSON file filled with random patient data """
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

    with open('data.json', 'w') as data_file:
        json.dump(patient_data, data_file)

def check_appointment_needed() -> list[str]:
    """ Returns a list of patient names that need a routine checkup """
    output = []

    with open('data.json', 'r') as data_file:
        patient_data = json.load(data_file)

    todays_date = datetime.today().strftime('%Y-%m-%d').split('-')
    for patient in patient_data.values():
        if (int(todays_date[0]) - int(patient["last_appointment"].split('-')[0]) > 0 and
            int(todays_date[1]) >= int(patient["last_appointment"].split('-')[1])):
            output.append(patient["name"])

    return output

def generate_embeddings(bedrock_client) -> Chroma:
    """ Converts JSON data into queryable vectorstore for RAG """
    with open('data.json', 'r') as data_file:
        patient_data = json.load(data_file)

    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    docs = splitter.create_documents(texts=[patient_data])

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=BedrockEmbeddings(client=bedrock_client)
    )

    return vectorstore

def format_docs(docs) -> str:
    # return "\n".join(doc.page_content for doc in docs)
    # Scalable Solution ^
    return docs[0].page_content

# create_json()