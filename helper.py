import json
import random
from datetime import datetime, timedelta
from langchain_chroma import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter



def create_doctors_schedule() -> dict:
    """ Returns a dictionary filled with the doctors schedule """
    doctors_schedule = {}

    for i in range(5):
        schedule = {}
        for hour in range(24):
            time = ""
            if hour == 0: time = "12am"
            elif hour == 12: time = "12pm"
            elif hour < 12: time = f"{hour}am"
            else: time = f"{hour % 12}pm"

            if hour < 9 or hour > 16: schedule[time] = "Closed"
            elif hour < 12: schedule[time] = "Booked"
            else: schedule[time] = "Open"
        
        todays_date = datetime.today().strftime('%Y-%m-%d')
        date = (datetime.strptime(todays_date, '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
        doctors_schedule[date] = schedule

    return doctors_schedule

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
    data["doctors_schedule"] = create_doctors_schedule()
    data["patient_data"] = create_patient_data()

    with open('data.json', 'w') as data_file:
        json.dump(data, data_file)

def check_appointment_needed() -> list[str]:
    """ Returns a list of patient names that need a routine checkup """
    output = []

    with open('data.json', 'r') as data_file:
        data = json.load(data_file)
        patient_data = data["patient_data"]

    todays_date = datetime.today().strftime('%Y-%m-%d').split('-')
    for patient, patient_record in patient_data.items():
        if (int(todays_date[0]) - int(patient_record["last_appointment"].split('-')[0]) > 0 and
            int(todays_date[1]) >= int(patient_record["last_appointment"].split('-')[1])):
            output.append(patient)

    return output

def generate_embeddings(bedrock_client) -> Chroma:
    """ Converts JSON data into queryable vectorstore for RAG """
    with open('data.json', 'r') as data_file:
        data = json.load(data_file)

    splitter = RecursiveJsonSplitter(max_chunk_size=500)
    docs = splitter.create_documents(texts=[data])

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=BedrockEmbeddings(client=bedrock_client)
    )

    return vectorstore

def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    create_JSON()