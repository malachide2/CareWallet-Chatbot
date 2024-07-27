import json
import random
from datetime import datetime

def CreateJSON() -> None:
    """ Creates a JSON file filled with random patient data """
    patient_data = {}
    patient_names = ["Laura Diaz", "Rose Jackson", "Julia Robinson", "Patrick Gray"]
    insurance_providers = ["United Healthcare", "Blue Cross Blue Shield", "Aetna", "Kaiser Permanente"]

    for patient_name in patient_names:
        patient = {}
        patient["name"] = patient_name
        patient["age"] = random.randrange(18, 91)
        patient["phone"] = f"{random.randrange(100, 1000)}-{random.randrange(0, 1000):03}-{random.randrange(0, 10000):04}"
        patient["insurance"] = random.choice(insurance_providers)
        patient["last_appointment"] = f"{random.randrange(2020, 2024)}-{random.randrange(1, 12):02}-{random.randrange(1, 30):02}"
        patient["next_appointment"] = "None"

        patient_data[patient_name] = patient

    with open('data.json', 'w') as data_file:
        json.dump(patient_data, data_file)

def CheckAppointmentNeeded() -> list[str]:
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



CreateJSON()