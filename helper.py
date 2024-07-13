import json
from datetime import datetime

def ResetJSON() -> None:
    patient_schedules = {
        "Abby": "2024-06-01",
        "Bob": "2022-06-05",
        "Carl": "2022-06-03"
    }

    with open('data.json', 'w') as data_file:
        json.dump(patient_schedules, data_file)

def CheckAppointmentNeeded() -> list[str]:
    """ Returns a list of patient names that need a routine checkup """
    output = []

    with open('data.json', 'r') as data_file:
        patient_schedules = json.load(data_file)

    for patient, last_appointment in patient_schedules.items():
        this_year = int(datetime.today().strftime('%Y-%m-%d').split('-')[0])
        year_of_last_appointment = int(last_appointment.split('-')[0])
        if this_year - year_of_last_appointment > 0:
            output.append(patient)

    return output

ResetJSON()