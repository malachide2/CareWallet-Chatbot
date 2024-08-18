import sys
import os
import unittest
import boto3
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import helper
from src import chatbot



class TestConversation(unittest.TestCase):
    def setUp(self):
        self.conversation = chatbot.Conversation("Laura Diaz")
        self.conversation.start_conversation("2024-08-05", True)

    def test_check_appointment(self):
        """ Ensures the patients who haven't had an appointment in a year are the only ones called """
        temp = tuple(helper.check_appointment_needed('test/test.json'))
        self.assertEqual(temp, ("Laura Diaz", "Patrick Gray"))

    def test_find_schedule(self):
        """ Ensures RAG properly finds the doctor's schedule for each date """
        isCorrect = True
        for i in range(5):
            date = (datetime.strptime("2024-08-05", '%Y-%m-%d') + timedelta(days=i)).strftime('%Y-%m-%d')
            isCorrect = isCorrect and date in self.conversation.find_schedule.invoke(date)
        self.assertTrue(isCorrect)

    def test_retrieve_patient_information(self):
        user_inputs = [
            "Hello?",
            "This is Laura Diaz",
            "What does my records say for my insurance?"
        ]
        response = ""
        for user_input in user_inputs:
            response = self.conversation.generate_response(user_input)
        self.assertIn("Kaiser Permanente", response.title())

    def test_correct_doctor_schedule_negative(self):
        response = self.conversation.generate_response("Is the doctor free August 8th at 1pm? Answer with a yes or no")
        self.assertIn("no", response.lower())

    def test_correct_doctor_schedule_affirmative(self):
        response = self.conversation.generate_response("Is the doctor free August 8th at 2pm? Answer with a yes or no")
        self.assertIn("yes", response.lower())



unittest.main()