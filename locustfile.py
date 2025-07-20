from locust import HttpUser, task, between
import random

class WebsiteUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def send_health_prediction(self):
        payload = {
            "Pregnancies": 2,
            "Glucose": random.randint(100, 180),
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 28.0,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 40,
            "sex": 1,
            "question": "What are early signs of diabetes?"
        }
        self.client.post("/predict", json=payload)
