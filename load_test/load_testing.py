import time
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def homepage(self):
        self.client.get(url="/")

    
    @task
    def prediction(self):
        self.client.get(url="/prediction")