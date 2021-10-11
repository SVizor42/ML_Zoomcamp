import requests


url = "http://localhost:9696/predict"
customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}
requests.post(url, json=customer).json()

response = requests.post(url, json=customer).json()
print(response)