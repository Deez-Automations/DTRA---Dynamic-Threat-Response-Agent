import requests

# Just ping the endpoint to see if it exists
try:
    response = requests.get('http://localhost:5000/api/analyze')
    print("GET Status:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
