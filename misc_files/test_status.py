import requests

# Test status endpoint
response = requests.get('http://localhost:5000/api/status')
print("Status endpoint:", response.status_code)
print(response.json())
