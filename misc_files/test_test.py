import requests

# Test the simple test endpoint
test_file = r"d:\GIKI\CS 351\DTRA\CICIDS2017\test_10k_1.csv"

with open(test_file, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/api/test', files=files)
    print("Test endpoint Status:", response.status_code)
    print("Response:", response.json())
