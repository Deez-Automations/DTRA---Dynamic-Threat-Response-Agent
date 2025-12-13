"""
Test if the API can process a file without crashing
"""
import requests

# Test file
test_file = r"d:\GIKI\CS 351\DTRA\CICIDS2017\test_10k_1.csv"

print("Testing API with:", test_file)

with open(test_file, 'rb') as f:
    files = {'file': f}
    try:
        response = requests.post('http://localhost:5000/api/analyze', files=files)
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print("Total packets:", data['stats']['total_packets'])
            print("Threats:", data['stats']['threats_detected'])
        else:
            print("❌ ERROR Response:")
            print(response.text)
    except Exception as e:
        print("❌ Exception:", e)
        import traceback
        traceback.print_exc()
