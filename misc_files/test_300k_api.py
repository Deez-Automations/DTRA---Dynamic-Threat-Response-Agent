"""
Test if the API can process TEST_300k.csv
"""
import requests

# Test with the BIG file
test_file = r"d:\GIKI\CS 351\DTRA\CICIDS2017\TEST_300k.csv"

print("Testing API with:", test_file)

with open(test_file, 'rb') as f:
    files = {'file': f}
    try:
        print("Uploading... (this will take a while)")
        response = requests.post('http://localhost:5000/api/analyze', files=files, timeout=300)
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print("Total packets:", data['stats']['total_packets'])
            print("Threats:", data['stats']['threats_detected'])
        else:
            print("❌ ERROR Response:")
            print(response.text[:500])
    except Exception as e:
        print("❌ Exception:", e)
