import requests

# Flask server URL
url = 'http://127.0.0.1:5000/predict'  # Server URL

# JSON data with the input sentence
data = {'sentence': '미치고 팔짝뛰겠는데 아 미치겠다'}

# Send POST request to the Flask server
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print('Prediction:', response.json()['predicted_result'])
else:
    print('Failed to get prediction. Status code:', response.status_code)