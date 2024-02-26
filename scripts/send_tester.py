import requests

# Flask server URL
url = 'http://127.0.0.1:5000/predict'  # Server URL

# JSON data with the input sentence
data = {'created_at': '20240222',
        'diaryContent': '숨이 막혀 온다. 입에 침이 바짝 마르고 있다. 긴장되는 순간이다.',
        'user_email': 'naver@google.gmail'}


# Send POST request to the Flask server
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    print('created_at : ', response.json()['created_at'],"\n",
          'emotion_classification : ', response.json()['emotion_classification'],"\n",
          'user_email : ', response.json()['user_email'])
else:
    print('Failed to get prediction. Status code:', response.status_code)