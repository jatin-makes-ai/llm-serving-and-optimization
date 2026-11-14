import requests

def query(prompt):
    msgs = [{'role': 'user', 'content': prompt}]
    r = requests.post('http://localhost:8000/v1/chat/completions', json={'model': 'qwen-1.5b', 'messages': msgs, 'temperature': 0.1, 'max_tokens': 150})
    return r.json()['choices'][0]['message']['content'].strip()

print('--- Hindi Devanagari Output ---')
print(query('भारत के प्रथम प्रधानमंत्री कौन थे? केवल उनका नाम एक वाक्य में लिखिए।'))

print('\n--- Bengali Output ---')
print(query('ভারতের জাতীয় পশু কোনটি? এক কথায় উত্তর দাও।'))

print('\n--- Tamil Output ---')
print(query('இந்தியாவின் தலைநகரம் எது? ஒரு வரியில் பதில் சொல்லுங்கள்.'))
