import requests

end_point = "http://127.0.0.1:6998/mistral_7b_instruct"

def make_chat_character(message, dialog):
    response = requests.post(end_point, json={'prompt':message, 'dialog':dialog})

    if response.status_code == 200:
        return response.json()
    else: 
        return "Error on the backend"