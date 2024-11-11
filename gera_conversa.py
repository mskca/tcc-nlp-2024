import json
import random
import requests
import time
from elevenlabs import generate
from elevenlabs import set_api_key
from elevenlabs import Voice, VoiceDesign, Gender, Age, Accent, play
from elevenlabs import save

PREV_KEY = None
KEY = None

def setApiKey():
    global PREV_KEY
    global KEY
    api_keys = ["sk_"]
    while True:
        KEY = random.choice(api_keys)
        if KEY != PREV_KEY:
            PREV_KEY = KEY
            break

    set_api_key(KEY)



base_path = r'C:\Users\msk\Documents\Projeto-NLP\API-ELEVENLABS-VOICE\roteiros-cpf'

def generate_voice(name, genero):
    if genero == 'M' or genero == 'Masculino' or genero == 'masculino':
        genero = Gender.male
    else:
        genero = Gender.female
    
    age = [Age.young, Age.middle_aged, Age.old]
    age = random.choice(age)

    design = VoiceDesign(
        name=name,
        text="Este será o texto padrão para todas as vozes geradas, com o intuito de chegar em 100 caracteres, espero que nao mude nada na voz.",
        voice_description="Exemplo de voz",
        gender=genero,
        age=age,
        accent=Accent.british,
        accent_strength=1.0,
    )
    voice = Voice.from_design(design)
    
    return voice

def gera_audio(voice, fala, conversaId, falaId):
    mp3 = "fala_"+conversaId+"_"+falaId+".mp3"

    audio = generate(
        text=fala,
        voice=voice,
        model="eleven_multilingual_v2"
    )
    
    save(audio, mp3)

    return 0

def remove_voices():
    base_url = "https://api.elevenlabs.io/v1/voices"

    headers = {
        "Accept": "application/json",
        "xi-api-key": f"{KEY}"
    }

    response = requests.get(base_url, headers=headers)

    carrega_json = json.loads(response.text)

    voice_ids = []

    for voice in carrega_json['voices']:
        if voice['category'] == 'generated':
            voice_ids.append(voice['voice_id'])

    for voice_id in voice_ids:
        url = base_url + "/" + voice_id
        response = requests.delete(url, headers=headers)

    time.sleep(2)

    return 0

conversaId = 23

for i in range(23, 50):
    print(f'Iniciando conversa: {conversaId}')

    file_path = f'{base_path}\\{i}.txt'
    try:
        with open(file_path, 'r') as file:
            # Read or process the contents of the file here
            file_contents = file.read()
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        data = json.loads(file_contents)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    
    nome1 = data['dialogo'][0]['nome']
    nome2 = data['dialogo'][1]['nome']

    setApiKey()

    print('removendo vozes\n')
    remove_voices()

    voice1 = generate_voice(nome1, data['dialogo'][0]['genero'])
    voice2 = generate_voice(nome2, data['dialogo'][1]['genero'])

    falaId = 0

    for fala in data['dialogo']:
        if fala['nome'] == nome1:
            gera_audio(voice1, fala['fala'], str(conversaId), str(falaId))
        else:
            gera_audio(voice2, fala['fala'], str(conversaId), str(falaId))
        
        falaId += 1

    conversaId += 1
    
    print('removendo vozes\n')
    remove_voices()
    print('reiniciando!!')
