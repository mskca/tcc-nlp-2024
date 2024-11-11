import requests
from bs4 import BeautifulSoup
import json
import time

# Define the URL and the POST data
url = "https://www.4devs.com.br/ferramentas_online.php"
data = {
    "acao": "gerar_cep",
    "cep_estado": "RS",
    "cep_cidade": "",
    "somente_numeros": "S"
}

# Define the headers
headers = {
    "Sec-Ch-Ua": '"Not/A)Brand";v="8", "Chromium";v="126"',
    "Accept-Language": "en-US",
    "Sec-Ch-Ua-Mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.127 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept": "*/*",
    "X-Requested-With": "XMLHttpRequest",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Origin": "https://www.4devs.com.br",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.4devs.com.br/gerador_de_cep",
    "Accept-Encoding": "gzip, deflate, br",
    "Priority": "u=1, i"
}

# Initialize an empty list to store the JSON objects
results = []

# Run the request 500 times
for i in range(100):
    # Send the POST request
    response = requests.post(url, headers=headers, data=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract values by ID
        try:
            cep = soup.find('div', id='cep').find('span').text
            endereco = soup.find('div', id='endereco').find('span').text
            bairro = soup.find('div', id='bairro').find('span').text
            cidade = soup.find('div', id='cidade').find('span').text
            estado = soup.find('div', id='estado').find('span').text

            # Create a JSON object
            result = {
                "cep": cep,
                "endereco": endereco,
                "bairro": bairro,
                "cidade": cidade,
                "estado": estado
            }

            # Append the result to the list
            print(result)
            results.append(result)


        except AttributeError:
            print(f"Failed to parse response at iteration {i+1}. Skipping.")

    else:
        print(f"Failed to retrieve data at iteration {i+1}. Status code: {response.status_code}")
    
    # Add a 1-second delay between each request
    time.sleep(1)

# Write the list of JSON objects to a file
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("Process completed. Results saved to 'results.json'.")
