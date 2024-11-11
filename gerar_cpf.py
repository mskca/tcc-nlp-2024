from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Inicializar o driver do Selenium
driver = webdriver.Chrome()

# Abrir a URL desejada
url = "https://www.4devs.com.br/gerador_de_cpf"
driver.get(url)
cpfs = []

# Execute um script JavaScript para obter o valor do campo textarea
for i in range(250):
    botao = driver.find_element(By.ID, "bt_gerar_cpf")
    botao.click()
    time.sleep(0.5)
    saida_cpf = driver.find_element(By.ID, "texto_cpf")
    cpf = saida_cpf.text
    cpfs.append(cpf)

print(cpfs)
driver.quit()

with open("cpfs.txt", "w") as file:
    file.write(cpfs)