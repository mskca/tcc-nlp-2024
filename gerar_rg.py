from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Inicializar o driver do Selenium
driver = webdriver.Chrome()

# Abrir a URL desejada
url = "https://www.4devs.com.br/gerador_de_rg"
driver.get(url)
rgs = []


# Execute um script JavaScript para obter o valor do campo textarea
for i in range(250):
    bt_gerar_rg = driver.find_element(By.ID, "bt_gerar_rg")
    bt_gerar_rg.click()

    time.sleep(0.5)

    rg = driver.find_element(By.ID, "texto_rg").get_attribute('value')
    rgs.append(rg)



print(rgs)
driver.quit()

with open("rgs.txt", "w") as file:
    file.write(rgs)