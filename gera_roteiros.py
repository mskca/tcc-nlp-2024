import openai
import json

def load_ceps_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def generate_script(identification, nome_usuario, nome_atendente, endereco, refer): # Função que utiliza uma requisição para a API do ChatGPT para gerar um roteiro utilizando um CPF e um nome aleatório
    openai.api_key = "sk-"

    completion = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Você é um roteirista que cria scripts de atendimento ao cliente e a saída deve sempre estar no formato JSON."},
        {"role": "user", "content": f"""
            Crie um script de uma chamada de suporte ao cliente onde o atendente se chama {nome_atendente} e o usuário se chama {nome_usuario}. 
            Durante a conversa, ambos devem citar o número de identificação {identification}, referindo-se a ele como {refer}. 
            O usuário deve fornecer o endereço {endereco} em algum momento da conversa. 
            A saída deve ser um objeto JSON com a seguinte estrutura:
            {{
                "falas": [
                    {{
                        "nome": "Nome da pessoa que fala",
                        "genero": "M ou F",
                        "tipo": "usuario ou atendente",
                        "fala": "Texto da fala"
                    }},
                    ...
                ]
            }}
            O JSON deve conter várias falas seguindo essa estrutura.
            """}
    ]
    )
    return completion


nome_usuario = [
"MARIA", "ANA", "JOAO", "GABRIEL", "LUCAS", "PEDRO", "MATEUS", "JOSE", "GUSTAVO", "VITORIA", "GUILHERME", "CARLOS", "JULIA", "VITOR", "FELIPE", "LETICIA", "MARCOS", "RAFAEL", "LUIZ", "DANIEL", "EDUARDO", "MATHEUS", "LUIS", "AMANDA", "BRUNO", "BEATRIZ", "GABRIELA", "LARISSA", "PAULO", "LEONARDO", "MARIANA", "VINICIUS", "DAVI", "BRUNA", "SAMUEL", "VICTOR", "CAMILA", "ISABELA", "FRANCISCO", "CAIO", "ANTONIO", "TIAGO", "LUANA", "SARA", "EDUARDA", "BIANCA", "RAFAELA", "GEOVANA", "FERNANDA", "IGOR", "NATALIA", "LAURA", "LUAN", "ANDRE", "NICOLAS", "JULIANA", "MIGUEL", "NICOLE", "THIAGO", "GABRIELE", "YASMIN", "ARTHUR", "HENRIQUE", "CAUA", "RODRIGO", "RAISSA", "DIEGO", "SABRINA", "LIVIA", "ARTUR", "LARA", "RUAN", "ALINE", "MILENA", "RIAN", "GIOVANA", "JESSICA", "BRENDA", "KAUAN", "VANESSA", "DEBORA", "PABLO", "DAVID", "ANDERSON", "CAROLINE", "BRENO", "RENAN", "MARCELO", "LEANDRO", "FERNANDO", "ALICE", "SOFIA", "LUIZA", "ALAN", "ANDRESSA", "ISADORA", "TAINA", "JULIO", "DIOGO", "KAUA", "REBECA", "MURILO", "ALEXANDRE", "DANILO", "DOUGLAS", "YASMIM", "ERICA", "WESLEY", "TAIS", "EMILY", "CARLA", "VINICIOS", "ALEX", "RAIANE", "ISABELE", "RAQUEL", "RICARDO", "FABIO", "EMILI", "ITALO", "LAIS", "DANIELE", "CAMILE", "TAINARA", "BARBARA", "CLARA", "TALITA", "JENIFER", "CAROLINA", "LORENA", "CAUAN", "PAMELA", "ENZO", "JEFERSON", "FABRICIO", "SAMARA", "ESTER", "EVELIN", "IAGO", "INGRID", "YURI", "DANIELA", "ERICK", "TAMIRES", "MARINA", "JAQUELINE", "ERIC", "IASMIN", "OTAVIO", "MAICON", "JEAN", "FLAVIA", "NATAN", "MARCELA", "THAIS", "RENATA", "IASMIM", "ALANA", "ESTEFANI", "EVERTON", "ALISSON", "ADRIELE", "EMERSON", "RYAN", "CAIQUE", "EMANUEL", "MANUELA", "MAIARA", "KAIQUE", "FRANCISCA", "GABRIELI", "JAMILE", "JOICE", "ANTONIA", "CRISTIAN", "WILLIAN", "MARIANE", "ALESSANDRA", "ADRIANO", "VICTORIA", "PAULA", "DAIANE", "ALISON", "HUGO", "JUAN", "EMILLY", "RENATO", "FLAVIO", "JONAS", "BERNARDO", "GISELE", "PATRICIA", "WESLEI", "GIOVANNA", "MICHELE", "LUCIANO", "EMILE", "RICHARD", "ADRIAN", "ANNA", "MELISSA", "ELEN", "MARCIO", "JORGE", "MARCO", "EMANUELE", "ERIK", "FRANCIELE", "VIVIANE", "IAN", "ELOISA", "KEVIN", "PAOLA", "KAIO", "CLEITON", "KAUE", "SAMIRA", "MAURICIO", "MAISA", "EDSON", "ROBERTA", "LUISA", "ROBSON", "NAIARA", "ALESSANDRO", "KAILANE", "ISABELLA", "ELIAS", "LAVINIA", "MARLON", "FILIPE", "KAUANE", "HELOISA", "KAREN", "ANDREI", "WILIAN", "PALOMA", "NATANAEL", "MOISES", "JANAINA", "SARAH", "MIRELA", "STEFANI", "CAUE", "JOANA", "IARA", "ARIANE", "MICAEL", "JONATAN", "SOPHIA", "KARINE", "ISABEL", "JUNIOR", "FABIANA", "ELLEN", "AUGUSTO", "LUCIANA", "IZABELA", "ESTEFANE", "GRAZIELE", "PIETRO", "ISABELI", "PRISCILA", "NATALI", "RAISA", "JOSUE", "HEITOR", "MAYARA", "POLIANA", "ROBERTO", "JONATAS", "ISAQUE", "RAYANE", "ISRAEL", "RAMON", "EZEQUIEL", "LAISA", "KARINA", "ICARO", "SERGIO", "EVELYN", "ADRIANA", "MANOEL", "WANDERSON", "ISAC", "YAN", "AGATA", "MICHAEL", "HELENA", "ELAINE", "NICOLI", "YAGO", "ANE", "ROBERT", "LAIANE", "VERONICA", "MONIQUE", "RAYSSA", "CLAUDIO", "GEAN", "WENDEL", "ISMAEL", "ADRIEL", "DEIVID", "LAILA", "NATHALIA", "MAIRA", "PATRICK", "IURI", "DENILSON", "JONATHAN", "MANOELA", "KAIKE", "CRISTIANO", "VALERIA", "TALIA", "SUELEN", "CAUANE", "KELVIN", "MICHEL", "ISABELLE", "JONATA", "MILENE", "ALICIA", "CECILIA", "CAROLAINE", "GEOVANNA", "RIQUELME", "BEATRIS", "CAMILI", "GEOVANE", "MIRIAN", "GUILERME", "MIKAEL", "HELEN", "ARIEL", "RONALDO", "TATIANE", "LUDIMILA", "IRIS", "KELLY", "ISAAC", "JHONATAN", "STEFANY", "ERIKA", "RAUL", "RAI", "LARISA", "CINTIA", "GABRIELLE", "RONALD", "GIULIA", "JENNIFER", "MARIO", "ANGELA", "TALES", "RAIMUNDO", "VIVIAN", "JULIANO", "TAIANE", "GABRIELLY", "RITA", "WELITON", "JULIANE", "CICERO", "WILLIAM", "ANGELO", "MONICA", "CESAR", "FABIANO", "TAMARA", "PIETRA", "MARCIA", "LEVI", "ELOA", "ROGERIO", "ANGELICA", "ANDREIA", "ISAIAS", "KEILA", "NAYARA", "ANDREZA", "VAGNER", "NATHAN", "JACKSON", "ROGER", "ALVARO", "ALEXANDRA", "JEOVANA", "LORRANE", "MARILIA", "CRISTIANE", "CARINE", "WELLINGTON", "MICAELE", "CLARICE", "JAINE", "KETLIN", "MAYCON", "ANDRESA", "GRAZIELA", "ALEXANDRO", "ISABELLY", "IVAN", "TAUANE", "BRYAN", "JOAQUIM", "JOYCE", "JOEL", "GESSICA", "ANTONY", "LEANDRA", "ESTEFANY", "JOSIANE", "ELISA", "ANTONI", "CLAUDIA", "DANIELI", "ANNE", "WEVERTON", "ADRIELI", "JAIANE", "HELLEN", "LUDMILA", "DENIS", "CAILANE", "CHARLES", "DENISE", "HIAGO", "JAMILI", "JHONATA", "EMANUELI", "TALISON", "CARINA", "CRISTINA", "AGATHA", "CAROL", "TAILA", "HIGOR", "KESIA", "ESTELA", "IZADORA", "VALENTINA", "EMANOEL", "SAULO", "FABIOLA", "ANDRIELE", "WAGNER", "ELIANE", "KEVEN", "NAIANE", "SAVIO", "CASSIO", "LANA", "CASSIA", "KAROLINE", "ERIQUE", "MARTA", "TAISSA", "IZABELE", "DIANA", "THAINA", "MANUEL", "ALLAN", "JHENIFER", "ROMULO", "ANDREY", "KAIK", "INGRIDE", "BRIAN", "JADSON", "SANDRO", "JAILSON", "TAISA", "CIBELE", "ALEXIA", "SAMANTA", "JEFFERSON", "LILIAN", "ADRIA", "CLEBER", "MIRELE", "WELINGTON", "MARCUS", "LORRAINE", "NATIELE", "CATARINA", "EVILIN", "TAILANE", "LAIZA", "BRAIAN", "VANDERSON", "ADRIANE", "LAZARO", "THALITA", "NATALY", "KAMILA", "LIDIA", "LUANE", "WALISON"
]

nome_atendente = [
"MARIELE", "KETLEN", "EMANUELA", "KAYLANE", "EVANDRO", "GEOVANI", "SANDRA", "CAMILLY", "YARA", "RAPHAEL", "MIQUEIAS", "BEATRICE", "ISIS", "ELTON", "NICOLY", "ARIELE", "CASSIANE", "GABRIELY", "RUTE", "JENIFFER", "LILIANE", "DARLAN", "WALLACE", "SUZANA", "LUCA", "GABRIELLA", "WALACE", "JORDANA", "CRISLAINE", "STEFANE", "EMANUELLE", "WELINTON", "RAILANE", "LIDIANE", "LORENZO", "CAMILY", "TALISSON", "MAX", "LAUANE", "BRAYAN", "GIOVANI", "MICAELA", "SIMONE", "JHON", "REINALDO", "BRENA", "ALBERTO", "WILSON", "TATIANA", "MAYRA", "ABRAAO", "MAURO", "NICOLLY", "IZAQUE", "KELI", "MICHELI", "PRICILA", "RAILSON", "WILIAM", "DAIANA", "MIKAELE", "MICAELI", "ALEXSANDRO", "MARCELE", "MAYSA", "ANNY", "EDILSON", "JOELMA", "GILBERTO", "TAYNA", "RAINARA", "JACSON", "JADE", "KAUANY", "ANANDA", "FABRICIA", "FRANCIELI", "KARLA", "SILAS", "ELOIZA", "GIOVANE", "TAYNARA", "CAMILLE", "FABIANE", "BRENDO", "DANDARA", "JOSIEL", "ELANE", "DEIVISON", "THALES", "NATASHA", "NAIRA", "RUBENS", "GISLAINE", "ABNER", "JANDERSON", "SEBASTIAO", "ESTHER", "ELIZA", "LUARA", "ELEM", "JAMILY", "VINICIO", "KALINE", "DEVID", "THALIA", "SANDY", "GRASIELE", "MIRIAM", "WEMERSON", "ANDRIELI", "REGINALDO", "WALISSON", "KETELIN", "THAYNA", "ANDREINA", "KAUANI", "KAMILLY", "THAMIRES", "TOMAS", "ENRIQUE", "SAMILA", "RAIANA", "INGRIDI", "ELIANA", "PETERSON", "AILTON", "EVELLYN", "ANALICE", "REGIANE", "LEIDIANE", "GILSON", "HUDSON", "GLORIA", "CLARISSE", "ELIEL", "SILVIA", "EVERSON", "RANIELE", "LUMA", "DAYANE", "NOEMI", "JARDEL", "JANIELE", "MAICOM", "WENDERSON", "RAILA", "ERIKE", "LUCCA", "MAIANE", "DANIELLE", "FATIMA", "ESTEVAO", "STHEFANY", "YGOR", "EMANUELLY", "ANDREA", "SAMILE", "IURY", "NUBIA", "JOSEANE", "GLENDA", "DEISE", "SUELEM", "ALANE", "KAYKY", "MANUELE", "GILMAR", "LUZIA", "ANGELINA", "ANTHONY", "SILVIO", "JOSIAS", "RAFAELLA", "JOHN", "LEILA", "THOMAS", "TAUAN", "GRACIELE", "VALTER", "ADILSON", "ROSANA", "GLEICE", "RUTH", "HELIO", "GLEICIANE", "CRISLANE", "RAFAELE", "ISABELLI", "TULIO", "TAMIRIS", "IZABEL", "JOELSON", "JAKSON", "SILVANA", "KAMILE", "EDER", "EDNA", "PATRIK", "ISABELY", "CRISTOFER", "MATIAS", "CLEBERSON", "GRAZIELI", "KELY", "MISAEL", "JOSEFA", "MIRELLA", "HENRY", "ROSANGELA", "MAIZA", "LAISLA", "MARA", "FRANCINE", "KAROLAINE", "SAMIA", "REGINA", "APARECIDA", "JAIRO", "ANDREW", "EVA", "IGO", "LORRANA", "NATACHA", "STEPHANY", "VANESA", "CLARISSA", "RAILAN", "FREDERICO", "KETLYN", "SHEILA", "ELISANGELA", "MESSIAS", "NARA", "JAIR", "CASSIANO", "EVELIM", "ELOISE", "LORANE", "OLIVIA", "CICERA", "JESICA", "IURE", "GEORGE", "NADIA", "INGRED", "RENAM", "CAUANI", "KATIA", "ALEF", "WENDER", "ALEXANDER", "GERSON", "RUAM", "EMELI", "ELIZABETE", "WESLLEY", "GABRIELLI", "LAISSA", "EMELY", "LIGIA", "RAINE", "CAUAM", "GEFERSON", "JUSSARA", "LUCIENE", "JAMILLY", "THEO", "NELSON", "MAXUEL", "LAINE", "WELLITON", "GISELI", "KEMILI", "FAGNER", "JOSIELE", "YURE", "VICENTE", "KAYKE", "KAWAN", "CHRISTIAN", "MEL", "LUCIO", "MACIEL", "DALILA", "ELIZEU", "KELIANE", "LUCIA", "KLEBER", "GENILSON", "JULHA", "KIARA", "IZABELI", "KAUAM", "MIRELI", "LAYSA", "KEMILY", "GLEISON", "MIZAEL", "KEZIA", "KAREM", "KAIKY", "DIENIFER", "CHAIANE", "GERALDO", "ROMARIO", "CALEBE", "RAICA", "RIVALDO", "ROSIANE", "THAINARA", "IANA", "MARCIEL", "DERICK", "PATRIC", "GEANE", "RIAM", "KAUANA", "MARCELI", "MIKAELI", "HANNA", "SAMOEL", "ELISON", "IZABELLA", "NATHALY", "NATIELI", "RHUAN", "AILA", "LUCIELE", "CAIKE", "INACIO", "STEPHANIE", "KEMILLY", "CAREN", "THAYNARA", "SILMARA", "LINCON", "LAISE", "CELSO", "ELISEU", "BERNADO", "ARTU", "LEO", "EDMILSON", "LUCIANE", "RAY", "LUIGI", "RAYSA", "ADRIELLY", "JEANE", "SIDNEI", "LORRAN", "TATIELE", "EVELY", "GLEIDSON", "RODOLFO", "DEIVIDE", "TAINAR", "WILIANE", "NICOLLE", "IANE", "WELISON", "ROSA", "PAULINA", "ELIS", "JOISE", "CLEISON", "GILVAN", "LAYLA", "ADSON", "TALIS", "RAMOM", "GEISIANE", "LIANDRA", "MILLENA", "RAIZA", "VANDERLEI", "ELIZANGELA", "GEISA", "JOABE", "SUELI", "ALESANDRA", "ATILA", "DARA", "RANA", "KAMILY", "NILTON", "HEBERT", "FRANCIANE", "THAISSA", "JACIARA", "KAILA", "EDILENE", "RAFAELI", "SUELLEN", "HUMBERTO", "AIRTON", "LAUREN", "MAGNO", "MICHELLE", "EDERSON", "WELLINTON", "MARISA", "MANUELLA", "ESDRAS", "KESSIA", "RAQUELE", "JONATHA", "NADSON", "AFONSO", "LARYSSA", "TAUANI", "KAUN", "DAMIAO", "TARCISIO", "HENRRIQUE", "RONI", "ADRIELY", "ANITA", "NINA", "KIMBERLY", "TANIA", "ADRIELLE", "LUCIMARA", "FLAVIANE", "VALQUIRIA", "MARIELI", "ILANA", "TEREZA", "ANY", "DAVILA", "UESLEI", "CLAUDIANE", "EDVALDO", "NICHOLAS", "EDGAR", "GIAN", "DOMINIQUE", "ARIANA", "MAIKON", "MIKAELA", "RIANA", "GENIFER", "VALESCA", "LORAINE", "EZEQUIAS", "SANTIAGO", "JAIME", "NATA", "NAELI", "VALMIR", "WELLIGTON", "ELVIS", "RUBIA", "IRAN", "MAITE", "KAIANE", "ADAILTON", "TIFANI", "THAISA", "MONALISA", "ROSEANE", "ROSILENE", "EVELI", "SORAIA", "VALDIR", "WASHINGTON", "DANIELLY", "RAIMUNDA", "VITORIO", "SAFIRA", "TAUANA", "ALESANDRO", "WANESSA", "LAYANE", "NIVIA", "JAKELINE", "JESSE", "CAMILLA", "LORRANY", "KETELEN", "RAQUELI", "JACKELINE", "ELDER", "ALEJANDRO", "CLEIDIANE", "EDIVAN", "KAROLINA", "WENDELL", "KAILAINE", "EMANUELY", "ELITON", "WELTON", "JEREMIAS", "ORLANDO", "ADENILSON", "EDVAN", "JORDAN", "DAFNE", "LIA"
]

rgs = ['30.748.114-1', '25.830.168-5', '36.881.147-5', '29.682.693-5', '10.946.486-2', '46.874.322-4', '50.956.913-4', '14.867.049-0', '29.125.849-9', '32.635.471-2', '41.411.643-4', '50.893.707-3', '44.620.287-3', '22.646.270-5', '38.132.407-2', '45.662.167-2', '40.543.347-5', '34.782.825-5', '38.057.064-6', '13.025.645-6', '28.883.643-1', '29.096.223-7', '13.421.825-5', '16.393.135-5', '48.258.682-5', '12.650.609-7', '41.797.790-6', '27.708.594-9', '45.431.475-9', '11.043.293-9', '28.463.435-9', '15.926.311-6', '36.033.628-0', '22.742.588-1', '26.233.541-4', '17.064.871-0', '22.022.519-9', '11.429.149-4', '35.422.937-0', '20.821.409-4', '12.351.117-3', '48.873.181-1', '42.268.651-7', '45.593.836-2', '39.579.521-7', '29.890.002-6', '43.307.144-8', '35.292.733-1', '17.505.939-1', '42.913.318-2', '20.442.148-2', '48.284.420-6', '24.160.040-6', '17.858.805-2', '19.576.371-3', '40.246.587-8', '35.302.215-9', '28.863.799-9', '50.741.088-9', '45.526.213-5', '23.347.734-2', '17.888.322-0', '41.610.097-1', '40.355.935-2', '41.489.092-9', '25.972.535-3', '15.602.758-6', '50.028.141-5', '14.761.458-2', '47.352.193-3', '47.297.171-2', '26.087.011-0', '13.642.453-3', '22.975.943-9', '46.099.336-7', '15.588.222-3', '40.260.743-0', '46.048.418-7', '31.456.948-0', '29.445.719-7', '27.082.611-7', '23.615.117-4', '24.265.900-7', '27.665.467-5', '50.183.859-4', '38.402.254-6', '25.446.560-2', '45.589.500-4', '21.681.345-1', '23.122.637-8', '12.076.973-6', '14.245.162-9', '46.754.378-1', '22.237.311-8', '43.022.085-6', '40.488.793-4', '11.707.539-5', '48.145.422-6', '47.193.829-4', '50.406.588-9', '40.170.660-6', '18.070.654-8', '32.106.700-9', '15.659.072-4', '39.126.989-6', '28.184.559-1', '11.591.355-5', '33.005.943-9', '11.544.952-8', '18.500.635-8', '21.468.427-1', '40.377.869-4', '37.216.760-3', '22.567.893-7', '44.770.869-7', '33.398.797-4', '24.632.940-3', '14.308.197-4', '23.509.587-4', '35.220.326-2', '38.858.054-9', '41.995.544-6', '17.605.535-6', '14.605.076-9', '47.437.205-4', '49.215.870-8', '11.538.380-3', '33.465.079-3', '16.111.107-5', '11.142.727-7', '12.209.194-2', '28.428.618-7', '48.435.132-1', '20.660.403-8', '47.724.769-6', '31.017.020-5', '41.474.894-3', '27.543.410-2', '29.278.693-1', '25.150.483-9', '30.201.934-0', '12.470.440-2', '15.022.066-2', '21.197.682-9', '10.727.947-2', '25.759.902-2', '24.660.747-6', '15.789.855-6', '50.995.319-0', '25.592.511-6', '26.439.621-2', '25.815.551-6', '14.637.092-2', '36.909.194-2', '44.200.019-4', '26.139.991-3', '44.242.222-2', '16.481.052-3', '43.473.755-0', '49.444.797-7', '36.430.529-0', '29.971.791-4', '40.313.993-4', '26.871.448-4', '17.546.527-7', '33.580.136-5', '40.742.831-8', '37.053.630-7', '38.594.196-1', '46.260.533-4', '23.214.611-1', '10.009.245-7', '47.745.206-1', '24.585.794-1', '10.796.751-0', '43.644.520-7', '20.146.913-3', '25.377.871-2', '43.966.062-2', '29.205.879-2', '47.215.273-7', '26.346.604-8', '49.549.517-7', '29.674.463-3', '43.798.473-4', '12.460.250-2', '47.717.518-1', '13.296.719-4', '15.404.820-3', '27.073.901-4', '48.935.929-2', '50.300.305-0', '31.389.700-1', '50.707.122-0', '44.549.837-7', '37.310.705-5', '13.077.429-7', '44.257.736-9', '41.822.384-1', '40.724.763-4', '32.270.983-0', '30.464.005-0', '44.229.314-8', '34.811.263-4', '17.272.474-0', '47.251.702-8', '24.174.899-9', '13.562.556-7', '14.573.078-5', '25.942.796-2', '11.084.358-7', '20.659.248-6', '23.761.343-8', '34.392.103-0', '36.043.240-2', '50.226.885-2', '32.957.451-6', '47.555.585-5', '38.677.320-8', '12.372.518-5', '22.657.389-8', '18.176.993-1', '50.886.078-7', '32.194.114-7', '15.997.456-2', '16.902.021-6', '16.833.623-6', '36.754.736-3', '42.309.115-3', '48.251.869-8', '29.057.068-2', '16.617.684-9', '39.107.594-9', '37.039.688-1', '48.230.704-3', '49.538.400-8', '16.616.052-0', '31.801.343-5', '30.451.644-2', '12.369.474-7', '29.942.291-4', '31.416.155-7', '20.588.568-8', '18.231.904-0', '30.335.783-6', '18.885.313-3', '20.588.128-2', '14.218.138-9', '36.423.982-7', '47.703.768-9']
cpfs = ['681.791.390-00', '777.779.320-98', '854.522.810-45', '890.441.630-27', '084.141.220-00', '182.736.320-75', '836.783.990-02', '892.397.990-28', '646.800.490-06', '246.947.740-93', '781.342.460-36', '364.641.670-06', '287.796.320-92', '256.158.490-96', '072.214.440-72', '382.588.600-03', '298.156.960-09', '614.919.580-57', '890.158.430-15', '269.684.640-55', '300.333.970-23', '976.133.800-25', '462.911.280-47', '135.530.900-06', '928.307.550-14', '623.804.180-39', '987.539.400-91', '437.817.700-52', '452.773.980-82', '639.731.380-87', '190.313.130-84', '553.564.620-26', '301.185.090-98', '682.379.470-45', '319.215.310-53', '551.730.320-06', '640.990.450-99', '643.194.910-25', '124.253.230-74', '397.293.720-20', '081.903.340-50', '337.528.940-59', '717.292.270-07', '847.445.650-93', '461.281.790-70', '111.992.830-37', '937.850.090-03', '688.809.270-43', '709.498.680-78', '770.180.630-00', '557.613.980-10', '294.738.290-00', '662.449.240-61', '891.732.550-50', '955.813.530-50', '767.842.690-98', '640.000.800-40', '652.133.490-56', '020.681.720-70', '009.155.290-76', '597.782.830-60', '332.134.430-00', '421.279.090-46', '497.045.500-73', '706.199.870-90', '383.921.000-34', '904.379.130-07', '535.974.180-58', '509.603.660-08', '242.121.470-09', '584.607.620-33', '397.046.610-53', '898.403.040-62', '978.317.890-30', '776.123.610-00', '042.981.590-51', '694.980.860-79', '845.299.630-63', '686.574.430-66', '909.540.780-20', '433.592.700-29', '463.920.620-84', '572.698.560-52', '730.764.000-70', '174.138.750-71', '237.407.860-43', '252.592.880-68', '461.090.350-41', '280.790.180-81', '338.114.990-33', '506.753.050-20', '997.047.740-44', '537.791.390-65', '114.792.780-41', '385.939.700-10', '620.542.750-85', '703.131.290-61', '437.761.340-54', '102.334.640-00', '016.652.170-14', '251.807.970-04', '552.697.250-02', '237.237.940-25', '288.851.580-64', '448.967.430-91', '957.641.060-60', '267.291.630-62', '289.487.240-29', '087.599.770-86', '855.020.970-80', '985.685.290-03', '446.141.620-88', '346.530.660-00', '126.164.510-34', '349.290.520-09', '113.659.670-42', '153.964.420-00', '678.764.020-03', '392.490.550-99', '314.744.150-40', '421.560.530-06', '603.785.420-36', '523.071.900-15', '080.184.030-90', '340.669.990-18', '872.187.680-08', '462.831.770-42', '093.149.260-28', '569.730.840-01', '463.987.420-01', '581.536.410-00', '502.477.700-00', '973.724.360-95', '819.505.610-54', '683.714.950-48', '238.314.000-75', '035.076.390-92', '493.666.530-49', '939.792.600-46', '767.442.060-44', '683.384.210-81', '487.166.670-00', '323.999.330-95', '656.718.920-74', '740.468.360-24', '908.191.690-48', '768.765.650-41', '890.416.880-56', '482.941.910-51', '168.226.530-74', '803.948.770-63', '080.917.850-89', '454.434.950-89', '798.190.390-45', '984.724.440-58', '429.587.210-51', '724.243.440-86', '061.973.900-27', '692.753.530-69', '622.405.340-52', '314.888.080-31', '372.114.140-73', '768.390.790-15', '166.238.860-87', '149.854.400-27', '453.338.330-01', '092.487.590-92', '191.048.420-22', '387.374.340-04', '325.703.960-36', '952.207.260-56', '570.120.970-94', '850.909.310-54', '323.874.800-90', '533.302.540-17', '337.091.650-91', '918.520.510-94', '980.438.850-28', '879.326.560-36', '381.099.310-73', '470.731.690-52', '513.581.020-07', '009.388.030-85', '772.197.310-43', '840.619.070-18', '354.829.420-07', '880.569.190-90', '247.490.880-30', '934.423.520-10', '302.548.900-67', '186.059.970-28', '698.237.760-16', '251.369.210-13', '995.947.670-73', '373.322.260-14', '533.566.660-96', '622.628.530-37', '773.592.730-48', '420.600.540-01', '706.012.010-60', '377.252.340-47', '593.558.360-74', '254.237.080-01', '841.139.090-08', '850.315.440-40', '072.744.390-90', '614.898.350-83', '179.497.810-09', '117.624.590-27', '795.685.240-45', '408.044.830-94', '863.113.720-02', '161.702.090-70', '981.202.490-54', '006.883.310-50', '211.318.550-43', '883.285.050-88', '538.930.150-19', '155.126.570-20', '134.432.770-20', '225.698.030-81', '653.044.770-90', '896.999.800-44', '863.128.270-70', '671.105.120-37', '609.751.000-41', '186.492.650-30', '976.930.610-02', '656.928.500-95', '733.980.940-66', '867.513.880-63', '204.160.620-75', '067.328.480-89', '100.426.990-02', '196.473.190-96', '488.759.930-76', '080.869.430-89', '880.060.820-57', '300.507.850-79', '922.053.620-03', '337.512.890-89', '444.935.300-57', '264.240.280-05', '531.613.260-24', '215.767.000-97', '229.798.710-28', '479.224.670-95', '953.054.040-02', '271.892.300-84', '334.084.280-93']

enderecos = load_ceps_from_json("enderecos.json")


print("Nomes de usuarios: \t", len(nome_usuario))
print("Nomes de atendentes: \t", len(nome_atendente))
print("RGs: \t\t\t", len(rgs))
print("CPFs: \t\t\t", len(cpfs))
print("Enderecos: \t\t", len(enderecos))

#(identification, nome_usuario, nome_atendente, endereco, refer)
#for i in range(143, 250):
#    file = f"roteiro_{i}"+".txt"
#    endereco = f"{ enderecos[i]["endereco"] }, {enderecos[i]["bairro"]} CEP {enderecos[i]["cep"]}" 
#    completion = generate_script(cpfs[i], nome_usuario[i], nome_atendente[i], endereco, "CPF")
#    script = str(completion.choices[0].message["content"])
#    with open(file, "w") as file:
#        file.write(script)

a=487

for i in range(183, 250):
    file = f"roteiro_{a}"+".txt"
    endereco = f"{enderecos[a]["endereco"]}, {enderecos[a]["bairro"]} CEP {enderecos[a]["cep"]}" 
    completion = generate_script(rgs[i], nome_usuario[a], nome_atendente[a], endereco, "CPF")
    script = str(completion.choices[0].message["content"])
    with open(file, "w") as file:
        file.write(script)
    a += 1