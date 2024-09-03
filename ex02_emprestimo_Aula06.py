# -*- coding: utf-8 -*-
"""

GERADOR DE 'DATASET' ARQUIVO NO FORMATO CSV:

colunas separadas por ';' e separador de cadas decimais ','
"""

'''Importação das Bibliotecas:
pandas para manipulação de dados.
numpy para gerar dados aleatórios.
files para manipular arquivos para download
'''

import pandas as pd
import numpy as np
from google.colab import files

# Montar o Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configurações para gerar dados aleatórios iguais
np.random.seed(0)  # Para reprodutibilidade

# Gerando dados aleatórios para as colunas
renda = np.random.uniform(5.0, 8.0, size=100)  # Valores entre 5 e 8
dividas = np.random.uniform(1.0, 3.0, size=100)  # Valores entre 1.0 e 3.0
pontuacao_credito = np.random.randint(60, 90, size=100)  # Valores entre 60 e 90
taxa_juros = np.random.choice([3, 5, 7, 10], size=100)  # Taxas de juros possíveis

# Criando o DataFrame
df = pd.DataFrame({
    'Renda': renda,
    'Dividas': dividas,
    'Pontuacao_Credito': pontuacao_credito,
    'Taxa_Juros': taxa_juros
})

# Arredondando os valores das colunas numéricas para 2 casas decimais
df['Renda'] = df['Renda'].round(2)
df['Dividas'] = df['Dividas'].round(2)

#---------------------------------------------------------
#CONFIGURACAO PARA BAIXAR O ARQUIVO DIRETO NO PC
#Salvando o DataFrame em um arquivo CSV
csv_file = 'dados_credito.csv'
df.to_csv(csv_file, sep=';', decimal=',', index=False)


#---------------------------------------------------------
#CONFIGURACAO PARA SALVAR DIRETO NA PASTA DO GOOGLE DRIVE
# Caminho completo para salvar o arquivo no Google Drive
file_path = '/content/drive/MyDrive/COLABS/SISTEMAS_APOIO_E_DECISAO/Ex02_Emprestimo/dados_credito.csv'

# Salvando o DataFrame em um arquivo CSV no Google Drive
df.to_csv(file_path, sep=';', decimal=',', index=False)
#---------------------------------------------------------


print(f"\nArquivo CSV salvo em: {file_path}")

"""Exercício 2 : Análise de Crédito para Empréstimos

Descrição:
Uma instituição financeira deseja usar uma árvore de decisão para determinar a taxa de juros para empréstimos com base no perfil do cliente. O banco fornece dados históricos com informações sobre a renda, a quantidade de dívidas e a pontuação de crédito dos clientes. O objetivo é criar um modelo que classifique a taxa de juros que deve ser aplicada ao empréstimo.

Dados:
Renda: Renda mensal do cliente (em milhares de reais)
Dívidas: Quantidade total de dívidas do cliente (em milhares de reais)
Pontuação_Crédito: Pontuação de crédito do cliente (de 0 a 100)
Taxa_Juros: Taxa de juros aplicada (3%, 5%, 7%, 10%)

Exercício 2: Passos:

Criação do DataFrame:
 Crie um DataFrame com os dados fornecidos.

Separação das Variáveis:
Separe as variáveis independentes (Renda, Dívidas, Pontuação_Crédito) e a variável dependente (Taxa_Juros).

Treinamento do Modelo:
Treine uma árvore de decisão para classificar a taxa de juros.

Visualização da Árvore:
Plote a árvore de decisão.

Resposta para Novos Dados:
Faça previsões para novos clientes.

"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



# Dados fornecidos
data = {
    'Renda': [5.0, 7.0, 4.5, 6.0, 8.0],
    'Dividas': [1.0, 2.5, 1.5, 2.0, 3.0],
    'Pontuacao_Credito': [80, 70, 85, 60, 90],
    'Taxa_Juros': [3, 5, 5, 7, 10]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Separando variáveis independentes e dependentes
X = df[['Renda', 'Dividas', 'Pontuação_Credito']]
y = df['Taxa_Juros']

# Treinando o modelo de árvore de decisão
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X, y)

# Plotando a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=['Renda', 'Dividas', 'Pontuacao_Credito'], class_names=['3%', '5%', '7%', '10%'], filled=True) #filled=true para colorir os nós da árvore
plt.title('Árvore de Decisão para Taxa de Juros de Empréstimo')
plt.show()

# Novos dados para previsão
novos_dados = pd.DataFrame({
    'Renda': [6.5, 4.0, 7.5],
    'Dividas': [2.0, 1.0, 2.5],
    'Pontuação_Credito': [75, 85, 65]
})

# Fazendo previsões
previsoes = clf.predict(novos_dados)
print("Previsões para os novos dados de taxa de juros:")
print(previsoes)

import pandas as pd
import numpy as np
from google.colab import files

# Caminho do arquivo CSV no Google Drive
file_path = '/content/drive/MyDrive/COLABS/SISTEMAS_APOIO_E_DECISAO/Ex02_Emprestimo/dados_credito.csv'

# Importando dados do arquivo CSV
df_csv = pd.read_csv(file_path, sep=';', decimal=',')

# Exibindo os dados importados
print("Dados importados do arquivo CSV:")
print(df_csv.head())

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Separando variáveis independentes e dependentes dos dados importados
X_csv = df_csv[['Renda', 'Dividas', 'Pontuacao_Credito']]
y_csv = df_csv['Taxa_Juros']

# Treinando o modelo com os dados do arquivo CSV
clf_csv = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf_csv.fit(X_csv, y_csv)

# Novos dados para previsão
novos_dados2 = pd.DataFrame({
    'Renda': [6.5, 4.0, 7.5],
    'Dividas': [2.0, 1.0, 2.5],
    'Pontuacao_Credito': [75, 85, 65]
})

# Fazendo previsões com o modelo treinado com dados do CSV
previsoes = clf_csv.predict(novos_dados2)
print("Previsões para os novos dados de taxa de juros:")
print(previsoes)


# Plotando a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(clf_csv, feature_names=['Renda', 'Dividas', 'Pontuacao_Credito'], class_names=['3%', '5%', '7%', '10%'], filled=True) #filled=true para colorir os nós da árvore
plt.title('Árvore de Decisão para Taxa de Juros de Empréstimo')
plt.show()
