import pandas as pd

# Importa o arquivo CSV com dados de vendas de videogames
# caminho absoluto usado para evitar problemas com diretório atual
df = pd.read_csv(r"D:\02_Machine-Learning\Mosh_ PythonMachineLearningTutorial(Data Science)\dataset\archive\vgsales.csv")

# Mostra a forma do DataFrame: (número de linhas, número de colunas)
print(df.shape)

# Exibe as primeiras 5 linhas do DataFrame (visão rápida dos dados)
print(df.head())

# Mostra estatísticas descritivas das colunas numéricas (contagem, média, desvio, etc.)
print(df.describe())

# Imprime os valores do DataFrame como um array NumPy (sem rótulos de coluna)
print(df.values)