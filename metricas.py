#%%

import pandas as pd

csv  = "C:/Users/nanda/OneDrive/Desktop/machine/data/dados_comunidade.csv"

df = pd.read_csv(csv)
df.head()
# %%
#todas as variaveis devem ser numéricas. Abaixo, essa modificação
df = df.replace({'Sim':1, 'Não':0})

num_vars = ['Curte games?','Curte futebol?',
            'Curte livros?','Curte jogos de tabuleiro?',
            'Curte jogos de fórmula 1?',
            'Curte jogos de MMA?','Idade']
 #usar get.dumies não é uma boa pratica, seria melhor usar hot encode.       
dummie_vars = [
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Área de Formação',
    'Estado que mora atualmente',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)', ]

df_analise = pd.get_dummies(df[dummie_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz'] = df_analise['Você se considera uma pessoa feliz'].copy()
df_analise

