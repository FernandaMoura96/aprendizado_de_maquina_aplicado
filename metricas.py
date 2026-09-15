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
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise


# %%
from sklearn import tree 

features =  df_analise.columns[:-1].tolist()

X = df_analise[features]
y= df_analise['pessoa feliz']

arvore = tree.DecisionTreeClassifier(random_state=42,
                                     min_samples_leaf=5,
                                     )

arvore.fit(X, y)

# %%

#Avaliando se a arvóre 'acertou a prdição'

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['pessoa feliz']]
df_predict['predict_arvore'] = arvore_predict
df_predict

# %%
# arvore demonstra 86,26% de acerto
#ACURÁCIA = MOSTRA O QUANTO ESTA CORRETO, MAS NÃO MOSTRA
#ONDE ESTA ACERTANDO E ONDE ESTA ERRANDO.
#PARA ISSO USA-SE A MATRIZ DE CONFUSÃO 

 
(df_predict['pessoa feliz'] == df_predict['predict_arvore']).mean()
# %%

#MATRIZ DE CONFUSÃO 

matriz = pd.crosstab(df_predict['pessoa feliz'], df_predict['predict_arvore'])
matriz 

# o zero representa o verdadeiro valor na base 
#%%
(df_predict['pessoa feliz'] == 1 ) .sum()

(df_predict['predict_arvore'] == 1 ) .sum()
# %%
# arquivo csv usado para validação dos dados no sheets 

df_predict.to_csv('predict.csv', sep = ';' , index= False)

# %%
 from sklearn import metrics

acc_arvore = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
acc_arvore

precisao_arvore = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
precisao_arvore

recall_arvore = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
roc = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['predict_arvore'])
roc



# %%
import matplotlib.pyplot as plt 

plt.plot(roc[0],roc[1])
# %%


#  (pegando a probabilidade da classe 1):
arvore_probs = arvore.predict_proba(X)[:, 1]

#cauculando  curva ROC com as probabilidades:
fpr, tpr, thresholds = metrics.roc_curve(df_predict['pessoa feliz'], arvore_probs)

# Plot
import matplotlib.pyplot as plt 
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label='Árvore de Decisão')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatório') # Linha de referência diagonal
plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
plt.title('Curva ROC')
plt.legend()
plt.show()
# %%
