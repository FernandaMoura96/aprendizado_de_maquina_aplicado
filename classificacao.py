#%%

#Lendo arquivos

import pandas as pd 
import matplotlib.pyplot  as plt 

df = pd.read_excel('C:/Users/nanda/OneDrive/Desktop/machine/data/dados_cerveja_nota.xlsx')

df['aprovado'] = (df['nota'] >5).astype(int)
df

# %%
#Plotando Primeiro gráfico
plt.plot(df['cerveja'], df['aprovado'], 'o', color = 'blue')
plt.grid(True)
plt.title("Cerveja VS Aprovação")
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')

# %%
from sklearn import linear_model
from sklearn  import tree 
from sklearn  import naive_bayes 

#Regressão Logistica 

reg =linear_model.LogisticRegression(penalty=None, 
                                     fit_intercept= True
                                     )
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#Arvores de decisão passando por todos os níveis 

arvore_full= tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#Arvore de decisão com 'max_depth=2'

arvore_d2= tree.DecisionTreeClassifier(random_state=42,max_depth= 2 )
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#Naive Bayes 

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]


#plotando garfico completo
plt.plot(df['cerveja'], df['aprovado'], 'o', color = 'blue')
plt.grid(True)
plt.title("Cerveja VS Aprovação")
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')
plt.plot(df['cerveja'].drop_duplicates(), reg_predict,  color ='tomato')
plt.plot(df['cerveja'].drop_duplicates(), reg_proba,  color ='red')
plt.hlines(0.5,xmin=1, xmax=9, linestyles= '--', colors= 'black')

#plotando arvore de decisão completa 

plt.plot(df['cerveja'].drop_duplicates(), nb_predict,  color ='green')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba,  color ='magenta')

# plotando arvore de decisaão D2 

plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict,  color ='blue')
plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_proba,  color ='black')

#plotando legenda 

plt.legend(["Observação", 'Reg Predict', 
            'Reg Proba',' ',
            "Naive Bayes Predict",
            "Naive Bayes Proba ",
            "Arvore D2 Predict",
            "Arvore D2 Proba "])
# %%
