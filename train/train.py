#%%
import pandas as pd

df = pd.read_csv("C:/Users/nanda/OneDrive/Desktop/machine/data/abt_churn.csv", sep= ',')
df.head()
# %%
# separando out of time 

oot = df[df['dtRef'] == df['dtRef'].max()]
oot

# %%
#Base de treino deve ser menor que a oot 

df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train['dtRef']
# %%
# para separa em treino e teste, é interessante definir 
#oque é variavel e oque é target. 

df_train.head()
features = df_train.columns[2:-2]
target = 'flagChurn'

#definindo x e y 

X , y = df_train[features], df_train[target]
# %%
#SAMPLE
from sklearn import model_selection 

X_train, X_test, y_train,y_test = model_selection.train_test_split(X,y,random_state=42,
                                                                   test_size= 0.2,
                                                                   stratify=y 
                                                                   #tamanho do teste 
                                                                   # #vai variar de acordo com o tam da base
)

print("Target train ", y_train.mean())
print("Target ttest ", y_test.mean())
# %%

#Explore Missing 

X_train.isna().sum().sort_values(ascending= False)


# %%

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean','median']).T
#aumentando a quantidade de linhas a sereme exibidas
#pd.set_option('display.max_rows', 500)
#print(sumario)
sumario

# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)
# %%
from sklearn import tree
import matplotlib.pyplot as plt 

arvore = tree.DecisionTreeClassifier(random_state=42,)
arvore.fit(X_train, y_train)

#usando a arvore para definir a importancia de cada variavel 
arvore.feature_importances_
# %%
pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# %%
