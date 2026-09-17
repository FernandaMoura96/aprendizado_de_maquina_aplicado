#%%
import pandas as pd 
from sklearn import tree
import matplotlib.pyplot as plt 

df = pd.read_parquet("C:/Users/nanda/OneDrive/Desktop/machine/data/dados_clones.parquet")
df
#%%
#entendendo quais as colunas do dataset 
print(df.columns)
# %%

features = ['p2o_master_id','Massa(em kilos)'
            ,'Estatura(cm)','Distância Ombro a ombro',
            'Tamanho do crânio','Tamanho dos pés',
            'Tempo de existência(em meses)',]
target = 'Status '

X = df[features]
y = df[target]
X = X.replace( {
    'Distância Ombro a ombro': {
        'Tipo 1': 1,
        'Tipo 2': 2,
        'Tipo 3': 3,
        'Tipo 4': 4,
        'Tipo 5': 5,
    },
    'Status ': {'Defeituoso': 0,
                 'Apto': 1,},
    'Tamanho do crânio': {
        'Tipo 1': 1,
        'Tipo 2': 2,
        'Tipo 3': 3,
        'Tipo 4': 4,
        'Tipo 5': 5,
    },
    'Tamanho dos pés': {
        'Tipo 1': 1,
        'Tipo 2': 2,
        'Tipo 3': 3,
        'Tipo 4': 4,
        'Tipo 5': 5,
    }
        
       })
# %%

model = tree.DecisionTreeClassifier(max_depth=4)

model.fit(X = X , y=y )

# %%


plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True 
               )
# %%
features =['Massa(em kilos)','Estatura(cm)']

df.groupby("Status ")[features].mean()
# %%
