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
features = df_train.columns[2:-1]
target = 'flagChurn'

#definindo x e y 

X , y = df_train[features], df_train[target]
# %%
from sklearn import model_selection 

X_train, X_test, y_train,y_test = model_selection.train_test_split(X,y,random_state=42,
                                                                   test_size= 0.2,
                                                                   stratify=y 
                                                                   #tamanho do teste 
                                                                   # #vai variar de acordo com o tam da base
)
#%%
print("Target train ", y_train.mean())
print("Target ttest ", y_test.mean())
# %%
