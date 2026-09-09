#%%
import pandas as pd 
import matplotlib.pyplot  as plt 

df = pd.read_excel('C:/Users/nanda/OneDrive/Desktop/machine/data/dados_cerveja_nota.xlsx')

df

df['aprovado'] = (df['nota'] >5).astype(int)
df

# %%

plt.plot(df['cerveja'], df['aprovado'], 'o', color = 'blue')
plt.grid(True)
plt.title("Cerveja VS Aprovação")
plt.xlabel('Cervejas')
plt.ylabel('Aprovado')

# %%
from sklearn import linear_model

reg =linear_model.LogisticRegression(penalty=None, 
                                     fit_intercept= True
                                     )
reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_predict


# %%
