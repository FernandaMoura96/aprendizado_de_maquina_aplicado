#%%
import pandas as pd

df = pd.read_excel('C:/Users/nanda/OneDrive/Desktop/machine/data/dados_cerveja_nota.xlsx')
df.head()


# %%
from sklearn import linear_model

X = df[['cerveja']] #matriz
y = df['nota'] # vetor

#Aprendizado de máquina
reg = linear_model.LinearRegression(fit_intercept=True)

reg.fit(X, y)

# %%
#Buscando e exibindo coeficientes

a, b =   reg.intercept_ , reg.coef_[0]
print(a,b )
# %%
predict = reg.predict(X.drop_duplicates())

# %%
import matplotlib.pyplot as plt 

plt.plot(X['cerveja'], y , 'o')
plt.grid(True)
plt.title("Relação cerveja VS Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

# adiconando reta
plt.plot(X.drop_duplicates()['cerveja'], predict)

plt.legend(["Observado", f'y = {a:.3f} + {b:.3f}*x'])
plt.show()
# %%
