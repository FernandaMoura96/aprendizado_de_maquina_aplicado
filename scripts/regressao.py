#%%
import pandas as pd


df = pd.read_excel('C:/Users/nanda/OneDrive/Desktop/machine/data/dados_cerveja_nota.xlsx')
df.head()


# %%
from sklearn import linear_model
from sklearn import tree


# Supondo que o DataFrame 'df' já esteja definido
X = df[['cerveja']]  # matriz
y = df['nota']     # vetor

# Aprendizado de máquina (Regressão Linear)
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

# Buscando e exibindo coeficientes
a, b = reg.intercept_, reg.coef_[0]
predict_reg = reg.predict(X.drop_duplicates())

# Árvore de Decisão Full(over fit)
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

#Árvore de decisão com parametro reestabelecido 

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth =2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
 # %%
import matplotlib.pyplot as plt 

plt.plot(X['cerveja'], y , 'o')
plt.grid(True)
plt.title("Relação cerveja VS Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

# adiconando reta
plt.plot(X.drop_duplicates()['cerveja'], predict_reg )
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full )
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2,color = "magenta" )

plt.legend(["Observado", f'y = {a:.3f} + {b:.3f}*x',
            'Árvore Full',
            'Árvore Depth =2', 
            ])
plt.show()
# %%
plt.figure(dpi=400)
tree.plot_tree(arvore_d2,
               feature_names=['cerveja'],
               filled= True)
# %%
                                                                                              