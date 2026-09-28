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
feature_importances = (pd.Series(arvore.feature_importances_,
                                 index=X_train.columns)
                       .sort_values(ascending=False)
                       .reset_index()
                       )

feature_importances['acum.']=feature_importances[0].cumsum()
feature_importances[feature_importances['acum.'] < 0.96]

# %%
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index'].tolist())
# %%
#MODIFY
from feature_engine import discretisation , encoding

## DESCRETIZAR 
tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                             regression= False,
                                                             bin_output='bin_number',
                                                             cv=3 )

tree_discretisation.fit(X_train[best_features], y_train)
X_train_transform = tree_discretisation.transform(X_train[best_features])

#OneHot 
##X_train_transform = X_train_transform.astype(str)  
###Serve para modificar o tipo da variavel para que o encoding reconheça
onehot= encoding.OneHotEncoder(variables=best_features,ignore_format= True)
onehot.fit(X_train_transform,y_train)

X_train_transform= onehot.transform(X_train_transform)
X_train_transform 
# %%
#MODEL
from sklearn import linear_model
from sklearn import pipeline
 
reg = linear_model.LogisticRegression(penalty= None, random_state=42, max_iter=10000)
model_pipeline = pipeline.Pipeline(
    steps=[
        ('Descretizar', tree_discretisation),
        ('Onehot', onehot),
        ('Model', reg ),
    ]
)
#MlFlow 
import mlflow
from sklearn import metrics
 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("churn_exp")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train[best_features], y_train)

    # MÉTRICAS - Base Treino

    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:, 1]
    
    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train,y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)
    
    # MÉTRICAS - Base Teste

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:, 1]
    
    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test= metrics.roc_curve(y_test,y_test_proba)

    print("Acurácia Test:", acc_test)
    print("AUC Test:", auc_test)
    
    # MÉTRICAS - Base OOT
    
    oot_predict = model_pipeline.predict(oot[best_features])
    oot_proba = model_pipeline.predict_proba(oot[best_features])[:, 1]
    
    acc_oot = metrics.accuracy_score(oot[target], oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], oot_proba)
    roc_oot = metrics.roc_curve(oot[target],oot_proba)

    print("Acurácia OOT:", acc_oot)
    print("AUC OOT:", auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot,
    })
# %%
X_train[best_features]
# %%
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.plot([0, 1], [0, 1], '--', color='black')
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title("Curva ROC")
plt.legend([
    f"Treino: {100*auc_train:.2f}",
    f"Teste: {100*auc_test:.2f}",
    f"Out-of-Time: {100*auc_oot:.2f}",
])
# %%