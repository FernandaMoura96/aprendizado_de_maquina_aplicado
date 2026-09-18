# 🧠 Aprendizado de Máquina Aplicado

Repositório de estudos práticos em **Machine Learning**, desenvolvido durante o curso de Machine Learning do [Téo Me Why](https://www.youtube.com/@TeoMeWhy). Reúne implementações de algoritmos de regressão, classificação e árvores de decisão aplicados a datasets didáticos, além de um exercício de modelo de **churn**, com foco em construir intuição sobre como cada modelo aprende, generaliza e se comporta diante de dificuldades de aprendizagem.

## 🎯 Sobre o projeto

Este repositório documenta a prática hands-on dos principais algoritmos supervisionados de ML em Python.

Cada script explora um problema e um conceito específico:

| Script | Problema | Conceitos aplicados |
|---|---|---|
| [`scripts/regressao.py`](./scripts/regressao.py) | Prever a nota de uma cerveja a partir do consumo | Regressão Linear, Árvore de Decisão (Regressor), comparação de overfitting entre modelo full e com `max_depth` limitado |
| [`scripts/classificacao.py`](./scripts/classificacao.py) | Prever aprovação com base no consumo de cerveja | Regressão Logística, Árvore de Decisão, Naive Bayes, comparação de fronteiras de decisão e probabilidades |
| [`scripts/cerveja.py`](./scripts/cerveja.py) | Classificar o estilo de uma cerveja (`classe`) a partir de temperatura, copo, espuma e cor | Árvore de Decisão (Classifier), encoding manual de variáveis categóricas, visualização da árvore |
| [`scripts/frutas.py`](./scripts/frutas.py) | Classificar frutas por características físicas | Árvore de Decisão (Classifier), predição de classes e probabilidades, visualização da árvore |
| [`scripts/star_wars.py`](./scripts/star_wars.py) | Classificar clones como aptos/defeituosos | Árvore de Decisão com dataset mais complexo (múltiplas features categóricas e numéricas), encoding manual de variáveis ordinais |
| [`scripts/metricas.py`](./scripts/metricas.py) | Prever se uma pessoa se considera feliz, a partir de uma pesquisa da comunidade | Engenharia de features (`get_dummies`), comparação de Árvore/Naive Bayes/Regressão Logística, métricas de avaliação (acurácia, precisão, recall, curva ROC, AUC), exportação do modelo com `pickle` |
| [`train.py`](./train.py) / [`train/train.py`](./train/train.py) | Preparar a base para um modelo de **churn** | Separação de amostra out-of-time (OOT), split treino/teste estratificado com `train_test_split` |

## 🛠️ Tecnologias

- **Python 3**
- **pandas** — manipulação e preparação de dados
- **scikit-learn** — modelos de regressão, classificação e árvores de decisão (`linear_model`, `tree`, `naive_bayes`, `model_selection`, `metrics`)
- **matplotlib** — visualização de dados e das árvores de decisão treinadas
- **openpyxl** / **pyarrow** — leitura de arquivos `.xlsx` e `.parquet`

## 📁 Estrutura

```
aprendizado_de_maquina_aplicado/
├── data/                       # datasets utilizados nos exercícios
│   ├── abt_churn.csv
│   ├── dados_cerveja.xlsx
│   ├── dados_cerveja_nota.xlsx
│   ├── dados_clones.parquet
│   ├── dados_comunidade.csv
│   ├── dados_frutas.xlsx
│   └── predict.csv
├── scripts/
│   ├── regressao.py
│   ├── classificacao.py
│   ├── cerveja.py
│   ├── frutas.py
│   ├── star_wars.py
│   ├── metricas.py
│   ├── churn.py
│   ├── model_feliz.pkl         # modelo exportado por metricas.py
│   └── predict.csv             # predições geradas por metricas.py
├── train/
│   └── train.py                # exploração inicial da ABT de churn
├── train.py                    # split treino/teste/OOT do modelo de churn
└── README.md
```

## 🚀 Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/FernandaMoura96/aprendizado_de_maquina_aplicado.git
   cd aprendizado_de_maquina_aplicado
   ```

2. Instale as dependências:
   ```bash
   pip install pandas scikit-learn matplotlib openpyxl pyarrow
   ```

3. Os scripts usam a sintaxe de células `# %%` (compatível com Jupyter/VS Code Interactive Window). Ajuste os caminhos dos arquivos em `data/` para o seu ambiente local (os caminhos originais apontam para uma pasta local, ex. `C:/Users/nanda/...`) e execute célula por célula, ou rode o arquivo completo:
   ```bash
   python scripts/regressao.py
   ```

> **Nota:** `scripts/churn.py` ainda está vazio — é o próximo passo planejado para consolidar o exercício de churn iniciado em `train.py` e `train/train.py`.

## 📌 Principais aprendizados

- Diferença prática entre regressão e classificação, e quando cada abordagem se aplica
- Como o `max_depth` de uma Árvore de Decisão impacta overfitting vs. generalização
- Comparação de modelos lineares (Regressão Linear/Logística) com modelos não paramétricos (Árvores, Naive Bayes) sobre o mesmo problema
- Preparação de dados categóricos (encoding) como etapa essencial antes da modelagem
- Avaliação de modelos de classificação com métricas além da acurácia (precisão, recall, ROC/AUC)
- Boas práticas de separação de dados em problemas com componente temporal (amostra out-of-time)

## 👩‍💻 Autora

**Fernanda Moura**
Em transição de carreira para Análise/Ciência de Dados, com background em operações comerciais.
[GitHub](https://github.com/FernandaMoura96)

---
*Projeto desenvolvido para fins de estudo, como parte do curso de Machine Learning do Téo Me Why.*