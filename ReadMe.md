# 🧠 Aprendizado de Máquina Aplicado

Repositório de estudos práticos em **Machine Learning**, desenvolvido durante o curso de Machine Learning do [Téo Me Why](https://www.youtube.com/@TeoMeWhy). Reúne implementações de algoritmos de regressão, classificação e árvores de decisão aplicados a datasets didáticos, com foco em construir intuição sobre como cada modelo aprende, generaliza e se comporta diante de overfitting.

## 🎯 Sobre o projeto

Este repositório documenta minha trajetória de transição de carreira — de 15 anos em operações comerciais na Ambev para análise e ciência de dados — através da prática hands-on dos principais algoritmos supervisionados de ML em Python.

Cada script explora um problema e um conceito específico:

| Script | Problema | Conceitos aplicados |
|---|---|---|
| [`regressao.py`](./regressao.py) | Prever a nota de uma cerveja a partir do consumo | Regressão Linear, Árvore de Decisão (Regressor), comparação de overfitting entre modelo full e com `max_depth` limitado |
| [`classificacao.py`](./classificacao.py) | Prever aprovação com base no consumo de cerveja | Regressão Logística, Árvore de Decisão, Naive Bayes, comparação de fronteiras de decisão e probabilidades |
| [`frutas.py`](./frutas.py) | Classificar frutas por características físicas | Árvore de Decisão (Classifier), predição de classes e probabilidades, visualização da árvore |
| [`star_wars.py`](./star_wars.py) | Classificar clones como aptos/defeituosos | Árvore de Decisão com dataset mais complexo (múltiplas features categóricas e numéricas), encoding manual de variáveis ordinais |
| [`metricas.py`](./metricas.py) | Preparar dados de uma pesquisa da comunidade para análise | Engenharia de features: encoding de variáveis binárias e categóricas (`get_dummies`), estruturação de dataset analítico |

## 🛠️ Tecnologias

- **Python 3**
- **pandas** — manipulação e preparação de dados
- **scikit-learn** — modelos de regressão, classificação e árvores de decisão (`linear_model`, `tree`, `naive_bayes`)
- **matplotlib** — visualização de dados e das árvores de decisão treinadas

## 📁 Estrutura

```
aprendizado_de_maquina_aplicado/
├── data/               # datasets utilizados nos exercícios
├── regressao.py
├── classificacao.py
├── frutas.py
├── star_wars.py
├── metricas.py
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

3. Os scripts usam a sintaxe de células `# %%` (compatível com Jupyter/VS Code Interactive Window). Ajuste os caminhos dos arquivos em `data/` para o seu ambiente local e execute célula por célula, ou rode o arquivo completo:
   ```bash
   python regressao.py
   ```

> **Nota:** os caminhos de leitura de dados nos scripts originais apontam para uma pasta local (`C:/Users/...`). Ao clonar, atualize-os para o diretório `data/` do repositório antes de executar.

## 📌 Principais aprendizados

- Diferença prática entre regressão e classificação, e quando cada abordagem se aplica
- Como o `max_depth` de uma Árvore de Decisão impacta overfitting vs. generalização
- Comparação de modelos lineares (Regressão Linear/Logística) com modelos não-paramétricos (Árvores, Naive Bayes) sobre o mesmo problema
- Preparação de dados categóricos (encoding) como etapa essencial antes da modelagem

## 👩‍💻 Autora

**Fernanda Moura**
Em transição de carreira para Análise/Ciência de Dados, com background em operações comerciais.
[GitHub](https://github.com/FernandaMoura96)

---
*Projeto desenvolvido para fins de estudo, como parte do curso de Machine Learning do Téo Me Why.*