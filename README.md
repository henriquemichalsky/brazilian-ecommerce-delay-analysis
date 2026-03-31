# 🚚 Previsão de Atrasos em Entregas — E-commerce Olist

> 🚚 Análise exploratória e modelo preditivo (Random Forest) para identificar pedidos com risco de atraso em um e-commerce brasileiro — dataset público Olist | Python · SQL · scikit-learn

---

## 📌 Objetivo

Entender os padrões de atraso em entregas do e-commerce brasileiro e construir um modelo de Machine Learning capaz de prever, antes do despacho, se um pedido tem alto risco de não ser entregue no prazo estimado.

---

## 🗂️ Estrutura do Projeto
Projeto_2/
├── Projeto_2.ipynb # Notebook principal com toda a análise
├── images/
│ ├── grafico_atraso_estado.png
│ ├── grafico_atraso_mes.png
│ ├── grafico_atraso_distancia.png
│ ├── grafico_volume_pedidos.png
│ ├── grafico_receita_ticket.png
│ ├── grafico_matriz_confusao.png
│ ├── grafico_importancia_features.png
│ └── grafico_curva_roc.png
└── README.md

> ⚠️ **Os arquivos CSV não estão incluídos neste repositório por serem grandes demais.**
> Faça o download diretamente pelo Kaggle e coloque-os na raiz do projeto antes de executar o notebook:
> 👉 [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## 🛢️ Fonte dos Dados

- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle
- **Tabelas utilizadas:** `orders`, `customers`, `sellers`, `products`, `order_items`, `order_payments`, `order_reviews`, `geolocation`, `product_category_name_translation`
- **Armazenamento:** os CSVs foram carregados e persistidos em um banco **SQLite** para consultas via SQL

Após baixar, a pasta deve conter os seguintes arquivos:
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
product_category_name_translation.csv
---

## 🔧 Tecnologias e Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e análise de dados |
| `sqlite3` | Criação do banco e consultas SQL |
| `numpy` | Cálculo vetorizado (Haversine) |
| `matplotlib` / `seaborn` | Visualizações da EDA e do modelo |
| `scikit-learn` | Pipeline de ML (split, encoding, Random Forest, métricas) |

---

## 📊 Etapas do Projeto

### 1. Banco de Dados e Query Fato
- Todos os CSVs foram ingeridos no SQLite via `pandas.to_sql()`
- Uma query SQL com múltiplos `LEFT JOIN` criou a tabela fato com **119.143 linhas** e 21 colunas

### 2. Análise Exploratória (EDA)

**Variável alvo:**
- `atrasou = 1` se a data de entrega real ultrapassou a estimada; `0` caso contrário
- Taxa de atraso geral: **9,2%** | Atraso médio: **10,6 dias**

**Principais achados:**

| Dimensão | Insight |
|---|---|
| **Por Estado** | AL (24,6%), RR (21,2%) e MA (21,0%) têm as maiores taxas — reflexo da distância logística |
| **Por Mês** | Março (17,0%) e Novembro (14,7%) são os meses mais críticos — sazonalidade de verão e Black Friday |
| **Por Distância** | Pedidos acima de 2.000 km atrasam **2× mais** do que pedidos locais (13,7% vs 6,6%) |
| **Série Temporal** | Crescimento contínuo de volume de pedidos e receita ao longo de 2016–2018 |

### 3. Preparação dos Dados para ML

- **Features selecionadas:** `customer_state`, `seller_state`, `product_category_name_english`, `payment_type`, `payment_installments`, `price`, `freight_value`, `distancia_km`, `mes_compra`
- Distância calculada com a fórmula de **Haversine** (coordenadas geográficas da tabela `geolocation`)
- Nulos tratados: mediana para numéricos, `"desconhecido"` para categóricos
- Encoding com `LabelEncoder`
- Split **80/20** com `stratify=y` → 95.314 treino / 23.829 teste (proporção de atrasos preservada: 9,22%)

### 4. Modelo Preditivo — Random Forest

O algoritmo escolhido foi o **Random Forest** por ser robusto com dados tabulares mistos, não exigir normalização e fornecer a importância das features nativamente. O parâmetro `class_weight="balanced"` foi fundamental para lidar com o desbalanceamento de classes (91% no prazo vs 9% atrasado), evitando que o modelo simplesmente ignorasse a classe minoritária.

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",  # compensa o desbalanceamento de classes
    random_state=42,
    n_jobs=-1
)
```

A avaliação foi feita com **Classification Report**, **Matriz de Confusão** e **Curva ROC**, permitindo analisar o desempenho tanto na classe majoritária quanto nos atrasos reais.

---

## 📈 Resultados

| Métrica | Valor |
|---|---|
| **AUC-ROC** | **0,7738** |
| Accuracy | 0,84 |
| Precision (Atrasado) | 0,29 |
| Recall (Atrasado) | **0,51** |
| F1-Score (Atrasado) | 0,37 |

> O modelo identifica **51% dos atrasos reais** antes que aconteçam — um resultado expressivo para um dataset com apenas 9% de classe positiva.

**Features mais importantes (por importância Gini):**
1. `distancia_km` — principal fator preditivo
2. `customer_state` — localização do comprador
3. `mes_compra` — sazonalidade

---

## 💡 Conclusões

- Estados do **Nordeste** concentram os maiores riscos de atraso, diretamente relacionados à distância dos centros distribuidores
- **Fevereiro, Março e Novembro** são os meses mais críticos para a logística
- A **distância geográfica** entre vendedor e cliente é a variável com maior poder preditivo no modelo
- Com este modelo, um e-commerce poderia gerar **alertas antecipados de risco** para pedidos de alto risco antes mesmo do despacho, permitindo ações proativas (reforço de estoque regional, comunicação ao cliente, priorização logística)

---

## 👤 Autor

**Henrique Michalsky**  
Portfólio de Data Science & Analytics  
Vila Velha, Espírito Santo — Brasil
