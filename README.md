# Rede Neural MLP para Classificação de Estrelas Pulsares

Este projeto apresenta a construção e o treinamento de uma Rede Neural **Multi-Layer Perceptron (MLP)** para a tarefa de classificação de estrelas pulsares. O grande diferencial desta implementação é que a rede neural foi desenvolvida **"do zero"**, utilizando apenas a biblioteca NumPy, sem o auxílio de frameworks de alto nível como TensorFlow ou Keras.

Além disso, o projeto inclui um **Algoritmo Genético** para otimizar os hiperparâmetros da rede, buscando a configuração que maximiza a performance do modelo.

## 🎯 Objetivo do Projeto

O objetivo é criar um classificador binário capaz de distinguir estrelas pulsares de outras fontes de ruído cósmico com base em 8 features contínuas extraídas de medições astronômicas. O desafio secundário é demonstrar a construção dos mecanismos internos de uma rede neural e como técnicas de meta-heurística podem ser aplicadas para sua otimização.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Bibliotecas Principais:** NumPy, Pandas
* **Visualização de Dados:** Matplotlib, Seaborn
* **Pré-processamento:** Scikit-learn (apenas para `MinMaxScaler` e `train_test_split`)
* **Métricas de Avaliação:** Scikit-learn

## 📊 Análise e Preparação dos Dados

O dataset utilizado foi o "Pulsar Star Prediction". A preparação dos dados consistiu nas seguintes etapas:

1.  **Leitura e Tratamento de Nulos:** Os dados foram carregados e os valores ausentes foram preenchidos utilizando a média da respectiva coluna.
    ```python
    dataset_train = pd.read_csv('pulsar_data_train.csv')
    dataset_test = pd.read_csv('pulsar_data_test.csv')
    dataset_train = dataset_train.fillna(dataset_train.mean())
    dataset_test = dataset_test.fillna(dataset_test.mean())
    ```

2.  **Análise de Correlação:** Uma matriz de correlação foi gerada para entender a relação entre as features.
    ![Matriz de Correlação](matriz_correlacao.png)
    *(**Nota:** Salve a imagem gerada pelo seu código como `matriz_correlacao.png` no repositório)*

3.  **Divisão e Normalização:** Os dados foram divididos em conjuntos de treino e teste, e as features foram normalizadas para a escala [0, 1] usando `MinMaxScaler` para garantir a estabilidade do treinamento da rede.
    ```python
    X = dataset_train.fillna(dataset_train.mean()).to_numpy()[:, 0:8]
    Y = dataset_train.fillna(dataset_train.mean()).to_numpy()[:, 8]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

## 🧠 Rede Neural MLP "From Scratch"

A classe `MultiLayerPerceptron` foi implementada para encapsular toda a lógica da rede, incluindo a inicialização de pesos, o feedforward, o cálculo da perda e o backpropagation.

### 1. Arquitetura
A rede possui a seguinte estrutura:
* Camada de Entrada (8 neurônios, correspondentes às features)
* Camada Oculta 1 (número de neurônios variável)
* Função de Ativação: Tangente Hiperbólica (`tanh`)
* Camada Oculta 2 (número de neurônios variável)
* Função de Ativação: Tangente Hiperbólica (`tanh`)
* Camada de Saída (1 neurônio)
* Função de Ativação: Sigmoide (para classificação binária)

A inicialização dos pesos é feita de forma a evitar o problema de "vanishing/exploding gradients", dividindo os valores aleatórios pela raiz quadrada do número de neurônios da camada anterior.

### 2. Mecanismos
* **Feedforward (`forward`):** Propaga os dados de entrada através das camadas, aplicando as funções de ativação, até gerar a predição na camada de saída.
    ```python
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = x.dot(self.W1) + self.B1
        self.f1 = np.tanh(self.z1)

        self.z2 = self.f1.dot(self.W2) + self.B2
        self.f2 = np.tanh(self.z2)

        z3 = self.f2.dot(self.W3) + self.B3
        self.output = 1 / (1 + np.exp(-z3))
        return self.output
    ```
* **Backpropagation (`backpropagation`):** Calcula o gradiente do erro em relação a cada peso e bias, propagando-o da camada de saída para a de entrada, e atualiza os parâmetros da rede usando o Gradiente Descendente.
    ```python
    def backpropagation(self, learning_rate: float) -> None:
        delta3 = self.output - self.y.reshape(-1, 1)
        dW3 = self.f2.T.dot(delta3) / self.x.shape[0]

        self.W1 -= learning_rate * dW1
    ```
* **Função de Perda (`loss`):** Utiliza a Entropia Cruzada Binária, ideal para problemas de classificação binária.

## 🧬 Otimização com Algoritmo Genético

Para encontrar a melhor combinação de hiperparâmetros, um Algoritmo Genético foi implementado.
* **Cromossomo:** Cada indivíduo da população é um "cromossomo" que representa uma configuração da rede, contendo: `[épocas, taxa_de_aprendizagem, neuronios_oculta1, neuronios_oculta2]`.
* **Função de Aptidão (Fitness):** A aptidão de cada cromossomo é avaliada pelo **F1-Score** obtido pela MLP com seus respectivos hiperparâmetros no conjunto de teste. O F1-Score foi escolhido por ser uma métrica robusta para datasets desbalanceados.
* **Evolução:** O algoritmo evolui a população por 20 gerações, utilizando:
    * **Seleção por Torneio:** Para escolher os pais.
    * **Crossover de Ponto Único:** Para gerar descendentes.
    * **Mutação:** Para introduzir diversidade e evitar mínimos locais.

## 📈 Resultados

Após o treinamento da MLP com os hiperparâmetros otimizados, o modelo alcançou as seguintes métricas no conjunto de teste:

* **Acurácia:** 0.98
* **Precisão:** 0.92
* **Recall:** 0.71
* **F1-Score:** 0.80
* **AUC-ROC:** 0.85

A matriz de confusão abaixo ilustra o desempenho do classificador.

## ✒️ Autor

**[Kauê Santos]**

* [LinkedIn](https://www.linkedin.com/in/kauê-santos-0a381b25a)
* [GitHub](https://github.com/KWSantos)