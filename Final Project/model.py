# Importar bibliotecas
import gradio as gr
import nltk
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# nltk.download('punkt')

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(experiment_name='Sentiment_Classification')
tags = {
        "Projeto": "Test MLflow",
        "team": "Data Science",
        "dataset": "sentiment"
       }

print("Iniciando o treinamento do modelo...")
# Carregar dados
dataframe = pd.read_csv('training.1600000.processed.noemoticon.csv',
                encoding='latin1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    
# Pré-processamento de dados
df = dataframe.copy()

# Remover colunas desnecessárias
df = df[['target', 'text']]

# Remover tweets neutros
df = df[df['target'] != 2]

# Mapear rótulos para classes (0 = negative, 1 = positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# Embaralhar os dados
df = df.sample(frac=1).reset_index(drop=True)

# Dividir dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
df['text'], df['target'], test_size=0.2, random_state=42)

with mlflow.start_run(run_name='SentimentClassifier'):        
    #Criação do modelo
    # Tokenizar as palavras
    tokenized_text = [word_tokenize(text.lower()) for text in X_train]

    # Treinar um modelo Word2Vec
    word2vec_model = Word2Vec(sentences=tokenized_text,
                                vector_size=100, window=5, min_count=1, workers=4)
    # Representar textos usando embeddings
    def text_to_embedding(text, model):
        tokens = word_tokenize(text.lower())
        embeddings = [model.wv[word] for word in tokens if word in model.wv]
        if not embeddings:
            return np.zeros(model.vector_size)
        return np.mean(embeddings, axis=0)

    X_train_embeddings = np.vstack([text_to_embedding(text, word2vec_model) for text in X_train])
    X_test_embeddings = np.vstack([text_to_embedding(text, word2vec_model) for text in X_test])
        
    # Treinar um modelo de classificação
    model = LogisticRegression()
    model.fit(X_train_embeddings, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test_embeddings)

    # Avaliar métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
        
    # Logar métricas
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1_score', f1)
    mlflow.log_artifact(local_path='./confusion_matrix.png', artifact_path='confusion_matrix')
    mlflow.set_tags(tags)

    # Logar parâmetros
    mlflow.log_param('vector_size', 100)
    mlflow.log_param('window', 5)
    mlflow.log_param('min_count', 1)
    mlflow.log_param('workers', 4)

    # Logar modelo
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact(local_path='./model.py', artifact_path='code')

def predict_emotion(text):
    tokenized_text = word_tokenize(text.lower())
    text_embedding = text_to_embedding(text, word2vec_model)
    

    prediction = model.predict([text_embedding])[0]
    

    emotion_label = "Positive" if prediction == 1 else "Negative"
    
    return emotion_label


iface = gr.Interface(
    fn=predict_emotion,
    inputs="text",
    outputs="text",
    live=True
)

# Lançar a interface
iface.launch()