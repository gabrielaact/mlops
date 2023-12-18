# Multiclass Text Classification

## Summary

This project aims to perform multiclass text classification using the [MLFlow](https://mlflow.org/) and [Gradio](https://www.gradio.app/) tools. The trained model is based on this [Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/), which seeks to extract the prevailing emotion in a given tweet. The main goal in this project is to apply best practices in Machine Learning Operations (MLOps) learned during the course.

Tweet classification is divided into two categories:

- `Positive`: for tweets that express positive emotions.

- `Negative`: for tweets that express negative emotions.

### Fetch Data

The database was downloaded from the Kaggle source site, and data retrieval for the code was done using the _pandas_ library.
```bash
dataframe = pd.read_csv('training.1600000.processed.noemoticon.csv',
                encoding='latin1', header=None, names=['target', 'ids', 'date', 'flag', 'user', 'text'])
```

### Preprocessing

After acquiring the raw data, preprocessing is necessary to clean the dataset by removing unwanted rows that could affect model training. In this case, unnecessary columns and neutral tweets are removed. This step also involves word tokenization.

### Data Segregation

The cleaned data is used as input for the "Data Segregation" step, which divides the data into training and testing sets. The ratio is 80% for training and 20% for testing.

### Train

In this step, the training of a Word2Vec model is performed. Finally, the trained model is obtained.

## Results

The trained Word2Vec model yielded the following results:

[INSERIR RESULTADOS]

## How to execute 

1 - Open a terminal and run the MLFlow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

2 - In another terminal, run the `model.py` file

```bash
python model.py
```

The terminal will display two local links, one for viewing MLflow and another for viewing Gradio. 


## References 

- [Ivanovitch's Repository](https://github.com/ivanovitchm/mlops)
- [Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140/)