---

# Named Entity Recognition (NER)

Welcome to the Named Entity Recognition (NER) project! This project focuses on identifying and classifying named entities (such as people, organizations, locations, etc.) in text using a machine-learning model built with TensorFlow and Keras.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation and Usage](#installation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Named Entity Recognition (NER) is a crucial task in natural language processing (NLP) that involves identifying and classifying entities in text. This project provides a comprehensive solution for NER using deep learning techniques.

## Features

- Preprocessing of text data
- Vectorization of sentences
- Building and training a neural network model
- Evaluation of model performance
- Making predictions on new sentences

## Installation and Usage

To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Shreyash-Gaur/Named_Entity_Recognition.git
    cd Named_Entity_Recognition
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies.

4. You can run the Jupyter notebook to explore the code and understand the step-by-step 
   process of building and using the NER model. Here’s how to start the notebook:

    ```bash
    jupyter notebook NER.ipynb
    ```

## Model Training

### Step 1: Exploring the Data
We use a dataset from Kaggle, which is preprocessed. The dataset consists of four columns: the sentence number, the word, the part of speech of the word, and the tags. A few tags we might expect to see are:
- `geo`: geographical entity
- `org`: organization
- `per`: person
- `gpe`: geopolitical entity
- `tim`: time indicator
- `art`: artifact
- `eve`: event
- `nat`: natural phenomenon
- `O`: filler word

### Step 2: Data Preprocessing

The first step involves preprocessing the text data to prepare it for model training. This includes tokenizing sentences and creating a mapping of entities to numerical labels. Tokenizing sentences helps in breaking down the text into individual words or tokens, which can then be assigned corresponding labels indicating the entity type (e.g., person, organization, location).

### Step 3: Vectorization

Next, the sentences are vectorized using a pre-trained embedding model. Vectorization transforms the text data into numerical form, which is required for feeding into the neural network. This process involves converting words or tokens into dense vectors of fixed size, capturing semantic information about the words.

- #### Encoding the sentences
We use `tf.keras.layers.TextVectorization` to transform the sentences into integers, enabling them to be fed into the model. The TextVectorization layer splits the sentences into tokens and maps each token to a unique integer.

- #### Encoding the labels
The labels are encoded using a custom function that maps each label to a unique integer. Padding is added to ensure that the labels have the same length as their corresponding sentences.

- #### Padding the labels
Padding ensures that all sequences (both sentences and labels) have the same length, which is necessary for batch processing in the model.

### Step 4: Building the Model

A neural network model is built using TensorFlow and Keras. The model architecture includes an embedding layer, LSTM layers, and a dense output layer. The embedding layer converts input tokens into dense vectors, the LSTM layers capture sequential dependencies in the data, and the dense layer outputs the final entity predictions.

### Step 5: Training

The model is trained on the preprocessed and vectorized data. During training, the model learns to associate input text sequences with their corresponding entity labels. The training process involves adjusting the model's parameters to minimize the prediction error on the training data, typically using an optimization algorithm like Adam.

## Evaluation

After training, the model achieves around 95.6% accuracy on the test set, indicating a high level of performance in identifying and classifying named entities.

## Prediction

You can use the trained model to make predictions on new sentences. This involves passing a new sentence through the model to obtain predicted entity labels for each token. The model’s output indicates the identified entities in the sentence, allowing you to extract structured information from unstructured text data.

## Contributing

We welcome contributions to this project! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---