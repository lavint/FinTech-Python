# Neural Networks

 
1. Preprocess data
2. Create Neural Network Structure
3. Compile the model
4. Fit the model
5. Evaluate the model
6. Make predictions using new data
7. Classification report
8. Compare models
9. Save the model


<br>


Every input data signal is weighted according to the relevance of each one under the context of the problem the perceptron was designed.


The perceptron took a weighted sum of the inputs


The bias is added as a particular input labeled as X0=1 with a negative weight


The activation function in the layers adds nonlinearity to the network and enables it to learn nonlinear relationships while the neural network is trained

It is a mathematical function with a characteristic S-shaped curve, also called the sigmoid curve.


Using an activation function, the output is a probability.

Instead of a yes/no decision, with an activation function, we get the probability of yes, similar to using logistic regression.

Using an activation function we can get a decision similar to real life - the final decision of watching a movie is a probability based on the input variables (what we know about the movie) and influenced by the bias (our personal preferences)


As the neural net learns, the loss function will decrease, and the decision boundaries will shift. 

The faster the loss function decreases, the more efficient the model; the lower the loss function becomes, the better it performs


We are going to use TensorFlow and Keras to build our Neural Networks.

TensorFlow is an end-to-end open-source platform for machine learning, that allows us to run our code across multiple platforms in a highly efficient way.


Keras is an abstraction layer on top of TensorFlow that makes it easier to build models.

There are two types of models in Keras:

1) Sequential model
    * Data flows from one layer to the next

2) Functional model
    * More customized


<br>

## *Preprocess data - transform categorical target `y` to numeric*

Before using a neural network, it is crucial to transform the categorical variable into (multiple) binary variables because neural networks cannot interpret non-numerical data

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv(file_path)
X = data.copy().drop(columns=["class"])
y = data["class"].copy()

# Create the OneHotEncoder instance
enc = OneHotEncoder()

# Reshape data
class_values = y.values.reshape(-1,1)[:3]

# Fit the OneHotEncoder
enc.fit(class_values)

# Fetch the categories identified by the OneHotEncoder
enc.categories_

# Transform categories
class_encoded = enc.transform(class_values).toarray()

# Create a DataFrame with the encoded class data
class_encoded_df = pd.DataFrame(
    class_encoded, 
    columns=["class1", "class2", "class3"]
)


# Use display() to view slices in dataframe
display(class_encoded_df.iloc[1:3])
display(class_encoded_df.iloc[10:12])
display(class_encoded_df.iloc[20:22])
```


## *Preprocess data - Normalize/Standardize `X` features*


**Train Test Split before Normalization**


Testing data points represent real-world data. Feature normalization (or data standardization) of the explanatory (or predictor) variables is a technique used to center and normalize the data by subtracting the mean and dividing by the variance. If you take the mean and variance of the whole dataset you'll be introducing future information into the training explanatory variables (i.e. the mean and variance).

Therefore, we should perform feature normalization over the training data. Then perform normalization on testing instances as well, but this time using the mean and variance of training explanatory variables. In this way, we can test and evaluate whether our model can generalize well to new, unseen data points.

Before using a neural network, it is crucial to normalize or standardize the data because neural networks typically perform better when each of the input features are on the same scale - Scikit-Learn's `MinMaxScaler` or `StandardScaler`


* Using standard scaler, each numerical variable has a mean of 0, and constant variance of 1

* Using MinMaxScaler, largest raw value for each column  has the value 1 and the smallest value for each column has the value 0


```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Create scaler instance
X_scaler = StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

<br>

## *Create Neural Network Structure*


```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Stores the architecture of the model in variable NN
NN = Sequential()

# Add the first layer
number_inputs = x_train_scaled.shape[1]
number_hidden_nodes = 100

NN.add(Dense(input_dim=number_inputs,
             units=number_hidden_nodes, 
             activation="relu" 
             ))


# Create output layer
number_classes = y_train["class"].value_counts()

NN.add(Dense(units=number_classes, activation="sigmoid"))


# Check summary
NN.summary()

```

* The Dense() class is used to add layers to the neural networks

* The activation parameter defines the activation function that is used to process the values of the input features as they are passed to the first hidden layer

* `Sigmoid` activation function is used for a binary classification model

<br>


## *Compile the model*

Once the structure of the model is defined, it is compiled using a loss function and optimizer

* `binary_crossentropy` is used for binary classification.

* `categorical_crossentropy` is used for classification models.

* `mean_squared_error` is used for regression models.

Check out more  [Kera Loss Function](https://keras.io/api/losses/)

* `adam` is a popular optimizer and is generally safe to use


```
NN.compile(loss="binary_crossentropy", 
           optimizer="adam", 
           metrics=["accuracy"])


# nn.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

# nn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.compile(
#     loss="binary_crossentropy",
#     optimizer="adam",
#     metrics=[
#         "accuracy",
#         tf.keras.metrics.TruePositives(name="tp"),
#         tf.keras.metrics.TrueNegatives(name="tn"),
#         tf.keras.metrics.FalsePositives(name="fp"),
#         tf.keras.metrics.FalseNegatives(name="fn"),
#         tf.keras.metrics.Precision(name="precision"),
#         tf.keras.metrics.Recall(name="recall"),
#         tf.keras.metrics.AUC(name="auc"),
#     ],
# )
```

* Optimizers are algorithms that shape and mold a neural network while it's trained to its most accurate possible form by updating the model in response to the output of the loss function



<br>

## *Fit the model*


After the model is compiled, it is trained with the data
```
model = NN.fit(X_train_scaled, 
               y_train, 
               epochs=30,
               shuffle=True,
               verbose=2
               )
```

* Training consists of using the optimizer and loss function to update weights during each iteration of your training cycle. 

* This training uses 80 epochs (iterations)

* After each epoch, the results of the loss function and the accuracy are displayed

<br>

An alternative to the `train_test_split`

```
model_1 = nn.fit(X, y, validation_split=0.3, epochs=200)
```

* Using the `validation_split` parameter, the model will set apart fraction of the training data, which it won't train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch


<br>

## *Evaluate the model*

After the training ends, the model is evaluated

1. Plot the loss function and accuracy

    ```
    # Create a DataFrame with the history dictionary
    df = pd.DataFrame(model.history, index=range(1, len(model.history["loss"]) + 1))

    # Plot the loss
    df.plot(y="loss")

    # Plot the accuracy
    df.plot(y="accuracy")
    ```

2. Use Test Data

    ```
    # Evaluate the model fit with linear dummy data
    model_loss, model_accuracy = NN.evaluate(X_test_scaled, y_test, verbose=2)
    ```


<br>

## *Make predictions using new data*

`predict` returns the scores of the regression

`predict_class` returns the class of the prediction

Imagine we are trying to predict if the picture is a dog or a cat (we have a classifier):

`predict` returns: 0.6 cat and 0.2 dog (for example).

`predict_class` returns cat

Now image we are trying to predict house prices (we have a regressor):

`predict` returns the predicted price

`predict_class` does not make sense here since we do not have a classifier
TL:DR: use predict_class for classifiers (outputs are labels) and use predict for regressions (outputs are non discrete)


```
# Make prediction
predictions = NN.predict_classes(new_X)

results = pd.DataFrame({"predictions": predictions.ravel(), "actual": new_y})


# Make predictions
predicted = model.predict(X_test_scaled)

# Convert the data back to the original representation
predicted = enc.inverse_transform(predicted).flatten().tolist()
results = pd.DataFrame({
    "Actual": y_test.activity.values,
    "Predicted": predicted
})

```

<br>

## *Classification report*

```
from sklearn.metrics import classification_report
print(classification_report(results.Actual, results.Predicted))
```

<br>

## *Compare models*

```
# Plot the loss function of the training results 

plt.plot(model_1.history["loss"])
plt.plot(model_2.history["loss"])
plt.show()

# Plot train vs test for model_1

plt.plot(model_1.history["loss"])
plt.plot(model_1.history["val_loss"])

# Plot train vs test for model_2
plt.plot(model_2.history["loss"])
plt.plot(model_2.history["val_loss"])

```

<br>

## *Improve model accuracy*

* Using more epochs for training 

* Adding more neurons  - too many can overfit the model

* Adding a second layer - this is part of deep learning

* Testing with different activation functions, especially when dealing with nonlinear data.



<br>

## *Save the model*
* We need to save both the model (using JSON) and the weights (using h5)

```
# Save model as JSON

nn_json = nn.to_json()

file_path = Path("../Resources/model.json")
with open(file_path, "w") as json_file:
    json_file.write(nn_json)


# Save weights

file_path = Path("../Resources/model.h5")
nn.save_weights(file_path)    

```

* Then we can load the saved model to make prediction
```
from tensorflow.keras.models import model_from_json

# load json and create model

file_path = Path("../Resources/model.json")
with open(file_path, "r") as json_file:
    model_json = json_file.read()

loaded_model = model_from_json(model_json)

# load weights into new model

file_path = Path("../Resources/model.h5")
loaded_model.load_weights("../Resources/model.h5")


# Make some predictions with the loaded model

df["prediction"] = loaded_model.predict(X)

```



<br>

Watch the [YouTube Video](https://www.youtube.com/watch?v=bfmFfD2RIcg) to learn more about Neural Network

<br>
<br>

# Deep Learning

Deep learning models are neural networks with more than one hidden layer

* They are much more effective than traditional machine learning approaches at discovering nonlinear relationships among data and thus are often the best-performing choice for complex or unstructured data like images, text, and voice

* The advantages of adding layers lie in the fact that each additional layer of neurons makes it possible to model more complex relationships and concepts

 * Imagine we are trying to classify whether a picture contains a cat. Conceptually, the first step may involve checking whether there exists some animal in the picture. Then, the model may detect the presence of paws etc. This breaking down of the problem continues until we reach the raw input of the model, which are the individual pixels in the picture. If this problem is correctly specified, each conceptual layer would need its own layer of neurons.

 * Adding layers does not always guarantee better performance - some layers can be redundant if the problem is not complex enough to warrant them. Also, overfitting may occur (when train accuracy is far higher than test accuracy)
 
 * **There is no easy analytical way of getting the number of layers we should use, the only solution to specifying the "correct" number of layers is to use trial and error**

<br>


 <br>
 <br>


# More on model evaluation

1. Receiver Operating Characteristic (ROC) Curve 
2. Area Under Curve (AUC) 

* They use the values from the confusion matrix to check and visualize the performance of a classification model


    True Positive Rate = TP / (TP + FN)

    False Positive Rate = FP / (FP + TN)

    The value of AUC ranges from 0 to 1

    * AUC = 0 means that the model predictions are 100% wrong

    * AUC = 1 means that model predictions are 100% correct
    
    * AUC = 0.50 means that the model is unable to distinguish between positive and negative classes 

* AUC measures the quality of the model's predictions regardless of the threshold

```
# Imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Creating training, validation, and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=1)


number_input_features = 10
hidden_nodes_layer1 = 15
hidden_nodes_layer2 = 5

# Define the LSTM RNN model
model = Sequential()

# Layer 1
model.add(
    Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Layer 2
model.add(Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
model.add(Dense(1, activation="sigmoid"))


# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

# Summarize the model
model.summary()

# Training the model
batch_size = 500
epochs = 20
training_history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
)
```

* The `validation_data` parameter specifies a dataset that is used to validate the model's performance along the training process, excluding the validation data sample as training data

* All the metrics are calculated on each epoch for the training and validation data

* The validation metrics have the val_ prefix

<br>


```
# Plotting loss
loss_df = pd.DataFrame(
    {
        "Epoch": range(1, epochs + 1),
        "Train": training_history.history["loss"],
        "Val": training_history.history["val_loss"],
    }
)
loss_df.set_index("Epoch", inplace=True)
loss_df.plot(title="Loss")


# Plotting accuracy
accuracy_df = pd.DataFrame(
    {
        "Epoch": range(1, epochs + 1),
        "Train": training_history.history["accuracy"],
        "Val": training_history.history["val_accuracy"],
    }
)
accuracy_df.set_index("Epoch", inplace=True)
accuracy_df.plot(title="Accuracy")



# Plotting AUC
auc_df = pd.DataFrame(
    {
        "Epoch": range(1, epochs + 1),
        "Train": training_history.history["auc"],
        "Val": training_history.history["val_auc"],
    }
)
auc_df.set_index("Epoch", inplace=True)
auc_df.plot(title="AUC")
```

* The metrics results of the training process are stored in the `history` dictionary of the `training_history` object



```
# Import roc_curve and auc metrics
from sklearn.metrics import roc_curve, auc

# Making predictions to feed the roc_curve module
train_predictions = model.predict(X_train, batch_size=1000)
test_predictions = model.predict(X_test, batch_size=1000)


# Calculate the ROC curve and AUC for the training set
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_predictions)
auc_train = auc(fpr_train, tpr_train)
auc_train = round(auc_train, 4)

# Calculate the ROC curve and AUC for the testing set
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, test_predictions)
auc_test = auc(fpr_test, tpr_test)
auc_test = round(auc_test, 4)


# Create a DataFrame with the fpr and tpr results
roc_df_train = pd.DataFrame({"FPR Train": fpr_train, "TPR Train": tpr_train,})

roc_df_test = pd.DataFrame({"FPR Test": fpr_test, "TPR Test": tpr_test,})


# Plotting the ROC Curves
roc_df_train.plot(
    x="FPR Train",
    y="TPR Train",
    xlim=([-0.05, 1.05]),
    title=f"Train ROC Curve (AUC={auc_train})",
)

roc_df_test.plot(
    x="FPR Test",
    y="TPR Test",
    color="red",
    style="--",
    xlim=([-0.05, 1.05]),
    title=f"Test ROC Curve (AUC={auc_test})",
)


# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

# Define metrics dictionary
metrics = {k: v for k, v in zip(model.metrics_names, scores)}

# Display evaluation metrics results
display(metrics)


# Define the confusion matrix data
cm_df = pd.DataFrame(
    {
        "Positive (1)": [f"TP={metrics['tp']}", f"FP={metrics['fn']}"],
        "Negative (0)": [f"FN={metrics['fp']}", f"TN={metrics['tn']}"],
    },
    index=["Positive(1)", "Negative(0)"],
)
cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

# Import
from sklearn.metrics import classification_report

# Predict classes using testing data
y_predict_classes = model.predict_classes(X_test, batch_size=1000)

# Display classification report
print(classification_report(y_predict_classes, y_test))

```
 
 <br>
 <br>


# ANNs v.s. RNNs v.s. LSTM-RNNs

**Artificial Neural Networks (ANNs)**

 * Do not have a memory mechanism

* Can be used to identify the type of car from a still image

* Cannot predict the direction of a car in movement because we don't know where the cas has been from a still image

<br>

**Recurrent Neural Networks (RNNs)**

* Have a sequential memory mechanism

* Combine the past knowledge with new inputs to make decisions

* Are suitable for sequential pattern recognition

* Can predict the direction of a car in movement because we know where the cas has been from a sequential history

* Have a feedback loop that allows information to flow from one step to the next along the sequence

* The feedback loop allows us to save the position of the car from one step to the next one as long as we have sequence data about the car's location

* However, RNNs have short-term memory, meaning they only remember the most recent few steps of a sequence, and this is resolved by using Long-Short-Term Memory Recurrent Neural Networks

* Suitable for:
    * Natural Language Processing

    * DNA sequences

    * Time series data

    * Music composition

<br>

**Long-Short-Term Memory Recurrent Neural Networks (LSTM-RNN)**

* Work like an original RNN but selectively decide which types of longer-term events are worth remembering and which are OK to forget

* Are capable of learning long-term dependencies using a mechanism called gates

Check out [Stanford Neural Network Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks) for more information

<br>
<br>

## *Data Preprocessing*

```
# Imports 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create X and y vectors
X = reviews_df["full_review_text"].values
y = reviews_df["sentiment"].values

# Create a Tokenizer instance and make all words to lower case
tokenizer = Tokenizer(lower=True)

# Encode each word with a unique word index
tokenizer.fit_on_texts(X)

# Transform the text data to numerical sequences
X_seq = tokenizer.texts_to_sequences(X)

# Padding sequences
X_pad = pad_sequences(X_seq, maxlen=140, padding="post")
```

* The RNN model requires that all the values of the `X` vector have the same length

* The `pad_sequences` method will ensure that all integer encoded reviews have the same size. 

* Each entry in `X` will be shortened to `100` integers, or pad with `0's` in case it's shorter.

```
# Create training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, random_state=78)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
```

<br>

## *Create LSTM RNN model architecture*

```
# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
```

```
# Model set-up
vocabulary_size = len(tokenizer.word_counts.keys()) + 1
max_words = 140
embedding_size = 64   # delivers the best result

# Define the LSTM RNN model
model = Sequential()

# Layer 1
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))

# Layer 2
model.add(LSTM(units=280))

# Output layer
model.add(Dense(1, activation="sigmoid"))

```
* Embedding layer: it processes the integer-encoded sequence of each review comment to create a dense vector representation that will be used by the LSTM layer

* LSTM layer: it transforms the dense vector into a single vector that contains information about the entire sequence that will be used by the activation function in the Dense layer to score the sentiment

* Dense layer: Use a sigmoid activation function to predict the probability of a review being positive


* Adding more LSTM layers and input units may lead to better results

* The `embedding_size` parameter specifies how many dimensions will be used to represent each word

* As a rule-of-thumb, a multiple of eight could be used


<br>


## *Compile the Model*
```
# Compile the model for binary classification
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
     metrics=[
        "accuracy",
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

# Summarize the model
model.summary()
```

<br>


## *Train the Model*
```
# Training the model
batch_size = 1000
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=batch_size,
    verbose=1,
)
```


<br>


## *Make Predictions*
```
# Make sentiment predictions
predicted = model.predict_classes(X_test, batch_size=1000)

# Create a DataFrame of Actual and Predicted values
sentiments = pd.DataFrame({"Text": X, "Actual": y_test[:10], "Predicted": predicted.ravel()})
```



<br>


## *Evaluate the Model*

Accuracy
```
from sklearn.metrics import accuracy_score
print("RNN LSTM Accuracy %.2f" % (accuracy_score(y_test, predicted)))
```


Confusion Matrix
```
from sklearn.metrics import confusion_matrix
tn_rnn, fp_rnn, fn_rnn, tp_rnn = confusion_matrix(y_test, predicted).ravel()

# Create a dataframe
cm_rnn_df = pd.DataFrame(
    {
        "Positive(1)": [f"TP={tp_rnn}", f"FP={fp_rnn}"],
        "Negative(0)": [f"FN={fn_rnn}", f"TN={tn_rnn}"],
    },
    index=["Positive(1)", "Negative(0)"],
)
cm_rnn_df.index.name = "Actual"
cm_rnn_df.columns.name = "Predicted"
print("Confusion Matrix from the RNN LSTM Model")
display(cm_rnn_df)
```


Classification Report
```
from sklearn.metrics import classification_report
print(classification_report(predicted, y_test))
```



Plotting the ROC Curve
```
from sklearn.metrics import roc_curve, auc

# Making predictions to feed the roc_curve module
test_predictions_rnn = model.predict(X_test, batch_size=1000)

# Data for ROC Curve - RNN LSTM Model
fpr_test_rnn, tpr_test_rnn, thresholds_test_rnn = roc_curve(y_test, test_predictions_rnn)

# AUC for the RNN LSTM Model
auc_test_rnn = auc(fpr_test_rnn, tpr_test_rnn)
auc_test_rnn = round(auc_test_rnn, 4)

# Dataframe to plot ROC Curve for the RNN LSTM model
roc_df_test_rnn = pd.DataFrame({"FPR Test": fpr_test_rnn, "TPR Test": tpr_test_rnn,})


roc_df_test_rnn.plot(
    x="FPR Test",
    y="TPR Test",
    color="blue",
    style="--",
    xlim=([-0.05, 1.05]),
    title=f"Test ROC Curve (AUC={auc_test_rnn})",
)
```