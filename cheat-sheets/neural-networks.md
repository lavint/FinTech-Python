# Neural Networks



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

## *Preprocess data*

Before using a neural network, it is crucial to normalize or standardize the data because neural networks typically perform better when each of the input features are on the same scale - Scikit-Learn's `MinMaxScaler` or `StandardScaler`

```
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
number_inputs = 2
number_hidden_nodes = 1

NN.add(Dense(input_dim=number_inputs,
             units=number_hidden_nodes, 
             activation="relu" 
             ))


# Create output layer
number_classes = 1

NN.add(Dense(units=number_classes, activation="sigmoid"))


# Check summary
NN.summary()

```

* The Dense() class is used to add layers to the neural networks
* The activation parameter defines the activation function that is used to process the values of the input features as they are passed to the first hidden layer

<br>


## *Compile the model*

Once the structure of the model is defined, it is compiled using a loss function and optimizer
```
NN.compile(loss="binary_crossentropy", 
           optimizer="adam", 
           metrics=["accuracy"])
```

* Optimizers are algorithms that shape and mold a neural network while it's trained to its most accurate possible form by updating the model in response to the output of the loss function



<br>

## *Fit the model*


After the model is compiled, it is trained with the data
```
model = NN.fit(X_train_scaled, y_train, epochs=80)
```

* Training consists of using the optimizer and loss function to update weights during each iteration of your training cycle. 

* This training uses 80 epochs (iterations)

* After each epoch, the results of the loss function and the accuracy are displayed


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
```
# Make prediction
predictions = NN.predict_classes(new_X)

results = pd.DataFrame({"predictions": predictions.ravel(), "actual": new_y})
```

