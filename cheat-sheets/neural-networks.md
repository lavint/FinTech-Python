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


# Saving the Neural Network Model

To use a neural net model in a production setting, we often need to save the model and have it predict outcomes on unseen data at a future date.

``

``
