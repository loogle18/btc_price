# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
plt.style.use("bmh")

# Neural Network library
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the multivariate time sequence
    """
    
    # Creating a list for both variables
    X, y = [], []
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out
        
        # Breaking out of the loop if we have exceeded the dataset"s length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)
  
  
def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(16,5))
    plt.plot(history["val_loss"])
    plt.plot(history["loss"])
    plt.legend(["val_loss", "loss"])
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    
    plt.figure(figsize=(16,5))
    plt.plot(history["val_accuracy"])
    plt.plot(history["accuracy"])
    plt.legend(["val_accuracy", "accuracy"])
    plt.title("accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("mae")
    plt.show()
    
    
def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):
    """
    Creates a specified number of hidden layers for an RNN
    Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
    """
    
    # Creating the specified number of hidden layers with the specified number of nodes
    for x in range(1,n_layers+1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

        # Adds a Dropout layer after every Nth hidden layer (the "drop" variable)
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass
          
          
def validater(n_per_in, n_per_out):
    """
    Runs a "For" loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    
    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=df.index, columns=["close"])
    

    for i in range(n_per_in, len(df)-n_per_in, n_per_out):
        # Creating rolling intervals to predict off of
        x = df[-i - n_per_in:-i]

        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

        # Transforming values back to their normal prices
        yhat = close_scaler.inverse_transform(yhat)[0]
        print(yhat)
        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhat, 
                               index=pd.date_range(start=x.index[-1], 
                                                   periods=len(yhat), 
                                                   freq="B"),
                               columns=["close"])

        # Updating the predictions DF
        predictions.update(pred_df)

    return predictions


def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()
    # Adding a new column with the closing prices from the second DF
    df["close2"] = df2["close"]
    
    # Dropping the NaN values
    df.dropna(inplace=True)
    
    # Adding another column containing the difference between the two DFs" closing prices
    df["diff"] = df["close"] - df["close2"]
    
    # Squaring the difference and getting the mean
    rms = (df[["diff"]]**2).mean()
    
    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms))



# Loading in the Data
headers = [
    "date", "open", "high", "low", "close", "volume_btc", "volume_usd",
    "market_cap_all",
    "fng_value",
    "btc_dominance",
    "neutral",
    "positive",
    "negative"
]
df = pd.read_csv("data/training/final.csv", sep=",", names=headers, skiprows=1, low_memory=False)
# df = df[["date", "close"]]

## datetime conversion
df["date"] = pd.to_datetime(df["date"])

# Setting the index
df.set_index("date", inplace=True)
# print(df.index)
# print(df.columns)
# print(df["close"])
# print(df["fng_value"])
# print(df["btc_dominance"])
# raise BaseException

# Dropping any NaNs
df.dropna(inplace=True)

## Scaling

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()

close_scaler.fit(df[["close"]])

# Normalizing/Scaling the DF
scaler = RobustScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

save_model = False
save_fig = False
epochs = 800
# How many periods looking back to learn
n_per_in  = 15
# How many periods to predict
n_per_out = 5
# Features 
n_features = df.shape[1]

title = "model-v3--in-%d--lb-%d--la-%d--ep-%d--aof-tanh-adam--lm-mse-acc" % (
    n_features,
    n_per_in,
    n_per_out,
    epochs
)

# Splitting the data into appropriate sequences
X, y = split_sequence(df.to_numpy(), n_per_in, n_per_out)

## Creating the NN

# Instatiating the model
model = Sequential()

# Activation
activ = "tanh"

# Input layer
model.add(LSTM(90, activation=activ, return_sequences=True, input_shape=(n_per_in, n_features)))

# Hidden layers
layer_maker(n_layers=1, n_nodes=30, activation=activ)

# Final Hidden layer
model.add(LSTM(60, activation=activ))

# Output layer
model.add(Dense(n_per_out))

# Model summary
model.summary()

# Compiling the data with selected specifications
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

## Fitting and Training
logs = "logs/%s" % title
tboard_callback = keras.callbacks.TensorBoard(log_dir=logs)
res = model.fit(X, y, epochs=epochs, batch_size=64, validation_split=0.2, callbacks=[tboard_callback])
if save_model:
    model.save(
        "models/%s" % title,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
visualize_training_results(res)

# model = keras.models.load_model("models/%s" % title)

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["close"]]), index=df.index, columns=["close"])

# Getting a DF of the predicted values to validate against
predictions = validater(n_per_in, n_per_out)

# print(actual)
# print(predictions)

# Printing the RMSE
print("RMSE:", val_rmse(actual, predictions))
    
# Plotting
plt.figure(figsize=(16,6))

# Plotting those predictions
plt.plot(predictions, label="Predicted")

# Plotting the actual values
plt.plot(actual, label="Actual")

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
# plt.xlim("2020-03-01", "2020-03-30")
if save_fig:
    plt.savefig(title, dpi=300)
plt.show()


# Predicting off of the most recent days from the original DF
yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))

# Transforming the predicted values back to their original format
yhat = close_scaler.inverse_transform(yhat)[0]

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat, 
                     index=pd.date_range(start=df.index[-1]+timedelta(days=1), 
                                         periods=len(yhat), 
                                         freq="B"), 
                     columns=["close"])

# Number of periods back to plot the actual values
pers = n_per_in

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(df[["close"]].tail(pers)), 
                      index=df.close.tail(pers).index, 
                      columns=["close"]).append(preds.head(1))

# Printing the predicted prices
print(preds)

# Plotting
plt.figure(figsize=(16,6))
plt.plot(actual, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()
