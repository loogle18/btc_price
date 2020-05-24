# Library Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
plt.style.use("ggplot")

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout


def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(12,4))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the univariate time sequence
    """
    X, y = [], []

    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out

        if out_end > len(seq):
            break

        seq_x, seq_y = seq[i:end], seq[end:out_end]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def get_data(path, scalers = None):
    headers = [
        "date", "open", "high", "low", "close", "volume_btc", "volume_usd",
        "market_cap_all",
        "fng_value",
        "btc_dominance",
        "neutral",
        "positive",
        "negative"
    ]
    df = pd.read_csv(path, sep=",", names=headers, skiprows=1, low_memory=False)
    df = df.set_index("date")[["close", "fng_value", "positive", "negative"]]
    if not scalers:
        scalers = {}
    for column in df.columns.values:
        scalers[column] = MinMaxScaler()
        df[column] = scalers[column].fit_transform(df[column].values.reshape(-1,1))
    return scalers, pd.DataFrame(df, columns=df.columns, index=df.index)


def get_x_y_shape(column, df, n_per_in, n_per_out, n_features):
    # Splitting the data into appropriate sequences
    x, y = split_sequence(list(df[column]), n_per_in, n_per_out)

    # Reshaping the x variable from 2D to 3D
    x = x.reshape((x.shape[0], x.shape[1], n_features))
    return x, y


def get_x_y_shapes(df, n_per_in, n_per_out, n_features):
    shapes = {}
    for column in df.columns.values:
        x, y = get_x_y_shape(column, df, n_per_in, n_per_out, n_features)
        shapes[column] = { "x": x, "y": y }
    return shapes


def build_model(columns, shape, n_per_out):
    inputs = []
    layers = []
    for column in columns:
        keras_input = keras.Input(shape=shape,name=column)
        inputs.append(keras_input)
        layers.append(LSTM(64, return_sequences=False)(keras_input))
    output = keras.layers.concatenate(inputs=layers)
    output = Dense(n_per_out, activation="linear", name="weighted_average")(output)
    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer="adamax", loss="mse")
    return model

save = False

epochs = 800

# How many periods looking back to train
n_per_in = 15

# How many periods ahead to predict
n_per_out = 5

# Features (in this case it's 1 because there is only one feature: price)
n_features = 1

title = "model-v7--il-4--lb-%d--la-%d--ep-%d--aof-linear-adamax--nl-64--bs-128--s-false--vs-01" % (n_per_in, n_per_out, epochs)

scalers, df = get_data("data/training/final.csv")
_, df_test = get_data("data/testing/final.csv", scalers)

shapes = get_x_y_shapes(df, n_per_in, n_per_out, n_features)
shapes_test = get_x_y_shapes(df_test, n_per_in, n_per_out, n_features)

training_data = list(map(lambda shape: shape["x"], shapes.values()))
testing_data = list(map(lambda shape: shape["x"], shapes_test.values()))

logs = "logs/%s" % title
tboard_callback = keras.callbacks.TensorBoard(log_dir=logs)
model = build_model(df.columns.values, (n_per_in, n_features), n_per_out)
res = model.fit(training_data, shapes["close"]["y"], epochs=epochs, batch_size=128, shuffle=False, validation_split=0.1, callbacks=[tboard_callback])
if save:
    model.save(
        "models/%s" % title,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
# visualize_training_results(res)
# model = keras.models.load_model("models/%s" % title)
plt.figure(figsize=(14,5))

predicted_list = []
actual_list = []
for n in reversed(range(1, len(shapes_test["close"]["x"]) + 1)):
    # print(scalers["close"].inverse_transform(shapes_test["close"]["y"][-n].reshape(-1,1)))
    yhat = model.predict([
        # shapes_test["open"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["high"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["low"]["x"][-n].reshape(1, n_per_in, n_features),
        shapes_test["close"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["volume_btc"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["volume_usd"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["market_cap_all"]["x"][-n].reshape(1, n_per_in, n_features),
        shapes_test["fng_value"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["btc_dominance"]["x"][-n].reshape(1, n_per_in, n_features),
        # shapes_test["neutral"]["x"][-n].reshape(1, n_per_in, n_features),
        shapes_test["positive"]["x"][-n].reshape(1, n_per_in, n_features),
        shapes_test["negative"]["x"][-n].reshape(1, n_per_in, n_features),
    ])[0].tolist()

    # Transforming values back to their normal prices
    yhat = scalers["close"].inverse_transform(np.array(yhat).reshape(-1,1)).flatten().tolist() #[0]
    # print(yhat)
    predicted_list.extend(yhat)

    # Getting the actual values from the last available y variable which correspond to its respective X variable
    actual = scalers["close"].inverse_transform(shapes_test["close"]["y"][-n].reshape(-1,1)).flatten().tolist() #[0]
    # print(actual)
    actual_list.extend(actual)


if len(predicted_list) > 5:
    predicted_list = predicted_list[::5] + predicted_list[-4:]
    actual_list = actual_list[::5] + actual_list[-4:]


# print("Predicted Prices:\n", predicted_list)
plt.plot(predicted_list, label='Predicted')

# print("\nActual Prices:\n", actual_list)
plt.plot(actual_list, label='Actual')

plt.title(title)
plt.ylabel("Price")
plt.legend()
if save:
    plt.savefig("prediction_plots/%s.png" % title, dpi=300)
plt.show()
