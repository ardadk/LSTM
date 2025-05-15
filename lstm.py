import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import warnings
warnings.filterwarnings('ignore')

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

df = pd.read_csv("TSLA.csv")

df["Date"] = pd.to_datetime(df["Date"])
tesla_df=df[["Date","Close"]]
tesla_df.index=tesla_df["Date"]
tesla_df.drop("Date",axis=1,inplace=True)

result_df=tesla_df.copy()

tesla_df=tesla_df.values

tesla_df = tesla_df.astype("float32")

def split_data(dataframe,test_size):
    pos = int(round(len(dataframe)*(1-test_size)))
    train=dataframe[:pos]
    test=dataframe[pos:]
    return train,test,pos
train,test,pos = split_data(tesla_df,0.20)

scaler_train = MinMaxScaler(feature_range=(0,1))
train=scaler_train.fit_transform(train)
scaler_test = MinMaxScaler(feature_range=(0,1))
test = scaler_test.fit_transform(test)


def create_features(data,lookback):
    X,Y =[],[]
    for i in range(lookback,len(data)):
        X.append(data[i-lookback:i,0])
        Y.append(data[i,0])
        
    return np.array(X),np.array(Y)
lookback=20

X_train, y_train = create_features(train,lookback)
X_test, y_test = create_features(test,lookback)



X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)


model = Sequential()
model.add(LSTM(units=50,
               activation="relu",
               input_shape=(X_train.shape[1],lookback)))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(loss="mean_squared_error",optimizer="adam")

callbacks = [EarlyStopping(monitor="val_loss",patience=3,verbose=1,mode="min"),
             ModelCheckpoint(filepath="mymodel.keras",monitor="val_loss",mode="min",
                             save_best_only=True,save_weights_only=False,verbose=1)]


history= model.fit(x=X_train,
                   y=y_train,
                   epochs=100,
                   batch_size=20,
                   validation_data=(X_test,y_test),
                   callbacks=callbacks,
                   shuffle=False)


loss = model.evaluate(X_test,y_test,batch_size=20)
print("\nTest loss: %.1f&&" % (100.0 *loss))

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler_train.inverse_transform(train_predict)
test_predict=scaler_test.inverse_transform(test_predict)

y_train = scaler_train.inverse_transform(y_train)
y_test = scaler_test.inverse_transform(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train,train_predict))

test_rmse = np.sqrt(mean_squared_error(y_test,test_predict))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

train_prediction_df =result_df[lookback:pos]
train_prediction_df["Predicted"] = train_predict
print(train_prediction_df.head())

test_prediction_df =result_df[pos+lookback:]
test_prediction_df["Predicted"] = test_predict
print(test_prediction_df.head())


plt.figure(figsize=(14,5))
plt.plot(result_df,label = "Real Values")
plt.plot(train_prediction_df["Predicted"],color="blue",label="Train Predicted")
plt.plot(test_prediction_df["Predicted"],color = "red",label="Test Predicted")
plt.xlabel("Time")
plt.ylabel("Stock Values")
plt.legend()
plt.show()
