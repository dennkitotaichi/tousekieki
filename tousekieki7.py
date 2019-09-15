import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


data_file = 'tousekieki.csv'
data = pd.read_csv(data_file, index_col=0, encoding="utf-8_sig", skipfooter=1, engine='python')

print (data)

df = pd.DataFrame(data)
print(df)

targets = df["target"]
print(targets )



target = data["target"].values
sentoral1 = data["sentoral1"].values
sentoral2 = data["sentoral2"].values
consoul1 = data["consoul"].values



print('欠損値の個数kessonnti kosuu')
print(data.isnull().sum(), '\n')

print('基本統計量kihonntoukeiryou')
print(data['target'].describe(), '\n')

plt.title('tousekieki yosokuti')
plt.plot(range(len(target)), target)
plt.plot(range(len(sentoral1)), sentoral1)
plt.plot(range(len(sentoral2)), sentoral2)
plt.plot(range(len(consoul1)), consoul1)
plt.show()


print(targets )

lookback = 3

## エポック数
epochs = 3000


    
## 訓練・検証・テスト用データを作成
## 過去30日分の株価より当日の株価とする
def data_split(data, start, end, lookback,consoul1,sentoral1,sentoral2):
    length = abs(start-end)
    X = np.zeros((length, lookback))
    y = np.zeros((length, 1))
    
    for i in range(length):
        j = start - lookback + i
        k = j + lookback
        
        X[i] = consoul1[k]
        X[:,[1]]=sentoral1[k]
        X[:,[2]]=sentoral2[k]
        y[i] = data[k]
        print("tugiha  i    dayo")
        print(i)
        print("tugihaXdayo")
        print(X)
        print("tugihaYdayo")
        print(y)
    return X, y
print("imawakaranaitokoro")



## 訓練・検証・テスト用データ
(X_train, y_train) = data_split(target, -100, -60, lookback,consoul1,sentoral1,sentoral2)
(X_valid, y_valid) = data_split(target, -60, -30, lookback,consoul1,sentoral1,sentoral2)
(X_test, y_test) = data_split(target, -30, 0, lookback,consoul1,sentoral1,sentoral2)


print("X_trainnoataidayo")
print(X_train.shape)

## 訓練
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))




model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])


result = model.fit(X_train, y_train, 
                  verbose=0,   ## 詳細表示モード 
                  epochs=epochs, 
                  batch_size=64, 
                  shuffle=True, 
                  validation_data=(X_valid, y_valid))

## 訓練の損失値をプロット
epochs = range(len(result.history['loss']))
plt.title('損失値（Loss）')
plt.plot(epochs, result.history['loss'], 'bo', alpha=0.6, marker='.', label='訓練kunnrenn', linewidth=1)
plt.plot(epochs, result.history['val_loss'], 'r', alpha=0.6, label='検証kennsyou', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.ylim(0, 2000)
plt.show()


## 予測値
df_predict =  pd.DataFrame(model.predict(X_test), columns=['予測値yosokuti'])


## kunnrennde-ta kennsyoude-ta  sonnsitutiをプロット
history_dict = result.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values) + 1)
pre_date = df.index[-len(y_test):].values
plt.title(" kunnrennde-ta kennsyoude-ta  sonnsitutiをプロット")
plt.plot(epochs,loss_values,'bo',label='TRAINNING LOSS')
plt.plot(epochs,val_loss_values,'r',label='VALIDATION LOSS')
plt.xlabel('Epochs')
plt.ylabel('lOSS')
plt.legend()
plt.show()


## 予測結果をプロット
pre_date = df.index[-len(y_test):].values
plt.title("実際の終値と予測値jissaino tousekiekinoudo  to yosokuti")
plt.plot(pre_date, y_test, 'b', alpha=0.6, marker='.', label="実際 jissaino tousekieki noudo ", linewidth=1)
plt.plot(pre_date, df_predict['予測値yosokuti'].values, 'r', alpha=0.6, marker='.', label="予測値 yosokuti", linewidth=1)
plt.xticks(rotation=70)
plt.legend()
plt.grid(True)
plt.show()

## RMSEの計算
print("二乗平均平方根誤差（RMSE） : %.3f" %  
       np.sqrt(mean_squared_error(y_test, df_predict["予測値yosokuti"].values)))
results = model.evaluate(X_test,y_test)
print(results )




