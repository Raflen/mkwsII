import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from copy import copy

#from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

ratio = 0.75
numpy.random.seed(123654)

#read data, take only interesting values
df = pandas.read_csv('slSimulated.txt', header=0)
dataset = df[['sl']][:1000].values

#copy the ground truth 
last_step = dataset[int(ratio*dataset.shape[0]) - 1, 0]
y_true = copy(dataset[int(ratio*dataset.shape[0]) : -2, 0])
ds_true = df[['sl', 'fi']][:1000].values

#shift values by 1
df = pandas.DataFrame(dataset)
columns = [df.shift(i) for i in range(1, 2)]
columns.append(df)
df = pandas.concat(columns, axis=1)
df.fillna(0, inplace=True)

#transform into stationary data, drop irrelevant lines
df = df.diff()
df.columns = ['sl', 'sl_shifted']
df.drop(df.index[[0, 1, -1, -2]], inplace=True)

#split to train and test
ds_train = df[ : int(ratio*df.shape[0])].values
ds_test = df[int(ratio*df.shape[0]) : ].values



#rescale train set, then use its bordes to rescale test set
maxs = numpy.amax(ds_train)
mins = numpy.amin(ds_train)
normalize = lambda x : 2 * ((x - mins) / (maxs - mins)) -1
dataset_train = numpy.array([[normalize(ds_train[i, 0]), normalize(ds_train[i, 0])] 
                                                        for i in range(ds_train.shape[0])])
dataset_test = numpy.array([[normalize(ds_test[i, 0]), normalize(ds_test[i, 0])] 
                                                        for i in range(ds_test.shape[0])])
x_train = dataset_train[ : , 0 : -1]
y_train = dataset_train[ : , 1]
x_test = dataset_test[ : , 0 : -1]
y_test = dataset_test[ : , 1]
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

#define a network
neurons = 8
nb_epoch = 500
batch_size = 1

print("I'm starting...")

model = Sequential()
model.add(LSTM(neurons,
               batch_input_shape=(batch_size, x_train.shape[1], x_train.shape[2]),
               dropout=0.5,
               stateful=True))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam')

print('Im done compiling.')

for i in range(nb_epoch):
    print('{}th epoch'.format(i))
    model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

#forecast the entire training dataset to build up state for forecasting
x_train_reshaped = x_train.reshape(len(x_train), 1, 1)
model.predict(x_train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
#last_step = y_train[-1]
for i in range(len(x_test)):
	# predict one step forward
    x = x_test[i, : ]
    x = x.reshape(1,1,1)
    y_predicted = model.predict(x, batch_size=1)[0]
	# rescale
    y_predicted = mins + (((y_predicted + 1) * (maxs - mins)) / 2)
	# reverse diff()     
    y_predicted = y_predicted + last_step
    last_step = y_predicted
	# store forecast
    predictions.append(y_predicted)

#prepare plot
preds = numpy.asarray([predictions[i][0] for i in range(len(y_test))])
colors = seaborn.crayon_palette(['Inchworm', 'Lavender'])
x = ds_true[ : , 1]
y  = ds_true[ : , 0]
plt.scatter(x, y, label='Real values', color=colors[0], s=8)
x1 = ds_true[int(ratio*dataset.shape[0]) + 1: , 1]
y1 = preds[:] 
plt.scatter(x1, y1, label='Predicted values', color=colors[1], s=8)
plt.xlabel('Phi')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.show()

