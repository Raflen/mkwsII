'''
Deep neural network to predict velocity of the
laminar combustion on given experimental data.
'''

import pandas
import numpy

numpy.random.seed(1234)

import matplotlib.pyplot as plt
import seaborn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import mean_squared_error
from copy import copy

split_ratio = 0.7
mse = dict()  # mean squared error
seaborn.set_style('white')
seaborn.set_context('paper')

# Read data.
dataframe = pandas.read_csv('slSimulated.csv', sep=',', header=0)
dataset = dataframe.values.astype("float32")

# Shuffle data.
for i in range(400):
    numpy.random.shuffle(dataset)

# Save data for plotting.
split_point = round(split_ratio * dataset.shape[0])
x_plot = copy(dataset[split_point : , 1 : ])
y_plot = copy(dataset[split_point : ,  : 1])

# Prepare data.
mins = numpy.amin(dataset, axis=0)
maxs = numpy.amax(dataset, axis=0)

for column in range(dataset.shape[1]):
    dataset[ : , column] = [(dataset[i, column] - mins[column]) /
                            (maxs[column] - mins[column])
                            for i in range(dataset.shape[0])]


# Split data.
x_train = dataset[0 : split_point, 1 : ]
y_train = dataset[0 : split_point,  : 1]
x_test = dataset[split_point : , 1 : ]
y_test = dataset[split_point : ,  : 1]

# Build model.
model = Sequential()

model.add(Dense(150, input_dim=3, activation='relu'))

model.add(Dense(3000, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(3000, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))

model.add(Dense(1, activation='sigmoid'))

# Compile and teach.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=12, batch_size=12, verbose=1)

# Predict.
predictions = model.predict(x_test, batch_size=1, verbose=1)

mse['Normalized'] = mean_squared_error(y_test, predictions)

# Rescale.
predictions = [predictions[i] * (maxs[0] - mins[0]) + mins[0]
                             for i in range(len(predictions))]

mse['Rescaled'] = mean_squared_error(y_plot, predictions)

# Compare.
plt.xlabel('phi')
plt.ylabel('v')
plt.title('Results.')
plt.plot(x_plot[ : , -1], y_plot, 'bo', markersize=1, label='real values')
plt.plot(x_plot[ : , -1], predictions, 'ro', markersize=1, label='predicted values')

print('Mean squared error: ', mse)
seaborn.despine()
plt.show()