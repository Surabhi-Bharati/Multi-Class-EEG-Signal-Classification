import pandas as pd
from sklearn import neighbors
import numpy as np
import seaborn
training_data = pd.DataFrame()

training_data['variable_1'] = [0.3051,0.4949,0.6974,0.3769,0.2231,0.341,0.4436,0.5897,0.6308,0.5]
training_data['variable_2'] = [0.5846,0.2654,0.2615,0.4538,0.4615,0.8308,0.4962,0.3269,0.5346,0.6731]
training_data['outcome'] = ['win','win','win','win','win','loss','loss','loss','loss','loss']

seaborn.lmplot(x='variable_1', y='variable_2', data=training_data, hue="outcome", fit_reg=False, scatter_kws  = {"marker": "D", "s": 100})
X = training_data[['variable_1', 'variable_2']].to_numpy()
# X = np.array(training_data[['variable_1', 'variable_2']])
y = np.array(training_data['outcome'])
print ("X: ",X, ", Shape:", X.shape)
print ("y: ",y, ", Shape:", y.shape)
clf = neighbors.KNeighborsClassifier(3, weights = 'uniform')
trained_model = clf.fit(X, y)
print ("trained_model: ", trained_model)
print (clf.predict(X))
print (y)
trained_model.score(X, y)
x_test = np.array([[.4,.6]])
trained_model.predict(x_test)
trained_model.predict_proba(x_test)
import numpy as np
# input_data = (.4, .6)  # Loss
input_data = (.3, .6)  # Win

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = trained_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'loss'):
  print('The prediction is LOSS')
else:
  print('The prediction is WIN')
import pickle
filename = 'trained_model.sav'
pickle.dump(trained_model, open(filename, 'wb'))
# loading the saved model
loaded_model = pickle.load(open(filename, 'rb'))
# input_data = (.4, .6)  # Loss
input_data = (.3, .6)  # Win

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 'loss'):
  print('The prediction is LOSS')
else:
  print('The prediction is WIN')