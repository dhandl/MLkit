##Create your first MLP in Keras
#from keras.models import Sequential
#from keras.layers import Dense
#import numpy
## fix random seed for reproducibility
#numpy.random.seed(7)
## load pima indians dataset
#dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
## split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
## create model
#model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
## Compile model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit the model
#model.fit(X, Y, epochs=150, batch_size=10)
## evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# load and prepare the dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
# 1. define the network
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(32, activation='relu'))
#model.add(Dense(6, activation='relu'))
#model.add(Dense(4, activation='relu'))
#model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. fit the network
history = model.fit(X, Y, epochs=200, batch_size=10)
# 4. evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. make predictions on independent dataset
new_dataset = numpy.loadtxt("pima-indians-diabetes_predictions.csv", delimiter=",")
p = new_dataset[:,0:8]
p_true = new_dataset[:,8]
probabilities = model.predict(p)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))
# round probabilities
rounded = [round(x[0]) for x in probabilities]
print("Rounded probabilities: ")
print(rounded)
print("True probabilities: ")
print(p_true)
checklist =  numpy.equal(rounded, p_true)
print checklist
y = 0
for i in checklist:
  if i: y += 1
print("Correct predictions: %i"%(y))
