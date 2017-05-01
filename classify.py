import numpy as np
import theano
import csv
from keras.models import Sequential
from keras.layers import Dense,Activation


def one_hot(line):
	for i,x in enumerate(line):
		if float(x)==1.0:
			return i
x=[]
y=[]
xTr=[]
yTr=[]
xTe=[]
with open('trainX.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
		xTr.append(np.asarray(line))
		x.append(np.asarray(line))

with open('trainY.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
#		y=one_hot(line)
		yTr.append(line)
		y.append(line)

with open('testX.csv','r') as f:
	reader=csv.reader(f,delimiter=",")
	for line in reader:
		xTe.append(line)		


xTr=np.asarray(xTr)
yTr=np.asarray(yTr)#.reshape(-1,1)
xTe=np.asarray(xTe)
x=np.asarray(x)
y=np.array(y)

sep=np.arange(len(yTr))
np.random.shuffle(sep)
xValid=xTr[sep[:500]]
yValid=yTr[sep[:500]]
xTr=xTr[sep[500:]]
yTr=yTr[sep[500:]]

print xTr.shape
print yTr.shape
print xTe.shape

model=Sequential()
model.add(Dense(units=784,input_dim=784)) #614656 for fully interconnected
model.add(Activation('sigmoid'))
model.add(Dense(units=784))
model.add(Activation('sigmoid'))
model.add(Dense(units=100))
model.add(Activation('sigmoid'))
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
model.add(Dense(units=10))
model.add(Activation('sigmoid'))


#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

num_epochs=500
model.fit(xTr, yTr,epochs=num_epochs)
loss_and_metrics = model.evaluate(xTr, yTr, batch_size=128)
print loss_and_metrics
classes = model.predict(xValid, batch_size=128)
ans=[]
res=[]
for i in range(len(xValid)):
	ans.append(np.argmax(classes[i])==one_hot(yValid[i]))
print sum(ans)*1.0/500
if sum(ans)*1.0/500 > .95:
	model.fit(x,y,epochs=num_epochs)
	classes = model.predict(xTe, batch_size=128)
	for i in range(len(xTe)):
		res.append((i,np.argmax(classes[i])))

	with open("answer.csv","w") as f:
		f.write("id,digit\n")
		w= csv.writer(f,delimiter=",")
		for i in res:
			w.writerow([i[0],i[1]])