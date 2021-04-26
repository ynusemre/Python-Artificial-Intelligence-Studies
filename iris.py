import pandas as pd
import numpy as np 
import tensorflow as tf


def convert_one_hot(v):
	one_hot = []
	
	for i in v:
		temp=[]
		for j in range(3):
			if(j==i):
				temp.append(1)
			else:
				temp.append(0)

		one_hot.append(temp)
	return one_hot

def read_data(v):
	
	columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
	df_data=pd.read_csv("iris.csv",names=columns)

	x=[]
	y=[]

	species_dict={"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}

	for i in range(len(df_data)):
		temp_values=[]
		temp_values.append(df_data["SepalLengthCm"][i])
		temp_values.append(df_data["SepalWidthCm"][i])
		temp_values.append(df_data["PetalLengthCm"][i])
		temp_values.append(df_data["PetalWidthCm"][i])
		x.append(temp_values)
		y.append(species_dict[df_data["Species"][i]])

	y=convert_one_hot(y)	

	x_train=[]
	x_test=[]
	y_train=[]
	y_test=[]

	
	for i in range(50):
		if(i<v):
			for j in range(3):
				x_train.append(x[i+50*j])
				y_train.append(y[i+50*j])

		else:
			for j in range(3):
				x_test.append(x[i+50*j])
				y_test.append(y[i+50*j])


	return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test=read_data(30)



n_nodes_hl1=16
n_nodes_hl2=16
n_nodes_hl3=16

n_classes=3 #output layer's node number

x=tf.placeholder("float",[None,4])
y=tf.placeholder("float")

def neural_network_model(data):
	hidden_1_layer={"weights":tf.Variable(tf.random_normal([4,n_nodes_hl1])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl1])) }

	hidden_2_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl2])) }
	
	hidden_3_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					"biases":tf.Variable(tf.random_normal([n_nodes_hl3])) }

	
	output_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
					"biases":tf.Variable(tf.random_normal([n_classes])) }


	l1=tf.add(tf.matmul(data,hidden_1_layer["weights"]),hidden_1_layer["biases"])				
	l1=tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_2_layer["weights"]),hidden_2_layer["biases"])				
	l2=tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_3_layer["weights"]),hidden_3_layer["biases"])				
	l3=tf.nn.relu(l3)

	output=tf.add(tf.matmul(l3,output_layer["weights"]),output_layer["biases"])				
	
	return output

def training_neural_network(x):
	prediction=neural_network_model(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

	hm_epochs=10000


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			_,c=sess.run([optimizer,cost],feed_dict={x:x_train,y:y_train})

		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,"float"))
		
		print("Accuracy=",accuracy.eval({x:x_test,y:y_test}))

training_neural_network(x)		