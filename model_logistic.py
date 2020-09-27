 
 # by R. Chandra
 #Source: https://github.com/rohitash-chandra/logisticregression_multiclass



from math import exp
import numpy as np

SIGMOID = 1
STEP = 2
LINEAR = 3

class logistic_regression:

	def __init__(self, num_epocs, train_data, test_data, num_features, learn_rate):
		self.train_data = train_data
		self.test_data = test_data 
		self.num_features = num_features
		self.num_outputs = self.train_data.shape[1] - num_features 
		self.num_train = self.train_data.shape[0]
		self.w = np.random.uniform(-0.5, 0.5, num_features)  # in case one output class 
		self.b = np.random.uniform(-0.5, 0.5, self.num_outputs) 
		self.learn_rate = learn_rate
		self.max_epoch = num_epocs
		self.use_activation = SIGMOID #SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear
		#self.use_activation = SIGMOID #SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear
		self.out_delta = np.zeros(self.num_outputs)

		print(self.w, ' self.w init') 
		print(self.b, ' self.b init') 
		print(self.out_delta, ' outdel init')


 
	def activation_func(self,z_vec):
		if self.use_activation == SIGMOID:
			y =  1 / (1 + np.exp(z_vec)) # sigmoid/logistic
		elif self.use_activation == STEP:
			y = (z_vec > 0).astype(int) # if greater than 0, use 1, else 0
			#https://stackoverflow.com/questions/32726701/convert-real-valued-numpy-array-to-binary-array-by-sign
		else:
			y = z_vec
		return y
 

	def predict(self, x_vec ): 
		z_vec = x_vec.dot(self.w) - self.b 
		output = self.activation_func(z_vec) # Output  
		return output
	
	def gradient(self, x_vec, output, actual):   
		if self.use_activation == SIGMOID :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =   (output - actual) 
		return out_delta

	def update(self, x_vec, output, actual):      
		self.w+= self.learn_rate *( x_vec *  self.out_delta)
		self.b+=  (1 * self.learn_rate * self.out_delta)
 

	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class

	def test_model(self, data, tolerance):  

		num_instances = data.shape[0]

		class_perf = 0
		sum_sqer = 0   
		for s in range(0, num_instances):	

			input_instance  =  self.train_data[s,0:self.num_features] 
			actual  = self.train_data[s,self.num_features:]  
			prediction = self.predict(input_instance) 
			sum_sqer += self.squared_error(prediction, actual)
			print(s, actual, prediction, sum_sqer, ' s, actual, prediction, sum_sqer')

			if(np.isclose(prediction, actual, atol=tolerance).any()):
				class_perf =  class_perf +1  

		rmse = np.sqrt(sum_sqer/num_instances)

		percentage_correct = float(class_perf/num_instances) * 100 

		print(percentage_correct, rmse,  ' class_perf, rmse') 
		# note RMSE is not a good measure for multi-class probs

		return ( rmse, percentage_correct)



 
	def SGD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.train_data[s,0:self.num_features]  
					actual  = self.train_data[s,self.num_features:]  
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual)
					self.out_delta = self.gradient( input_instance, prediction, actual)    # major difference when compared to GD
					#print(input_instance, prediction, actual, s, sum_sqer)
					self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.train_data, 0.3) 
			rmse_test, test_perc = self.test_model(self.test_data, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				

	def GD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.train_data[s,0:self.num_features]  
					actual  = self.train_data[s,self.num_features:]   
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual) 
					self.out_delta+= self.gradient( input_instance, prediction, actual)    # this is major difference when compared with SGD

					#print(input_instance, prediction, actual, s, sum_sqer)
				self.update(input_instance, prediction, actual)

			
				print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.train_data, 0.3) 
			rmse_test, test_perc = self.test_model(self.test_data, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				
	
 

#------------------------------------------------------------------
#MAIN



 
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]


train_data = np.asarray(dataset) # convert list data to numpy
test_data = train_data

 

learn_rate = 0.3
num_features = 2
num_epocs = 20

print(train_data)
 

lreg = logistic_regression(num_epocs, train_data, test_data, num_features, learn_rate)
(train_perc, test_perc, rmse_train, rmse_test) = lreg.SGD()
(train_perc, test_perc, rmse_train, rmse_test) = lreg.GD() 
 

#-------------------------------
#xor data


xor_dataset= [[0,0,0],
	[0,1,1],
	[1,0,1],
	[1,1,0] ]

xor_data = np.asarray(xor_dataset) # convert list data to numpy



num_epocs = 20
learn_rate = 0.9
num_features = 2

lreg = logistic_regression(num_epocs, xor_data, xor_data, num_features, learn_rate)
(train_perc, test_perc, rmse_train, rmse_test) = lreg.SGD()
(train_perc, test_perc, rmse_train, rmse_test) = lreg.GD() 
 