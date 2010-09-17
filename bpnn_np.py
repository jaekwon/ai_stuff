# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
#
# Replaced core with Numpy operations. 
# Released to the public by demand.
# Jae Kwon <jae@glyphtree.com>

import numpy
import string

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0, random=False):
	if random:
		return (numpy.random.random( (I, J) ) - 0.5) * (fill*2.0)
	else:
		return numpy.ones( (I, J) ) * fill

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
	return numpy.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
	return 1.0 - y**2.0

class NN:
	def __init__(self, ni, nh, no):
		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# activations for nodes
		self.ai = numpy.ones( (self.ni) )
		self.ah = numpy.ones( (self.nh) )
		self.ao = numpy.ones( (self.no) )
		
		# create weights
		self.wi = makeMatrix(self.ni, self.nh, random=True, fill=0.2)
		self.wo = makeMatrix(self.nh, self.no, random=True, fill=2.0)
		# last change in weights for momentum   
		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	def update(self, inputs):
		if len(inputs) != self.ni-1:
			raise ValueError('wrong number of inputs')
		self.ai[:-1] = inputs # or copy it?
		# hidden activations
		self.ah = sigmoid(numpy.dot(self.ai, self.wi))
		# output activations
		self.ao = sigmoid(numpy.dot(self.ah, self.wo))
		return self.ao

	def backPropagate(self, targets, N, M):
		if len(targets) != self.no:
			raise ValueError('wrong number of target values')
		# calculate error terms for output
		output_deltas = dsigmoid(self.ao) * (targets - self.ao)
		# calculate error terms for hidden
		hidden_deltas = dsigmoid(self.ah) * numpy.dot(self.wo, output_deltas)
		# update output weights
		change = self.ah.reshape((self.nh, 1))*output_deltas
		self.wo += N*change + M*self.co
		self.co = change
		# update input weights
		change = self.ai.reshape((self.ni, 1))*hidden_deltas
		self.wi += N*change + M*self.ci
		self.ci = change
		# calculate error
		return (0.5*(targets-self.ao)**2.0).sum()

	def test(self, patterns):
		for p in patterns:
			print(p[0], '->', self.update(p[0]))

	def weights(self):
		print('Input weights:')
		for i in range(self.ni):
			print(self.wi[i])
		print()
		print('Output weights:')
		for j in range(self.nh):
			print(self.wo[j])

	def train(self, patterns, iterations=1000, N=0.5, M=0.1):
		# N: learning rate
		# M: momentum factor
		import sys
		for i in range(iterations):
			#print "training %i" % i
			error = 0.0
			for p in patterns:
				#sys.stdout.write('.')
				#sys.stdout.flush()
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 100 == 0 or True:
				#print('error %-.5f' % error)
				pass
		return error


def demo():
	# Teach network XOR function
	pat = [
		[numpy.array([0,0]), numpy.array([0])],
		[numpy.array([0,1]), numpy.array([1])],
		[numpy.array([1,0]), numpy.array([1])],
		[numpy.array([1,1]), numpy.array([0])],
	]

	# create a network with two input, two hidden, and one output nodes
	n = NN(2, 2, 1)
	# train it with some patterns
	n.train(pat)
	# test it
	n.test(pat)



if __name__ == '__main__':
	demo()

