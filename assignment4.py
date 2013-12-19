import animate, numpy, pylab, random

#transition matrix initialization
trans = ((1, 3 ,4 ,12),
		 (0, 2, 5, 13),
		 (3, 1, 6, 14),
		 (2, 0, 7, 15),
		 (5, 7, 0, 8),
		 (4, 6, 1, 9),
         (7, 5, 2, 10),
		 (6, 4, 3, 11),
		 (9, 11, 12, 4),
		 (8, 10, 13, 5),
		 (11, 9, 14, 6),
		 (10, 8, 15, 7),
		 (13, 15, 8, 0),
		 (12, 14, 9, 1),
		 (15, 13, 10, 2),
		 (14, 12, 11, 3))



#several reward matrices initialization
'''rew = ((1, 0, 1, 0),
	   (-1, 1, 0, 0),
	   (1, -1, 0, 0),
	   (-1, 0, 1, 0),
	   (0, 0, -1, 1),
	   (-1, 1, -1, 1),
	   (1, -1, 1, 0),
	   (0, 0, -1, 1),
	   (0, -1, 1, 0),
	   (0, 1, 1, -1),
	   (1, -1, 1, -1),
	   (0, 0, 1, -1),
	   (1, 0, -1, 0),
	   (-1, 1, 0, 0),
	   (1, -1, 0, 0),
	   (-1, 0, -1, 0))
'''
'''rew = ((0, 3, 0, 3),
	   (-1, 0, -1, 3),
	   (0, -1, -3, 3),
	   (-2, -1, 0, 3),
	   (-3, 1, -1, 0),
	   (0, 0, 0, 0),
	   (0, 0, 0, 0),
	   (-3, -1, -1, 0),
	   (-3, 3, 1, -1),
	   (0, 0, 0, 0),
	   (0, 0, 0, 0),
	   (-3, -1, 3, -1),
	   (0, 3, -1, -1),
	   (-1, 1, -3, -1),
	   (1, -1, -3, -1),
	   (-1, -1, -1, -1))'''

'''rew = ((0, 0, 0, 0),
	   (-1, 0, -1, 0),
	   (0, -1, -1, 0),
	   (-1, -1, 0, 1),
	   (-1, 1, -1, 0),
	   (0, 0, 0, 0),
	   (0, 0, 0, 0),
	   (-1, -1, -1, 0),
	   (-1, 0, 0, -1),
	   (0, 0, 0, 0),
	   (0, 0, 0, 0),
	   (-1, -1, 1, -1),
	   (0, 1, -1, -1),
	   (-1, 0, -1, -1),
	   (1, -1, -1, -1),
	   (-1, 0, -1, 0))'''

rew = ((0, -1, 0, -1), 	# 0
	  (-2, 0, -2, -1), 	# 1
		(0,	-1,	-2,	-1), 	# 2
			(-2, -1, 2,	-1), 	# 3
			(-2, -1, -2, 0), 	# 4
			(0,	0, 0, 0), 	# 5
			(0,	-1,	0, 0), 	# 6
			(-2, -1, -2, 0), 	# 7
			(	-2,	-1,	0,	-1), 	# 8
			(	0,	0,	0,	-1), 	# 9
			(	0,	-1,	0,	-1), 	# 10
			(	-2,	0,	0,	-1), 	# 11
			(	2,	-1,	-2,	-1), 	# 12
			(	-2,	0,	-2,	-1), 	# 13
			(	-2,	-1,	-2,	0), 	# 14
			(	0,	-1,	0,	-1)) 



def argmax(f, args):
	'''
	argmax function that does same stuff like argmax in matlab or in optimization
	'''
	mi = None
	m = -1e10
	for i in args:
		v = f(i)
		if v > m:
			m = v
			mi = i
	return mi


policy = [None for s in trans]
value = [0 for s in trans]
gamma = 0.1


#policy iteration implementation
for p in range(100):
	for s in range(len(policy)):
		policy[s] = argmax(lambda(a): rew[s][a] + gamma * value[trans[s][a]], range(4))
	for s in range(len(value)):
		a = policy[s]
		value[s] = rew[s][a] + gamma * value[trans[s][a]]

#sequence from policy retrival
sequence = []

sequence.append(0)
for i in range(len(policy)):
	sequence.append(trans[i][policy[i]])

print sequence

#loading all the available images
images = (pylab.imread('step1.png'),
          pylab.imread('step2.png'),
          pylab.imread('step3.png'),
          pylab.imread('step4.png'),
          pylab.imread('step5.png'),
          pylab.imread('step6.png'),
          pylab.imread('step7.png'),
          pylab.imread('step8.png'),
          pylab.imread('step9.png'),
          pylab.imread('step10.png'),
          pylab.imread('step11.png'),
          pylab.imread('step12.png'),
          pylab.imread('step13.png'),
          pylab.imread('step14.png'),
          pylab.imread('step15.png'),
          pylab.imread('step16.png'))

#visualization of the robot walk

#comic = numpy.concatenate([images[i] for i in sequence], axis=1)

#pylab.imshow(comic)
#pylab.show()
animate.draw(sequence)



class Environment :
	'''
	Representation of the environment for the Q-learning algorithm
	'''
	def __init__(self, state=0):
		self.state = state
		self.trans = ((1, 3 ,4 ,12),
		 			  (0, 2, 5, 13),
		 			  (3, 1, 6, 14),
		 			  (2, 0, 7, 15),
		 			  (5, 7, 0, 8),
		 			  (4, 6, 1, 9),
         			  (7, 5, 2, 10),
		 			  (6, 4, 3, 11),
		 			  (9, 11, 12, 4),
		 			  (8, 10, 13, 5),
		 			  (11, 9, 14, 6),
		 			  (10, 8, 15, 7),
		 			  (13, 15, 8, 0),
		 			  (12, 14, 9, 1),
		 			  (15, 13, 10, 2),
		 			  (14, 12, 11, 3))
		self.rew = ((0, -1, 0, -1), 	# 0
	  (-2, 0, -2, -1), 	# 1
		(0,	-1,	-2,	-1), 	# 2
			(-2, -1, 2,	-1), 	# 3
			(-2, -1, -2, 0), 	# 4
			(0,	0, 0, 0), 	# 5
			(0,	-1,	0, 0), 	# 6
			(-2, -1, -2, 0), 	# 7
			(	-2,	-1,	0,	-1), 	# 8
			(	0,	0,	0,	-1), 	# 9
			(	0,	-1,	0,	-1), 	# 10
			(	-2,	0,	0,	-1), 	# 11
			(	2,	-1,	-2,	-1), 	# 12
			(	-2,	0,	-2,	-1), 	# 13
			(	-2,	-1,	-2,	0), 	# 14
			(	0,	-1,	0,	-1)) 

	def go(self, a):
		'''
		performing one step of the robot
		'''
		r = self.rew[self.state][a]
		self.state = self.trans[self.state][a] 
		return self.state, r


#Q-learning implementation
environment = Environment()
epsilon = 0.7
Q = numpy.zeros((16,4))
stepSize = 0.2
discount = 0.8
state = 0

for p in range(10000):
	transition = []
	action = 0
	if random.random() > epsilon:
		action = random.randint(0, 3)
		transition = environment.go(action)	
	else:
		action = argmax(lambda(a): Q[state][a], range(4))
		transition = environment.go(action)

	Q[state][action] = Q[state][action] + stepSize * (transition[1] + discount * Q[transition[0]][argmax(lambda(a): Q[transition[0]][a], range(4))] - Q[state][action])
	state = transition[0]		

print Q


#retrieve the path from the initial state
sequence = []
sequence.append(state)
for i in range(16):
	action = argmax(lambda(a): Q[state][a], range(4))
	#print action
	transition = environment.go(action)
	#print transition
	sequence.append(transition[0])
	state = transition[0]

print sequence
animate.draw(sequence)
#visualization of the robot walk
comic = numpy.concatenate([images[i] for i in sequence], axis=1)

pylab.imshow(comic)
pylab.show()

