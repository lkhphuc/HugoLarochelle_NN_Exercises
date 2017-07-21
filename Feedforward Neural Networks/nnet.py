from mlpython.learners.generic import Learner
import numpy as np
from scipy.special import expit

class NeuralNetwork(Learner):
    """
    Neural network for classification.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constante for the learning rate.
 
    Option ``sizes`` is the list of hidden layer sizes.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``seed`` is the seed of the random number generator.
 
    Option ``tanh`` is a boolean indicating whether to use the
    hyperbolic tangent activation function (True) instead of the
    sigmoid activation function (True).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``: Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 sizes=[200,100,50],
                 L2=0.001,
                 L1=0,
                 seed=1234,
                 tanh=True,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1
        self.seed=seed
        self.tanh=tanh
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations (DONE)
        and initializes the parameters of the neural network (TODO)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        n_hidden_layers = len(self.sizes)
        #############################################################################
        # Allocate space for the hidden and output layers, as well as the gradients #
        #############################################################################
        self.hs = []
        self.grad_hs = []
        # Create additional list holding preactivation at each hidden layers
        self.pre_act = []
        for h in range(n_hidden_layers):         
            self.hs += [np.zeros((self.sizes[h],))]       # hidden layer
            self.grad_hs += [np.zeros((self.sizes[h],))]  # ... and gradient
            self.pre_act += [np.zeros((self.sizes[h],))]
        self.hs += [np.zeros((self.n_classes,))]       # output layer
        self.grad_hs += [np.zeros((self.n_classes,))]  # ... and gradient
        self.pre_act += [np.zeros((self.n_classes,))]
        
        ##################################################################
        # Allocate space for the neural network parameters and gradients #
        ##################################################################
        self.weights = [np.zeros((self.input_size,self.sizes[0]))]       # input to 1st hidden layer weights
        self.grad_weights = [np.zeros((self.input_size,self.sizes[0]))]  # ... and gradient

        self.biases = [np.zeros((self.sizes[0]))]                        # 1st hidden layer biases
        self.grad_biases = [np.zeros((self.sizes[0]))]                   # ... and gradient

        for h in range(1,n_hidden_layers):
            self.weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]        # h-1 to h hidden layer weights
            self.grad_weights += [np.zeros((self.sizes[h-1],self.sizes[h]))]   # ... and gradient

            self.biases += [np.zeros((self.sizes[h]))]                   # hth hidden layer biases
            self.grad_biases += [np.zeros((self.sizes[h]))]              # ... and gradient

        self.weights += [np.zeros((self.sizes[-1],self.n_classes))]      # last hidden to output layer weights
        self.grad_weights += [np.zeros((self.sizes[-1],self.n_classes))] # ... and gradient

        self.biases += [np.zeros((self.n_classes))]                   # output layer biases
        self.grad_biases += [np.zeros((self.n_classes))]              # ... and gradient
            
        #########################
        # Initialize parameters #
        #########################

        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator

        ## PUT CODE HERE ##

        # initialize W0 and b0 at the first layer
        # distance used for calculating the range in which weights are initialized
        distance = self.sizes[0] + self.input_size
        self.weights[0] = self.rng.uniform(-np.sqrt(6.0 / distance), np.sqrt(6.0 / distance), self.weights[0].shape)

        #initialize for hidden layer
        for h in range(1, n_hidden_layers):
            distance = self.sizes[h] + self.sizes[h-1]
            self.weights[h] = self.rng.uniform(-np.sqrt(6.0 / distance), np.sqrt(6.0 / distance), self.weights[h].shape)
        
        # initialize for last layer 
        distance = self.sizes[-1] + self.n_classes
        self.weights[-1] = self.rng.uniform(-np.sqrt(6.0 / distance), np.sqrt(6.0 / distance), self.weights[-1].shape)
        
        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate
        
    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.targets)
        self.epoch = 0
        
    def train(self,trainset):
        """
        Trains the neural network until it reaches a total number of
        training epochs of ``self.n_epochs`` since it was
        initialize. (DONE)

        Field ``self.epoch`` keeps track of the number of training
        epochs since initialization, so training continues until 
        ``self.epoch == self.n_epochs``.
        
        If ``self.epoch == 0``, first initialize the model.
        """

        if self.epoch == 0:
            input_size = trainset.metadata['input_size']
            n_classes = len(trainset.metadata['targets'])
            self.initialize(input_size,n_classes)
            
        for it in range(self.epoch,self.n_epochs):
            for input,target in trainset:
                self.fprop(input,target)
                self.bprop(input,target)
                self.update()
        self.epoch = self.n_epochs
        
    # use the available sigmoid function of scipy
    def sigmoid(self, x):
        """ Compute sigmoid values for each value in array"""
        return expit(x)
    
    def sigmoid_deri(self, x):
        """Comppute the derivative of sigmoid"""
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def ftanh(self, x):
        """Compute tanh values for each value in array"""
        return (np.exp(np.dot(x,2)) - 1) / (np.exp(np.dot(x,2)) + 1)
    
    def ftanh_deri(self, x):
        """Compute the tanh derivative for each value in array"""
        return 1 - np.square(self.ftanh(x))
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def L2_loss(self):
        """Compute the L2 loss"""
        L2 = 0
        for i in range(len(self.weights)):
            L2 += np.sum(np.power(self.weights[i], 2))
        return L2
    
    def grad_L2(self):
        """Compute the gradient of L2 loss with respect to weights"""
        l2_grad = self.weights
        for i in range(len(l2_grad)):
            l2_grad[i] *= 2
        return l2_grad
    
    def L1_loss(self):
        """Compute the L1 loss"""
        L1 = 0
        for i in range(len(self.weights)):
            L1 += np.sum(self.weights[i])
        return L1
    
    def grad_L1(self):
        """Compute the gradient of L1 loss with respect to weights"""
        l1_grad = self.weights
        for h in range(len(l1_grad)):
            l1_grad[h] = 1.0 * (l1_grad[h] > 0)
            for i in range(len(l1_grad[h])):
                for j in range(len(l1_grad[h][i])):
                    if (l1_grad[h][i][j] == 0):
                        l1_grad[h][i][j] = -1.0
        return l1_grad
    
    def fprop(self,input,target):
        """
        Forward propagation: 
        - fills the hidden layers and output layer in self.hs
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        # First layer
        self.pre_act[0] = self.biases[0] + np.matmul(np.transpose(self.weights[0]), input)
        self.hs[0] = self.sigmoid(self.pre_act[0]) if not self.tanh else self.ftanh(self.pre_act[0])
        
        #Hidden layer
        for h in range(1, len(self.hs) - 1):
            self.pre_act[h] = self.biases[h] + np.matmul(np.transpose(self.weights[h]), self.hs[h-1])
            self.hs[h] = self.sigmoid(self.pre_act[h]) if not self.tanh else self.ftanh(self.pre_act[h])
        
        # Output layer
        self.pre_act[-1] = self.biases[-1] + np.matmul(np.transpose(self.weights[-1]), self.hs[-2])
        self.hs[-1] = self.softmax(self.pre_act[-1]) 
        
        # Return the training loss between the target and the output layer
        return self.training_loss(self.hs[-1], target)
    
    def training_loss(self,output,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """

        ## PUT CODE HERE ##

        return -np.log(output[target]) + self.L1 * self.L1_loss() + self.L2 * self.L2_loss()

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the hidden layers and output layer gradients in self.grad_hs
        - fills in the neural network gradients of weights and biases in self.grad_weights and self.grad_biases
        - returns nothing
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        ## PUT CODE HERE ##
#==============================================================================
#         Here the variable self.grad_hs actually contains the gradient for the preactivation
#         of hidden layers, not the activated hidden values
#==============================================================================

        # Last layer
        # one hot vector
        e_y = np.zeros([self.hs[-1].size,])
        e_y[target] = 1
        # preactivation gradient of last layer
        self.grad_hs[-1] = -(e_y - self.hs[-1])
        
        # Hidden layers
        for h in range(len(self.grad_hs)-1, 0, -1):
            # Calculate the gradient of weights and biases
            self.grad_weights[h] = np.matmul(self.hs[h-1][:, None], np.atleast_2d(self.grad_hs[h]))
            self.grad_biases[h] = self.grad_hs[h]
            
            #Calculate the gradients of hidden layer below
            grad_hid = np.matmul(self.weights[h], self.grad_hs[h])
            #Calculate the gradient of preactivation of hidden layer below
            self.grad_hs[h-1] = np.multiply(grad_hid, self.ftanh_deri(self.pre_act[h-1])) if (self.tanh == True) else np.multiply(grad_hid, self.sigmoid_deri(self.pre_act[h-1]))

        #Gradient of weights and biases for the first layer
        self.grad_weights[0] = np.matmul(input[:, None], np.atleast_2d(self.grad_hs[0]))
        self.grad_biases[0] = self.grad_hs[0]

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """

        ## PUT CODE HERE ##
        
        l1 = self.grad_L1()
        l2 = self.grad_L2()
        
        for h in range(len(self.weights)):
            self.weights[h] -= self.lr * self.grad_weights[h] + self.L1 * l1[h] + self.L2 * l2[h]
            self.biases[h] -= self.lr * self.grad_biases[h]
           
    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a Numpy 2D array of size
          len(dataset) by (nb of classes + 1)
        - the ith row of the array contains the outputs for the ith example
        - the outputs for each example should contain
          the predicted class (first element) and the
          output probabilities for each class (following elements)
        Argument ``dataset`` is an MLProblem object.
        """
 
        outputs = np.zeros((len(dataset),self.n_classes+1))

        ## PUT CODE HERE ##
        
        for i, [data, target] in enumerate(dataset):
            self.fprop(data, target)
            outputs[i, 0] = np.argmax(self.hs[-1])
            outputs[i, 1:] = self.hs[-1]
            
        return outputs
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of 
        those outputs for ``dataset``:
        - the errors should be a Numpy 2D array of size
          len(dataset) by 2
        - the ith row of the array contains the errors for the ith example
        - the errors for each example should contain 
          the 0/1 classification error (first element) and the 
          regularized negative log-likelihood (second element)
        Argument ``dataset`` is an MLProblem object.
        """
          
        outputs = self.use(dataset)
        errors = np.zeros((len(dataset),2))
        
        ## PUT CODE HERE ##
        
        for i, [input, target] in enumerate(dataset):
            errors[i][0] = 1 if outputs[1][0] == target else 0
            errors[i][1] = self.training_loss(outputs[i, 1:], target)
            
        return outputs, errors
 
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """
        
        print('WARNING: calling verify_gradients reinitializes the learner')
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.seed = 1234
        self.sizes = [4,5, 6]
        self.initialize(20,3)
        example = (rng.rand(20)<0.5,2)
        input,target = example
        epsilon=1e-6
        self.lr = 0.1
        self.decrease_constant = 0
  
        self.fprop(input,target)
        self.bprop(input,target) # compute gradients

        import copy
        emp_grad_weights = copy.deepcopy(self.weights)
  
        for h in range(len(self.weights)):
            for i in range(self.weights[h].shape[0]):
                for j in range(self.weights[h].shape[1]):
                    self.weights[h][i,j] += epsilon
                    a = self.fprop(input,target)
                    self.weights[h][i,j] -= epsilon
                    
                    self.weights[h][i,j] -= epsilon
                    b = self.fprop(input,target)
                    self.weights[h][i,j] += epsilon
                    
                    emp_grad_weights[h][i,j] = (a-b)/(2.*epsilon)
                    
            print 'grad_weights[',h,'] diff.:',np.sum(np.abs(self.grad_weights[h].ravel()-emp_grad_weights[h].ravel()))/self.weights[h].ravel().shape[0]
  
        emp_grad_biases = copy.deepcopy(self.biases)    
        for h in range(len(self.biases)):
            for i in range(self.biases[h].shape[0]):
                self.biases[h][i] += epsilon
                a = self.fprop(input,target)
                self.biases[h][i] -= epsilon
                
                self.biases[h][i] -= epsilon
                b = self.fprop(input,target)
                self.biases[h][i] += epsilon
                
                emp_grad_biases[h][i] = (a-b)/(2.*epsilon)
            print 'grad_biases[',h,'] diff.:',np.sum(np.abs(self.grad_biases[h].ravel()-emp_grad_biases[h].ravel()))/self.biases[h].ravel().shape[0]
        