
from mlpython.learners.generic import Learner
import numpy as np

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
        for h in range(n_hidden_layers):         
            self.hs += [np.zeros((self.sizes[h],))]       # hidden layer
            self.grad_hs += [np.zeros((self.sizes[h],))]  # ... and gradient
        self.hs += [np.zeros((self.n_classes,))]       # output layer
        self.grad_hs += [np.zeros((self.n_classes,))]  # ... and gradient
        
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

        raise NotImplementedError()
        
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
        
    def fprop(self,input,target):
        """
        Forward propagation: 
        - fills the hidden layers and output layer in self.hs
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()
        
    def training_loss(self,output,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()

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

        raise NotImplementedError()

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()
           
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

        raise NotImplementedError()
            
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

        raise NotImplementedError()
            
        return outputs, errors
 
    def verify_gradients(self):
        """
        Verifies the implementation of the fprop and bprop methods
        using a comparison with a finite difference approximation of
        the gradients.
        """
        
        print 'WARNING: calling verify_gradients reinitializes the learner'
  
        rng = np.random.mtrand.RandomState(1234)
  
        self.seed = 1234
        self.sizes = [4,5]
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


        print 'grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0]
        print 'grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0]
        print 'grad_weights[2] diff.:',np.sum(np.abs(self.grad_weights[2].ravel()-emp_grad_weights[2].ravel()))/self.weights[2].ravel().shape[0]
  
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

        print 'grad_biases[0] diff.:',np.sum(np.abs(self.grad_biases[0].ravel()-emp_grad_biases[0].ravel()))/self.biases[0].ravel().shape[0]
        print 'grad_biases[1] diff.:',np.sum(np.abs(self.grad_biases[1].ravel()-emp_grad_biases[1].ravel()))/self.biases[1].ravel().shape[0]
        print 'grad_biases[2] diff.:',np.sum(np.abs(self.grad_biases[2].ravel()-emp_grad_biases[2].ravel()))/self.biases[2].ravel().shape[0]

