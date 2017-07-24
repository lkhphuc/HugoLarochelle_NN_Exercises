
from mlpython.learners.generic import Learner
import numpy as np

class LinearChainCRF(Learner):
    """
    Linear chain conditional random field. The contex window size
    has a radius of 1.
 
    Option ``lr`` is the learning rate.
 
    Option ``dc`` is the decrease constante for the learning rate.
 
    Option ``L2`` is the L2 regularization weight (weight decay).
 
    Option ``L1`` is the L1 regularization weight (weight decay).
 
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``:    Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=1e-10,
                 L2=0.001,
                 L1=0,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.L2=L2
        self.L1=L1
        self.n_epochs=n_epochs

        # internal variable keeping track of the number of training iterations since initialization
        self.epoch = 0 

    def initialize(self,input_size,n_classes):
        """
        This method allocates memory for the fprop/bprop computations
        and initializes the parameters of the CRF to 0 (DONE)
        """

        self.n_classes = n_classes
        self.input_size = input_size

        # Can't allocate space for the alpha/beta tables of
        # belief propagation (forward-backward), since their size
        # depends on the input sequence size, which will change from
        # one example to another.

        self.alpha = np.zeros((0,0))
        self.beta = np.zeros((0,0))
        
        ###########################################
        # Allocate space for the linear chain CRF #
        ###########################################
        # - self.weights[0] are the connections with the image at the current position
        # - self.weights[-1] are the connections with the image on the left of the current position
        # - self.weights[1] are the connections with the image on the right of the current position
        self.weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        # - self.bias is the bias vector of the output at the current position
        self.bias = np.zeros((self.n_classes))

        # - self.lateral_weights are the linear chain connections between target at adjacent positions
        self.lateral_weights = np.zeros((self.n_classes,self.n_classes))
        
        self.grad_weights = [np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes)),
                        np.zeros((self.input_size,self.n_classes))]
        self.grad_bias = np.zeros((self.n_classes))
        self.grad_lateral_weights = np.zeros((self.n_classes,self.n_classes))
                    
        #########################
        # Initialize parameters #
        #########################

        # Since the CRF log factors are linear in the parameters,
        # the optimization is convex and there's no need to use a random
        # initialization.

        self.n_updates = 0 # To keep track of the number of updates, to decrease the learning rate

    def forget(self):
        """
        Resets the neural network to its original state (DONE)
        """
        self.initialize(self.input_size,self.n_classes)
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
        - computes the value of the unary log factors for the target given the input (the field
          self.target_unary_log_factors should be assigned accordingly)
        - computes the alpha and beta tables using the belief propagation (forward-backward) 
          algorithm for linear chain CRF (the field ``self.alpha`` and ``self.beta`` 
          should be allocated and filled accordingly)
        - returns the training loss, i.e. the 
          regularized negative log-likelihood for this (``input``,``target``) pair
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """

        ## PUT CODE HERE ##
        # (your code should call belief_propagation and training_loss)
        
        raise NotImplementedError()

    def belief_propagation(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """
        ## PUT CODE HERE ##

        raise NotImplementedError()
    
    def training_loss(self,target,target_unary_log_factors,alpha,beta):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given the true target, the unary log factors of the target space and alpha/beta tables
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the CRF gradients of the weights, lateral weights and bias 
          in self.grad_weights, self.grad_lateral_weights and self.grad_bias
        - returns nothing
        Argument ``input`` is a Numpy 2D array where the number of
        rows if the sequence size and the number of columns is the
        input size. 
        Argument ``target`` is a Numpy 1D array of integers between 
        0 and nb. of classe - 1. Its size is the same as the number of
        rows of argument ``input``.
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the CRF parameters self.weights,
          self.lateral_weights and self.bias, using the gradients in 
          self.grad_weights, self.grad_lateral_weights and self.grad_bias
        """

        ## PUT CODE HERE ##

        raise NotImplementedError()
           
    def use(self,dataset):
        """
        Computes and returns the outputs of the Learner for
        ``dataset``:
        - the outputs should be a list of size ``len(dataset)``, containing
          a Numpy 1D array that gives the class prediction for each position
          in the sequence, for each input sequence in ``dataset``
        Argument ``dataset`` is an MLProblem object.
        """
 
        ## PUT CODE HERE ##

        outputs = []

        raise NotImplementedError()
            
        return outputs
        
    def test(self,dataset):
        """
        Computes and returns the outputs of the Learner as well as the errors of the
        CRF for ``dataset``:
        - the errors should be a list of size ``len(dataset)``, containing
          a pair ``(classif_errors,nll)`` for each examples in ``dataset``, where 
            - ``classif_errors`` is a Numpy 1D array that gives the class prediction error 
              (0/1) at each position in the sequence
            - ``nll`` is a positive float giving the regularized negative log-likelihood of the target given
              the input sequence
        Argument ``dataset`` is an MLProblem object.
        """
        
        outputs = self.use(dataset)
        errors = []

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
  
        self.initialize(10,3)
        example = (rng.rand(4,10),np.array([0,1,1,2]))
        input,target = example
        epsilon=1e-6
        self.lr = 0.1
        self.decrease_constant = 0

        self.weights = [0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes),
                        0.01*rng.rand(self.input_size,self.n_classes)]
        self.bias = 0.01*rng.rand(self.n_classes)
        self.lateral_weights = 0.01*rng.rand(self.n_classes,self.n_classes)
        
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


        print 'grad_weights[-1] diff.:',np.sum(np.abs(self.grad_weights[-1].ravel()-emp_grad_weights[-1].ravel()))/self.weights[-1].ravel().shape[0]
        print 'grad_weights[0] diff.:',np.sum(np.abs(self.grad_weights[0].ravel()-emp_grad_weights[0].ravel()))/self.weights[0].ravel().shape[0]
        print 'grad_weights[1] diff.:',np.sum(np.abs(self.grad_weights[1].ravel()-emp_grad_weights[1].ravel()))/self.weights[1].ravel().shape[0]
  
        emp_grad_lateral_weights = copy.deepcopy(self.lateral_weights)
  
        for i in range(self.lateral_weights.shape[0]):
            for j in range(self.lateral_weights.shape[1]):
                self.lateral_weights[i,j] += epsilon
                a = self.fprop(input,target)
                self.lateral_weights[i,j] -= epsilon

                self.lateral_weights[i,j] -= epsilon
                b = self.fprop(input,target)
                self.lateral_weights[i,j] += epsilon
                
                emp_grad_lateral_weights[i,j] = (a-b)/(2.*epsilon)


        print 'grad_lateral_weights diff.:',np.sum(np.abs(self.grad_lateral_weights.ravel()-emp_grad_lateral_weights.ravel()))/self.lateral_weights.ravel().shape[0]

        emp_grad_bias = copy.deepcopy(self.bias)
        for i in range(self.bias.shape[0]):
            self.bias[i] += epsilon
            a = self.fprop(input,target)
            self.bias[i] -= epsilon
            
            self.bias[i] -= epsilon
            b = self.fprop(input,target)
            self.bias[i] += epsilon
            
            emp_grad_bias[i] = (a-b)/(2.*epsilon)

        print 'grad_bias diff.:',np.sum(np.abs(self.grad_bias.ravel()-emp_grad_bias.ravel()))/self.bias.ravel().shape[0]

