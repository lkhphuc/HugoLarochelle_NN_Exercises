
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

    Option ``parameter_initialization`` is a pair of lists,
    giving the initializations for the biases (first list)
    and the weight matrices (second list). If ``None``,
    then a random initialization is used.
    
    Option ``n_epochs`` number of training epochs.
 
    **Required metadata:**
 
    * ``'input_size'``: Size of the input.
    * ``'targets'``: Set of possible targets.
 
    """
    
    def __init__(self,
                 lr=0.001,
                 dc=0,
                 sizes=[200,100,50],
                 L2=0,
                 L1=0,
                 seed=1234,
                 tanh=False,
                 parameter_initialization=None,
                 n_epochs=10):
        self.lr=lr
        self.dc=dc
        self.sizes=sizes
        self.L2=L2
        self.L1=L1
        self.seed=seed
        self.tanh=tanh
        self.parameter_initialization = parameter_initialization
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

        if self.parameter_initialization is not None:
            self.biases = [ b.copy() for b in self.parameter_initialization[0] ]
            self.weights = [ w.copy() for w in self.parameter_initialization[1] ]

            if len(self.biases) != n_hidden_layers + 1:
                raise ValueError("Biases provided for initialization are not compatible")
            for i,b in enumerate(self.biases):
                if i == n_hidden_layers and len(b.shape) == 1 and b.shape[0] != self.n_classes:
                    raise ValueError("Biases provided for initialization are not of the expected size")
                if i < n_hidden_layers and len(b.shape) == 1 and b.shape[0] != self.sizes[i]:
                    raise ValueError("Biases provided for initialization are not of the expected size")

            if len(self.weights) != n_hidden_layers+1:
                raise ValueError("Weights provided for initialization are not compatible")
            for i,w in enumerate(self.weights):
                if i == n_hidden_layers and len(w.shape) == 2 and w.shape != (self.sizes[-1],self.n_classes):
                    raise ValueError("Weights provided for initialization are not of the expected size")
                if i == 0  and len(w.shape) == 2 and w.shape != (self.input_size,self.sizes[i]):
                    raise ValueError("Weights provided for initialization are not of the expected size")
                if i < n_hidden_layers and i > 0 and len(w.shape) == 2 and w.shape != (self.sizes[i-1],self.sizes[i]):
                    raise ValueError("Weights provided for initialization are not of the expected size")

        else:
            self.weights = [(2*self.rng.rand(self.input_size,self.sizes[0])-1)/self.input_size]       # input to 1st hidden layer weights
            
            for h in range(1,n_hidden_layers):
                self.weights += [(2*self.rng.rand(self.sizes[h-1],self.sizes[h])-1)/self.sizes[h-1]]        # h-1 to h hidden layer weights
            
            self.weights += [(2*self.rng.rand(self.sizes[-1],self.n_classes)-1)/self.sizes[-1]]      # last hidden to output layer weights
        
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
          regularized negative log-likelihood for this (input,target) pair
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        if self.tanh:
            def compute_h(input,weights,bias):
                exptwoact = np.exp(-2*(bias+np.dot(input,weights)))
                return (1-exptwoact)/(exptwoact+1)
        else:
            def compute_h(input,weights,bias):
                act = bias+np.dot(input,weights)
                return 1./(1+np.exp(-act))
                
        self.hs[0][:] = compute_h(input,self.weights[0],self.biases[0])        
        for h in range(1,len(self.weights)-1):
            self.hs[h][:] = compute_h(self.hs[h-1],self.weights[h],self.biases[h])

        def softmax(act):
            act = act-act.max()
            expact = np.exp(act)
            return expact/expact.sum()
            
        self.hs[-1][:] = softmax(self.biases[-1]+np.dot(self.hs[-2],self.weights[-1]))

        return self.training_loss(self.hs[-1],target)

    def training_loss(self,output,target):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given output vector (probabilities of each class) and target (class ID)
        """

        l = -np.log(output[target])
        if self.L2 != 0:
            for h in range(len(self.weights)):
                l += self.L2*(self.weights[h]**2).sum()

        if self.L1 != 0:
            for h in range(len(self.weights)):
                l += self.L1*(np.abs(self.weights[h])).sum()
            
        return l

    def bprop(self,input,target):
        """
        Backpropagation:
        - fills in the hidden layers and output layer gradients in self.grad_hs
        - fills in the neural network gradients of weights and biases in self.grad_weights and self.grad_biases
        - returns nothing
        Argument ``input`` is a Numpy 1D array and ``target`` is an
        integer between 0 and nb. of classe - 1.
        """

        if self.tanh:
            def compute_grad_h(input,weights,bias,h,grad_weights,grad_bias,grad_h):
                grad_act = (1-h**2) * grad_h
                grad_weights[:,:] = np.outer(input,grad_act) 
                grad_bias[:] = grad_act
                return np.dot(grad_act,weights.T)
        else:
            def compute_grad_h(input,weights,bias,h,grad_weights,grad_bias,grad_h):
                grad_act = h*(1-h)*grad_h
                grad_weights[:,:] = np.outer(input,grad_act) 
                grad_bias[:] = grad_act
                return np.dot(grad_act,weights.T)

        # Output layer gradients
        self.grad_hs[-1][:] = self.hs[-1]
        self.grad_hs[-1][target] -= 1
        self.grad_weights[-1][:,:] = np.outer(self.hs[-2],self.grad_hs[-1])
        self.grad_biases[-1][:] = self.grad_hs[-1]
        self.grad_hs[-2][:] = np.dot(self.grad_hs[-1],self.weights[-1].T)
        
        # Hidden layer gradients
        for h in range(len(self.weights)-2,0,-1):
            self.grad_hs[h-1][:] = compute_grad_h(self.hs[h-1],self.weights[h],self.biases[h],
                                                  self.hs[h],self.grad_weights[h],self.grad_biases[h],
                                                  self.grad_hs[h])

        compute_grad_h(input,self.weights[0],self.biases[0],
                       self.hs[0],self.grad_weights[0],self.grad_biases[0],self.grad_hs[0])

        # Add regularization gradients
        for h in range(len(self.weights)):
            if self.L2 != 0:
                self.grad_weights[h] += 2*self.L2*self.weights[h]
            if self.L1 != 0:
                self.grad_weights[h] += self.L1*np.sign(self.weights[h])

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the neural network parameters self.weights and self.biases,
          using the gradients in self.grad_weights and self.grad_biases
        """

        for h in range(len(self.weights)):
            self.weights[h] -= self.lr /(1.+self.n_updates*self.dc) * self.grad_weights[h]
            self.biases[h] -= self.lr /(1.+self.n_updates*self.dc) * self.grad_biases[h]

        self.n_updates += 1
            
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
        """
 
        outputs = np.zeros((len(dataset),self.n_classes+1))

        t=0
        for input,target in dataset:
            self.fprop(input,target)
            outputs[t,0] = self.hs[-1].argmax()
            outputs[t,1:] = self.hs[-1]
            t+=1
            
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
         """
          
        outputs = self.use(dataset)
        errors = np.zeros((len(dataset),2))
        
        t=0
        for input,target in dataset:
            output = outputs[t,:]
            errors[t,0] = output[0] != target
            errors[t,1] = self.training_loss(output[1:],target)
            t+=1
            
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

