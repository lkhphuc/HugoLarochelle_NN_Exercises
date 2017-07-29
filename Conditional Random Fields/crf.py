
from mlpython.learners.generic import Learner
import numpy as np
from scipy.misc import logsumexp

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
            self.lr *= self.dc # decrease the learning rate after each training set
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
        self.unary_log_factors_0 = np.dot(input, self.weights[0]) + self.bias #bias is broadcasted
        self.unary_log_factors_1 = np.dot(input, self.weights[-1]) #+ self.bias
        self.unary_log_factors_2 = np.dot(input, self.weights[1]) #+ self.bias
        
        self.unary_log_factors_0[1:,:] += self.unary_log_factors_1[:-1,:]
        self.unary_log_factors_0[:-1,:] += self.unary_log_factors_2[1:,:]
        
        # Calculate log_alpha, log_beta table        
        self.belief_propagation(input)
        
        # unary_log_factors is holding value for weights[0]
        self.target_unary_log_factors = self.unary_log_factors_0[np.arange(input.shape[0]), target]
        
        return self.training_loss(target, self.target_unary_log_factors, self.alpha, self.beta)
        

    def belief_propagation(self,input):
        """
        Returns the alpha/beta tables (i.e. the factor messages) using
        belief propagation (which is equivalent to forward-backward in HMMs).
        """
        ## PUT CODE HERE ##
        
        # Alpha/beta table will be a numpy 2D array, each row is a sequence,
        # and each column represent the class in each position of sequence
        
        ## LOG-ALPHA
        self.alpha = np.zeros((0,0))
        # Create the first row, with 0 column
        self.alpha = np.vstack((self.alpha, [[]]))
        # Add #number_of_classes columns
        self.alpha = np.hstack((self.alpha, np.zeros([1, self.n_classes])))
        
        # Initialize for alpha1
        for y2 in range(self.n_classes) :
            tmp = self.unary_log_factors_0[0] + self.lateral_weights[:,y2]
#            self.alpha[0][y2] += np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp))))
            self.alpha[0][y2] += logsumexp(tmp)
            
        alpha_tmp = np.zeros([1, self.n_classes])
        for k in range(1, input.shape[0]-1):
            for y2 in range(self.n_classes):
                tmp = self.unary_log_factors_0[k] + self.lateral_weights[:,y2] + self.alpha[k-1]
#                alpha_tmp[0][y2] = np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp))))
                alpha_tmp[0][y2] = logsumexp(tmp)
                
            self.alpha = np.vstack((self.alpha, alpha_tmp))
        
        
        ## LOG-BETA
        self.beta = np.zeros((0,0))
        # Create the first row, with 0 column
        self.beta = np.vstack((self.beta, [[]]))
        # Add #number_of_classes columns
        self.beta = np.hstack((self.beta, np.zeros([1, self.n_classes])))
        
        # Initialize for beta1
        for y1 in range(self.n_classes):
            tmp = self.unary_log_factors_0[-1] + self.lateral_weights[y1,:]
#            self.beta[0][y1] += np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp))))
            self.beta[0][y1] += logsumexp(tmp)
            
        beta_tmp = np.zeros((1, self.n_classes))            
        for k in range(input.shape[0]-2, 0, -1):
            for y1 in range(self.n_classes):
                # because beta table is insert from the bottom up, so referring 
                # to the previous one is simply the first one up to now
                tmp = self.unary_log_factors_0[k] + self.lateral_weights[y1,:] + self.beta[0]
#                beta_tmp[0][y1] = np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp)))) 
                beta_tmp[0][y1] = logsumexp(tmp)
                
            self.beta = np.vstack((beta_tmp, self.beta))
            
        ## PARTITION FUNCTION LOG-Z(X)
        tmp = self.unary_log_factors_0[-1] + self.alpha[-1]
#        self.Z_alpha = np.exp(np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp)))))
        self.Z_alpha = logsumexp(tmp)
        
        tmp = self.unary_log_factors_0[0] + self.beta[0]
#        self.Z_beta = np.exp(np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp)))))
        self.Z_beta = logsumexp(tmp)
        
    def L2_loss(self):
        return (np.sum(np.square(self.weights[0])) 
              + np.sum(np.square(self.weights[1])) 
              + np.sum(np.square(self.weights[-1])) )
    
    def L1_loss(self):
        return (np.sum(self.weights[0])
              + np.sum(self.weights[1]) 
              + np.sum(self.weights[-1]))
            
    def training_loss(self,target,target_unary_log_factors,alpha,beta):
        """
        Computation of the loss:
        - returns the regularized negative log-likelihood loss associated with the
          given the true target, the unary log factors of the target space and alpha/beta tables
        """

        ## PUT CODE HERE ##
        
        # Return negative log likelyhood;
        # -log(exp(A)/Z) = - (log(exp(A)) - log(Z)) = log(Z) - A
        return (self.Z_alpha - (np.sum(target_unary_log_factors) + np.sum(self.lateral_weights[target[:-1], target[1:]]))
               + self.L2 * self.L2_loss() 
               + self.L1 * self.L1_loss()) 

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

        # Compute all the marginal probabilities P(y-kth|X)
        # by calculating the pre-exp matrix of size [sequence size x n_classes]

        exp_matrix = np.zeros([input.shape[0], self.n_classes])
        
        for k in range(exp_matrix.shape[0]):
            for c in range(exp_matrix.shape[1]):
                exp_matrix[k][c] = self.unary_log_factors_0[k][c]
                if (k>0):
                    exp_matrix[k][c] += self.alpha[k-1][c]
                if (k+1 < exp_matrix.shape[0]):
                    # indexing !!!!!
                    exp_matrix[k][c] += self.beta[k][c]
                # THE VALUE INSIDE THE MATRIX ARE NOT YET EXPONENTIATED
#                exp_matrix[k][c] = np.exp(exp_matrix[k][c])
                
                
        # Marginal probabilites matrix
        P_y = np.zeros((input.shape[0], self.n_classes))
        for k in range(P_y.shape[0]):
            # P_y[k] = exp(A) / sum(exp(all_A) 
            #        =  exp(A) / exp(log(sum(exp(all_A))))
            #        = exp (A - logsumexp(all_A))
            
#            P_y[k] = exp_matrix[k] / np.sum(exp_matrix[k])
            P_y[k] = np.exp(exp_matrix[k] - logsumexp(exp_matrix[k]))
        
        # Output
        self.predict = np.argmax(P_y, axis=1)
        # One hot matrix
        E_y = np.zeros((input.shape[0], self.n_classes))
        for k in range(E_y.shape[0]):
            E_y[k][target[k]] = 1
            
        
        # grad_bias
        self.grad_bias = np.sum(-(E_y - P_y), axis=0)
        # grad_weights
        self.grad_weights[0] = np.dot(np.transpose(input), -(E_y - P_y))
        self.grad_weights[-1] = np.dot(np.transpose(input[:-1, :]), -(E_y - P_y)[1:, :])
        self.grad_weights[1] = np.dot(np.transpose(input[1:, :]), -(E_y - P_y)[:-1, :])
        # L2 regularization
        self.grad_weights[0] += self.L2 * 2 * self.weights[0]
        self.grad_weights[-1] += self.L2 * 2 * self.weights[-1]
        self.grad_weights[1] += self.L2 * 2 * self.weights[1]
        # L1 regularization
        self.grad_weights[0] += self.L1 * np.sign(self.weights[0])
        self.grad_weights[-1] += self.L1 * np.sign(self.weights[-1])
        self.grad_weights[1] += self.L1 * np.sign(self.weights[1])
        
        # Compute all the marginal probabilites P(y-k,y+1|X) by calculating 
        # the 3D matrix of size [sequence size x [n_classes x n_classes]]
        exp_matrix_3D = np.zeros((input.shape[0], self.n_classes, self.n_classes))

                            
        for k in range(exp_matrix_3D.shape[0]-1):
            for c1 in range(exp_matrix_3D.shape[1]):
                for c2 in range(exp_matrix_3D.shape[2]):
                    exp_matrix_3D[k][c1][c2] = self.unary_log_factors_0[k][c1] + self.lateral_weights[c1][c2] 
                    if (k>0):
                        exp_matrix_3D[k][c1][c2] += self.alpha[k-1][c1]
                    if (k+1 < input.shape[0]):
                        exp_matrix_3D[k][c1][c2] += self.unary_log_factors_0[k+1][c2]
                        if (k+2< input.shape[0]):
                            exp_matrix_3D[k][c1][c2] += self.beta[k+1][c2]
                    # THE VALUES INSIDE MATRIX ARE NOT YET EXPONENTIATED
#                    exp_matrix_3D[k][c1][c2] = np.exp(exp_matrix_3D[k][c1][c2])
                    
        # Marginal probabilities 3D matrix
        P_y_y1 = np.zeros((input.shape[0], self.n_classes, self.n_classes))
        for k in range(P_y_y1.shape[0]):
#            P_y_y1[k] = exp_matrix_3D[k] / np.sum(exp_matrix[k])
            # P_y_Y1[k] = exp(A) / sum(exp(all_A) 
            #           =  exp(A) / exp(log(sum(exp(all_A))))
            #            = exp (A - logsumexp(all_A))
            P_y_y1[k] = np.exp(exp_matrix_3D[k] - logsumexp(exp_matrix_3D[k]))
        
        # Summing over k=1 to K-1
        for k in range(input.shape[0]-1):
            E_y_y1 = np.zeros((self.n_classes, self.n_classes))
            E_y_y1[target[k], target[k+1]] = 1
            self.grad_lateral_weights += -(E_y_y1 - P_y_y1[k])

    def update(self):
        """
        Stochastic gradient update:
        - performs a gradient step update of the CRF parameters self.weights,
          self.lateral_weights and self.bias, using the gradients in 
          self.grad_weights, self.grad_lateral_weights and self.grad_bias
        """

        ## PUT CODE HERE ##

        self.bias -= self.lr * self.grad_bias
        self.weights[0] -= self.lr * self.grad_weights[0]
        self.weights[1] -= self.lr * self.grad_weights[1]
        self.weights[-1] -= self.lr * self.grad_weights[-1]
        self.lateral_weights -= self.lr * self.grad_lateral_weights
           
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

        for input, target in dataset:
            self.fprop(input, target)
            self.bprop(input, target)
            outputs += [self.predict]
            
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
        for k, (input, target) in enumerate(dataset):
            nll = self.fprop(input, target)
            classif_errors = 1 * (outputs[k] == target)
            errors += [(classif_errors, nll)]
            
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

