import numpy as np
import mlpython.learners.generic as mlgen 
import mlpython.learners.classification as mlclass
import mlpython.mlproblems.generic as mlpb 
import mlpython.mlproblems.classification as mlpbclass


class Autoencoder(mlgen.Learner):
    """
    Autoencoder trained with unsupervised learning.

    Option ``lr`` is the learning rate.

    Option ``hidden_size`` is the size of the hidden layer.

    Option ``noise_prob`` is the noise or corruption probability of
    setting each input to 0.

    Option ``seed`` is the seed of the random number generator.
    
    Option ``n_epochs`` number of training epochs.
    """
    
    def __init__(self, 
                 lr,              # learning rate
                 hidden_size,     # hidden layer size
                 noise_prob=0.1,  # probability of setting an input to 0
                 seed=1234,       # seed for random number generator
                 n_epochs=10      # nb. of training iterations
                 ):
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.lr = lr
        self.noise_prob = noise_prob
        self.seed = seed
        self.rng = np.random.mtrand.RandomState(self.seed)   # create random number generator
        
    def train(self,trainset):
        """
        Train autoencoder for ``self.n_epochs`` iterations.
        Use ``self.W.T`` as the output weight matrix (i.e. use tied weights).
        """
        # Initialize parameters
        input_size = trainset.metadata['input_size']
        self.W = (self.rng.rand(input_size,self.hidden_size)-0.5)/(max(input_size,self.hidden_size))
        self.b = np.zeros((self.hidden_size,))
        self.c = np.zeros((input_size,))
        
        for it in range(self.n_epochs):
            for input in trainset:

                # fprop
                "PUT CODE HERE"

                # bprop
                "PUT CODE HERE"
                
                # Updating the parameters
                "PUT CODE HERE"

    def show_filters(self):
        from matplotlib.pylab import show, draw, ion
        import mlpython.misc.visualize as mlvis
        mlvis.show_filters(0.5*self.W.T,
                           200,
                           16,
                           8,
                           10,20,2)
        show()
