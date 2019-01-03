# deep-belief-network

**DeepBeliefNetwork**: straightforward implementation of a restricted Boltzman machine, and deep belief network. It contains the following:
  * sigmoid: our chosen activation function for this implementation,
  * init_model: initializes a single restricted Boltzman machine (i.e. create a random weight matrix and two biais),
  * sample: sample a vector of 0's and 1's with respect to a prescribed probability array,
  * gibbs_step: perform one Gibbs step,
  * gibbs_sample: do a Gibbs sample,
  * train_step: compute the outcome Delta_W, Delta_bh, Delta_bv using contrastive divergence, 
  * vis_to_hid and hid_to_vis: compute hidden to visible and visible to hidden activities,
  * phony data: creates phony data, just to check consistency of the above code,
  * producing eights: consider the 'eights' from mnist_small.csv (see below) and optimizes an RBM for this dataset,
  * small grid search on the best possible architecture for producing eights,
  * stacking RBM to get a DBN (possibly for later use, see 'Deep Belief Nets' by G. Hinton. 
  
**mnist_small.csv**: a set of 10000 examples from the mnist dataset.
