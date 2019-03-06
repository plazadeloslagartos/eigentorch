# eigentorch
Eigentorch is a simple extension of Pytorch which aims to provide a neural network layer
model (SPDNet) for the processing of features formed by symmetric-positive defininte (SPD)
matrices while performing all weight updates on the underlying Riemannian
geometry.  SPD matrices represent a special Lie group which describes it's own differential
geometry upon which Riemannian metrics can be calculated.  By constraining gradient calculation, and thus weight
updates to this underlying geometry, features can therefore be manipulated via forward and backward propogation
without distortion of the true underlying feature space.  The network described in this project provides 
mechanisms for the creation (via forward propogation) of new SPD features (via multiple filters) while 
simulataneously reducing matrix rank.  This is intentionally similar to CNN architecture, except without the need
for additional pooling.  It also provides a mechanism for eigenvalue based regularization ensuring that resultant 
SPD matrices don't approach becoming singular.  Furthermore it provides for projection (via a Riemmanian metric),
of resultant features from the manifold onto a flattened space which can be discriminated by Euclidean methods. 
This allows for classification of the resultant features in Euclidean space via customary methods such as an 
MLP layer.

This project is inspired by the methodology described in the journal paper below:

_Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132_

### modules
* __eigenfunctions.py:__ This provides the custom autograd functions necessary to implement the riemannian math
    * _BiMap_: Transforms an SPD matrix into another SPD matrix.  This is the only function with parameters.
    Parameters are assumed to be points on Stiefel Manifold and the gradients returned are referenced to the corresponding
    tangent space.  Therefore, a custom optimizer is necessary to project the gradients back down to the manifold.
    * _ReEig_:  Inspired by ReLu, this performs SPD regularization by thresholding eigenvalues against a minimum value.
    * _LogEig_:  Performs LogEuclidean metric calculation which projects an SPD matrix into a flattened representation of the 
    underlying manifold.
     
* __eigenoptim.py:__ This provides the custom optimizer object which updates weights for the BiMap layer on the 
                    underlying semi-orthogonal Stiefel manifold.
* __spdnn.py__: This provides the SpdNet Layer to be use in Pytorch networks. Parameters for BiMap are of instance 
                _StiefelParameter_ which is subclassed from _nn.Parameter_.  
 