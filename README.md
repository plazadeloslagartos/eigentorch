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
Huang, Z., & Van Gool, L. (2016). A Riemannian Network for SPD Matrix Learning, 2036–2042.
https://doi.org/10.1109/CVPR.2014.132