#Stanford
# Lecture 6 - Non-Parametric Few-Shot Learning
Mainly for classification models - probably need to read up on embedding.
[[cs330_nonparametric_2022.pdf]]
Can we embed a learning procedure without a second-order optimization?
![[Pasted image 20241014170643.png]]
## Non-parametric Methods
![[Pasted image 20241014171739.png]]
$l_2$ doesn't work:
![[Pasted image 20241014171803.png]]
![[Pasted image 20241014172110.png]]
Label 1: If same class.
Siamese network - have 2 inputs and pass into exact same neural network.
![[Pasted image 20241014172238.png]]
Basically pass image into Siamese and compared to each image and check which one it thinks is closest.
But meta train and meta test are different as meta test time is all at once instead of one by one - meta test time equation: ![[Pasted image 20241014172553.png]]
So can instead backpropagate through this equation but get rid of $>0.5$ as can't differentiate: ![[Pasted image 20241014172648.png]]
Which is analogues to this:
![[Pasted image 20241014173503.png]]
Basically bit before the $X$ is embedding space and $X$ is multiple. (Bi-directional LSTM can be replaced by bi-directional transformer).
So basically meta-train and meta-test happen at same time.
![[Pasted image 20241014173707.png]]
![[Pasted image 20241014174345.png]]
Snell et al. Prototypical Networks, NeurIPS ‘17
![[Pasted image 20241014174518.png]]
## ProtoTransformer
![[Pasted image 20241014175149.png]]
![[Pasted image 20241014175200.png]]
![[Pasted image 20241014175438.png]]
## Black-box vs. Optimization vs. Non-Parametric
![[Pasted image 20241014180334.png]]
![[Pasted image 20241014180429.png]]
![[Pasted image 20241014180440.png]]
K = number of data points
![[Pasted image 20241014180457.png]]
## Application: One-Shot Imitation Learning
![[Pasted image 20241014181528.png]]
@TODO
## Application: Low-Resource Molecular Property Prediction
![[Pasted image 20241014181628.png]]
Only update last layer
## Application: Few-Shot Human Motion Prediction
![[Pasted image 20241014181659.png]]
@TODO
## Closing note for today
![[Pasted image 20241014181713.png]]
