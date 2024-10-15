#Stanford
# Lecture 5 - Optimization-Based Meta-Learning
[[cs330_optbased_metalearning_2022.pdf]]
## ~~Black-Box Adaption~~ Optimization-Based Adaptation
![[Pasted image 20241014111356.png]]
So basically use small amount of models to train network - one way is optimise for initial parameters of network.
## Recall: Fine-tuning
![[Pasted image 20241014111604.png]]
## Optimization-Based Adaptation for Fine Tuning
![[Pasted image 20241014111738.png]]
Finn, Abbeel, Levine. Model-Agnostic Meta-Learning. ICML 2017
![[Pasted image 20241014112147.png]]
Can optimise for other things like learning rate etc.
Difference with black box is in black box parameters are found by passing it through a neural network and this is by gradient descent.
![[Pasted image 20241014112555.png]]
Model-Agnostic Meta learning as as long as you can perform gradient descent it is agnostic to architecture of model.
![[Pasted image 20241014113348.png]]
Gradients are computed by difference equations. Hessians are calculated by by taking a gradient of gradient. Number of gradient steps define order of gradient used (e.g. use second optimization requires 3rd order gradient). More gradient steps mean more memory usage linearly - each additional gradient step means additional back propagation. Taking 5 is not too bad but 100 is very bad.
![[Pasted image 20241014113600.png]]
## Optimization vs. Black-Box Adaptation
![[Pasted image 20241014115227.png]]
(@TODO - VERY MUCH SO).
## Optimization vs. Black-Box Adaptation
![[Pasted image 20241014115748.png]]
Does embedding gradient descent come at a cost?
![[Pasted image 20241014115828.png]]
So no doesn't lose expressive power as long as model is deep and the other assumptions - the black box RNN approach may be more expressive for small datasets.
Reason is duplications in gradient.
## Optimization-Based Adaptation
![[Pasted image 20241014120029.png]]
![[Pasted image 20241014164903.png]]
![[Pasted image 20241014165225.png]]
## Summary
![[Pasted image 20241014165239.png]]
