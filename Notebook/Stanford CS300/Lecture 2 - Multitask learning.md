#Stanford
# Lecture 2 - Multitask Learning
[[cs330_multitask_transfer_2022.pdf]]
## Notation
![[Pasted image 20241008162049.png]]
## Examples of Tasks
![[Pasted image 20241008162633.png]]
## Task Descriptor
![[Pasted image 20241008162813.png]]
One Hot vector - $[1,0,0]$ or $[0,1,0]$ etc.
T = tasks.
Pass in a task descriptor into a network.
## Conditioning Questions
How should $z_i$ be passed into the model.
What parameters of the model should be shared?
### Conditioning on the Task
Assuming $z_i$ is a one-hot task index.
Question: How should you condition on the task in order to share as little as possible?
Reason: To get different neural networks for each task.
![[Pasted image 20241008163353.png]]
Basically just use a different neural network for each task.
(This is one extreme).
#### Other Extreme
![[Pasted image 20241008163458.png]]
Basically add $z_i$ to one input layer.
## An Alternative View on the Multi-Task Architecture
![[Pasted image 20241008163600.png]]
### Some Common Choices
#### Concatenation-based Conditioning and Additive Conditioning
Concatenation:
![[Pasted image 20241008163953.png]]
Additive conditioning:
![[Pasted image 20241008164033.png]]
But these are actually the same.
Reason:
![[Pasted image 20241008164335.png]]
#### Multi-head Architecture
![[Pasted image 20241008164354.png|200]]
Shared bottom layers.
#### Multiplicative Conditioning
![[Pasted image 20241008164431.png]]
![[Pasted image 20241008164441.png]]
The multiplication can sort of gate the network.
Where multiplication happens also sort of matters.
### Conditioning: More Complex Choices
![[Pasted image 20241008164536.png]]
### Conditioning Choices
But the conditioning choices are like normal neural network architecture tuning - problem dependent, largely guided by intuition or knowledge of the problem.
## Objective
How should the objective be formed?
![[Pasted image 20241008165228.png]]
Manually chosen weights is pretty good for a. approach.
DRO (Distributionally robust optimisation) is an extension on b.
## Optimization
How should the objective be optimized.
![[Pasted image 20241008170529.png]]
## Challenges
### Challenge #1: Negative Transfer
![[Pasted image 20241008170615.png]]
![[Pasted image 20241008170713.png]]
Soft parameter sharing is you have separate networks but have a constraint that encourages parameters to be similar.
### Challenge #2: Overfitting
Not sharing enough - share more.
Multitasking is a form of regularisation.
### Challenge #3: What if You Have a Lot of Tasks?
Bad news: No closed form solution for measuring task similarity and what tasks are complementary or if they are only complementary for parts of training (normally early on).
![[Pasted image 20241008171155.png]]
## Multi-Task Learning Recap
![[Pasted image 20241008171237.png]]
## Case Study
![[Pasted image 20241008171848.png]]
![[Pasted image 20241008171857.png]]
![[Pasted image 20241008171909.png]]
![[Pasted image 20241008171916.png]]
![[Pasted image 20241008171934.png]]
![[Pasted image 20241008172035.png]]
