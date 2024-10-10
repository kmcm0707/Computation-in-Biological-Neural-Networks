#Stanford
# Lecture 3 - Transfer Learning + Start of Meta-Learning
## Transfer Learning
### Multi-Task Learning vs. Transfer Learning
Multi-task - solve multiple tasks at once.
Transfer - solve task b after solving task a by transferring knowledge from task a,
In transfer learning, can't access data a during transfer.
Transfer learning is a valid solution to multi-task learning.
![[Pasted image 20241010220832.png]]
### Transfer Learning via Fine-tuning
![[Pasted image 20241010220859.png]]
Where do you get the pre-trained parameters?
- ImageNet classification
- Models trained on large language corpora (BERT, LMs)
- Other unsupervised learning techniques
- Whatever large, diverse dataset you might have (typically for many gradient steps)
Common design choices
- Fine-tune with a smaller learning rate
- Smaller learning rate for earlier layers
- Freeze earlier layers, gradually unfreeze
- Reinitialise last layer
- Search over hyperparameters via cross-val
- Architecture choices matter (e.g. ResNets)
Pre-trained models often available online
### Fine-tuning Notes
Fine-tuning only last layer works well.
For fine tuning to low level image corruption - first layer may work well.
Recommend default: Train last layer, then fine tune entire network.
Regularisation to initial parameters may be useful.
Fine-tuning doesn’t work well with very small target task datasets. This is where meta-learning can help.
![[Pasted image 20241010222811.png]]
Here the pre train is unsupervised learning (was NLP so was possible).
(Unsupervised = Recreating input (@TODO - read more))
## Meta-Learning
### From Transfer Learning to Meta-Learning
Transfer learning: Initialize model. Hope that it helps the target task.
Meta-learning: Can we explicitly optimize for transferability?
![[Pasted image 20241010223044.png]]
### Two Ways to view Meta-learning Algorithms
![[Pasted image 20241010223106.png]]
### Directed Graphical Models (or Bayes Nets) (Probabilistic view)
Random variables (R.V.) represented as circle with dependencies represented as arrows.
Can see if things are independent - if path between variables they are independent.
![[Pasted image 20241010223505.png]]
![[Pasted image 20241010223654.png]]
The square is plate notation and means for all variables i (so in this case $y_i$ depends on $x_i$).
![[Pasted image 20241010223819.png|300]]
So in this case $j$ = datapoint, $i$ = task.
So in a task there is a set of tasks.
So $\theta$ is shared latent information between tasks - want to meta learn this.
![[Pasted image 20241010224145.png]]
### How Would Bayes view It? (Probabilistic view)
![[Pasted image 20241010224453.png|500]]
### Meta Learning Mechanistic View
Example:
![[Pasted image 20241010225137.png]]
![[Pasted image 20241010225317.png]]
#### Some Terminology
![[Pasted image 20241010225408.png]]
## Problem Settings Recap
![[Pasted image 20241010225436.png]]
