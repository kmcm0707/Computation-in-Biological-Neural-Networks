#Stanford
# Lecture 4 - Black-box Meta-learning & In-context Learning
[[cs330_metalearning_bbox_2022.pdf]]
## One View on the Meta-Learning Problem
![[Pasted image 20241010230244.png]]
Reduces the meta-learning problem to the design to optimization of $f$.
Finn. Learning to Learn with Gradients. PhD Thesis. 2018 (@TODO)
$i$ = task, $j$ = index in task.
## General Recipe
![[Pasted image 20241010230609.png]]
## Black-box Adaptation Approaches
There is 2 versions.
### Running Example
![[Pasted image 20241010230929.png]]
![[Pasted image 20241010230957.png]]
Looking at a 3-way classification problem with a 1 shot learning problem.
### Black-Box Adaption
#### Method 1
Method to start:
1. Sample task (3 characters as 3 -way)
2. Sample 2 images per character (So 1 for train and 1 for test).
3. Assign consistent labels (labels when go back to 1).
Pass training data points into recurrent neural network (or transformers etc) and output set of parameters $\phi_i$. (So it's predicting a set of parameters for another neural network)
Then pass test examples into set of parameters network and check it, then use this loss to BPTT and update meta-parameters (not $\phi_i$).
Then redo over and over.
![[Pasted image 20241010231552.png]]
Notes:
- In training pass in label and data, in test only pass in data (want to find label).
- Test and train need to be different otherwise will learn to memorise.
Called a hypernetwork (@TODO) - a network that outputs the parameters of another network.
![[Pasted image 20241010233700.png]]
Note as a RNN order of images matters.
#### Method 2
![[Pasted image 20241010232047.png]]
Get the RNN to output a hidden vector instead.
$h_i$ is activation so doesn't get updated by BPTT only weights - $\theta_g$ is updated by BPTT.
More parameter sharing but can be unwieldy.
Feedforward is worse than RNN.
![[Pasted image 20241011001141.png]]
#### Meta-Test Time
![[Pasted image 20241010233306.png]]
### Black-Box Adaptation Architectures
![[Pasted image 20241011001724.png]]
NTM is outdated.
Another thing is to do feedforward and average - this is order invariant.
NTM and neural attentive are more relevant is neuroscience (@TODO).
### Black-Box Adaption Overall
![[Pasted image 20241011001938.png]]
### GPT-3
![[Pasted image 20241011002017.png]]
![[Pasted image 20241011002356.png]]
![[Pasted image 20241011002438.png]]
![[Pasted image 20241011002526.png]]
Source: https://github.com/shreyashankar/gpt3-sandbox/blob/master/docs/priming.md
## What is Needed for Few-shot Learning to Emerge?
![[Pasted image 20241011002625.png]]
