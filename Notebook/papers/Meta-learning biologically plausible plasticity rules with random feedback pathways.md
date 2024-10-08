---
category: literaturenote
tags: Computational science, Learning algorithms paper
citekey: shervani-tabarMetalearningBiologicallyPlausible2023
---
# Meta-learning Biologically Plausible Plasticity Rules with Random Feedback Pathways
> [!Cite]
> Shervani-Tabar, N., & Rosenbaum, R. (2023). Meta-learning biologically plausible plasticity rules with random feedback pathways. _Nature Communications_, _14_(1), 1805. [https://doi.org/10.1038/s41467-023-37562-1](https://doi.org/10.1038/s41467-023-37562-1)

> [!md]
> **First Author**::Shervani-Tabar, Navid > **Author**::Rosenbaum, Robert 
> 
> **Title**:: Meta-learning biologically plausible plasticity rules with random feedback pathways
> **Year**:: 2023
> **Citekey**:: shervani-tabarMetalearningBiologicallyPlausible2023 
> **itemType**:: journalArticle
> **Journal**:: _Nature Communications_ 
> **Volume**:: 14 
> **Issue**:: 1 

> [!LINK] 
> 
> [Full Text PDF](file://C:\Users\Kyle\Zotero\storage\E6MFRN9F\Shervani-Tabar%20and%20Rosenbaum%20-%202023%20-%20Meta-learning%20biologically%20plausible%20plasticity%20rules%20with%20random%20feedback%20pathways.pdf) 
> Obsidian Link: [[Shervani-Tabar and Rosenbaum - 2023 - Meta-learning biologically plausible plasticity rules with random feedback pathways.pdf]]

> [!Abstract]
> 
> Backpropagation is widely used to train artificial neural networks, but its relationship to synaptic plasticity in the brain is unknown. Some biological models of backpropagation rely on feedback projections that are symmetric with feedforward connections, but experiments do not corroborate the existence of such symmetric backward connectivity. Random feedback alignment offers an alternative model in which errors are propagated backward through fixed, random backward connections. This approach successfully trains shallow models, but learns slowly and does not perform well with deeper models or online learning. In this study, we develop a meta-learning approach to discover interpretable, biologically plausible plasticity rules that improve online learning performance with fixed random feedback connections. The resulting plasticity rules show improved online training of deep models in the low data regime. Our results highlight the potential of meta-learning to discover effective, interpretable learning rules satisfying biological constraints.
> >  

## Notes
### Intro
Backprop is biologically implausible.
Weight transport problem - transmitting gradients to upstream layers requires symmetric feedback connections. These don't exist in reality.
Lillicrap (@TODO) shows random backwards connections can work but can't train deep networks.
Nøkland (@TODO) proposes rewiring feedback connections and sending teaching signals directly from output to upstream layers - better but still not as good as symmetric in low data.
Liao (@TODO) tries dismissing symmetry in magnitude but assigning symmetric signs to feedback connections. But decreasing batch size is bad and batch normalisation required - doesn't work for online stream where batch = 1 - so no biological plausibility.

Alternatively add a secondary update rule.
- Akrout (@TODO) says use a Hebbian plasticity rule (@TODO) - adjust feedback matrices parallel (so that it's the transpose) to approximate gradient update of forward path.
- Kunin (@TODO) says very sensitive to hyper parameter tuning - redefines objective as loss function based on forward path and regularisation terms to update forward and backwards concurrently - shows more stable.

Meta-learning - framework of learning process around an optimisation loop and learning some aspect of the inner procedure (Lindsey @TODO is an example).
Issues with a variety of papers - meta learning weight initialization obscures whether learning rule is working and meta learning for per weight is confusing.

**So want to use meta-learning for all weights without initialisation.** 
Early work:
- Bengio (@TODO)
- **Andrychowicz (@TODO)** - parametrize the learning rule with a Recurrent Neural Network (RNN) and meta-learn weights of the RNN model.
- **Confavreux (@TODO)** - Bio

Metz (unsupervised Meta) (@TODO)
### Method
1. No weight initialization.
2. Meta-parameter sharing (same for all weights).
3. L1 penalty on the plasticity coefficients in our meta-loss function. (stops too many coefficients).
4. Inner learning loop = online learning (batch = 1) and small data points.
### Results
Goal is to find set of weights that minimises a loss function and weights are updated by teaching signal derived from that loss function. - e.g. BPTT is one teaching signal.
#### Limitations of Feedback Alignment in Deep Networks
Random Feedback Alignment uses fixed random backward connections that are not bound to the forward weights (so not transpose as in BPTT).
For feedback alignment, the teaching signal is not an exact gradient, but an approximating pseudo-gradient term.
Well on simple tasks and shallow networks.
![[Pasted image 20240922024114.png]]
But bad on little data and large networks.
#### Meta-learning to Discover Interpretable Plasticity Rules
Meta learning framework consists of a two-level learning scheme: An inner adaptation loop that learns parameters of a model using a parameterized plasticity rule and an outer meta-optimization loop that modifies the plasticity meta-parameters.
The meta-training dataset contains a set of tasks consisting of training data and query data per class.
Training data trains model while query data trains meta parameters.

Algorithm in paper
Key details:
- In each episode new random initialized model is trained using online data sequence. (Random initialisation removes dependence on start).
- Then meta-loss evaluated by using model on query data of same task.
- Weights updated using meta parameters, meta parameters updated by gradient approach.
![[Pasted image 20240922145253.png]]
#### Benchmarking Backprop and Feedback Alignment via Meta-learning
![[Pasted image 20240922145554.png]]
#### Biologically Plausible Plasticity Rules
Combine 10 plasticity terms in a linear combination - $F(\theta)=\sum^{R-1}_{r=0}\theta_{r}F^r$.
![[Pasted image 20240922162931.png]]
Becomes as good as BPTT.
Only pseudo-gradient ($F_{0}$), Hebbian plasticity ($F_{2}$) and Oja's rule ($F_{9}$) don't converge to 0.
![[Pasted image 20240922154323.png]]
#### Hebbian-style Error-based Plasticity Rule
Using only Hebbian and pseudo-gradient.
Much better than just pseudo-gradient but worse than bio.
Teaching signals more aligned with BPTT than pseudo-gradient.
Hebbian terms explicitly transfers information from feedback to previous weight in one iteration, while pseudo-gradient takes 2.
Maths - Hebbian pushes weight towards transpose of feedback - like in BPTT.
#### Oja's Rule
A purely local learning rule that updates the weights based on its current state and the local activations in the forward path.
Doesn't align as much as Hebbian.
Instead Oja's rule helps by circumventing the backwards path.
Oja’s rule implements a Hebbian learning rule subjected to an orthonormality constraint on the weights.
Basically Oja's rule if used recursively is like PCA.
But even when not - it reduces correlation between weight rows and improves feature extraction.

### Discussion
Weight updates depend on forward pass and error signals from backwards pass.
So as use error signal some disagreement if local.

Hebbian may at best match BPTT as wasn't to align weights but Oja's doesn't so can be used to enhance learning.

Considered all quadratic combinations of errors and activations.
Except pure Hebbian plasticity ($yy^t$) is replaced with Oja's rule as causes blow-up of activations.
Terms ($F^{1,2,4,6,7,8}$) requires pre-synaptic error for first layer: $$e_{0}=B_{1,0}e_{1}(1-\exp(-\beta y_{0}))$$
$B$ is set of feedback connections.

#### Meta-training
Optimisation method is gradient (though alternatives like evolution could be used).

In the meta-optimization phase, this gradient-based optimizer differentiates through the unrolled computational graph of the adaptation phase. Thus, the non-linear layers are double differentiated, once to compute $e_L$ and a second time by the meta-optimizer. This only allows a two-times differentiable non-linear layer, which prohibits using the Rectified Linear Unit, ReLU, as the activation function σ. Instead, we use the softplus function a continuous, twice-differentiable approximation of the ReLU function.

L1 norm is not differentiable but in pyTorch is defined as 0;.
