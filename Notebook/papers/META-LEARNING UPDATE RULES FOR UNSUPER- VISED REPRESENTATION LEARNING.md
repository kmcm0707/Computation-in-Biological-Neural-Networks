---
category: literaturenote
tags:  paper
citekey: metzMETALEARNINGUPDATERULES2019
---
# META-LEARNING UPDATE RULES FOR UNSUPER- VISED REPRESENTATION LEARNING
> [!Cite]
> Metz, L., Maheswaranathan, N., Cheung, B., & Sohl-Dickstein, J. (2019). _META-LEARNING UPDATE RULES FOR UNSUPER- VISED REPRESENTATION LEARNING_.

>[!md]
> **First Author**::Metz, Luke > **Author**::Maheswaranathan, Niru > **Author**::Cheung, Brian > **Author**::Sohl-Dickstein, Jascha 
> 
> **Title**:: META-LEARNING UPDATE RULES FOR UNSUPER- VISED REPRESENTATION LEARNING
> **Year**:: 2019
> **Citekey**:: metzMETALEARNINGUPDATERULES2019 
> **itemType**:: journalArticle
> **Journal**:: ** 

> [!LINK] 
> 
>  [PDF](file://C:\Users\Kyle\Zotero\storage\L86ZJU64\Metz%20et%20al.%20-%202019%20-%20META-LEARNING%20UPDATE%20RULES%20FOR%20UNSUPER-%20VISED%20REPRESENTATION%20LEARNING.pdf)  >  Obsidian Link: @TODO

> [!Abstract]
> 
> A major goal of unsupervised learning is to discover data representations that are useful for subsequent tasks, without access to supervised labels during training. Typically, this involves minimizing a surrogate objective, such as the negative log likelihood of a generative model, with the hope that representations useful for subsequent tasks will arise as a side effect. In this work, we propose instead to directly target later desired tasks by meta-learning an unsupervised learning rule which leads to representations useful for those tasks. Speciﬁcally, we target semi-supervised classiﬁcation performance, and we meta-learn an algorithm –an unsupervised weight update rule – that produces representations useful for this task. Additionally, we constrain our unsupervised update rule to a be a biologically-motivated, neuron-local function, which enables it to generalize to different neural network architectures, datasets, and data modalities. We show that the meta-learned update rule produces useful features and sometimes outperforms existing unsupervised learning techniques. We further show that the meta-learned unsupervised update rule generalizes to train networks with different widths, depths, and nonlinearities. It also generalizes to train on data with randomly permuted input dimensions and even generalizes from image datasets to a text task.
> > 

## Annotations  
%% begin annotations %%



<mark style="background-color: #ff6666">By recasting unsupervised representation learning as meta-learning, we treat the creation of the unsupervised update rule as a transfer learning problem.</mark>

<mark style="background-color: #ff6666">Instead of learning transferable features, we learn a transferable learning rule which does not require access to labels and generalizes across both data domains and neural network architectures. Although we focus on the meta-objective of semi-supervised classification here, in principle a learning rule could be optimized to generate representations for any subsequent task.</mark>

<mark style="background-color: #2ea8e5">UNSUPERVISED REPRESENTATION LEARNING</mark>

<mark style="background-color: #2ea8e5">Autoencoders (Hinton and Salakhutdinov, 2006) work by first compressing and optimizing reconstruction loss.</mark>

<mark style="background-color: #2ea8e5">Extensions have been made to de-noise data (Vincent et al., 2008; 2010), as well as compress information in an information theoretic way (Kingma and Welling, 2013). Le et al. (2011) further explored scaling up these unsupervised methods to large image datasets.</mark>

<mark style="background-color: #2ea8e5">Generative adversarial networks (Goodfellow et al., 2014) take another approach to unsupervised feature learning. Instead of a loss function, an explicit min-max optimization is defined to learn a generative model of a data distribution. Recent work has shown that this training procedure can learn unsupervised features useful for few shot learning (Radford et al., 2015; Donahue et al., 2016; Dumoulin et al., 2016).</mark>

<mark style="background-color: #2ea8e5">Other techniques rely on self-supervision where labels are easily generated to create a non-trivial ‘supervised’ loss. Domain knowledge of the input is often necessary to define these losses. Noroozi and Favaro (2016) use unscrambling jigsaw-like crops of an image. Techniques used by Misra et al. (2016) and Sermanet et al. (2017) rely on using temporal ordering from videos.</mark>

<mark style="background-color: #2ea8e5">Another approach to unsupervised learning relies on feature space design such as clustering. Coates and Ng (2012) showed that k-means can be used for feature learning. Xie et al. (2016) jointly learn features and cluster assignments. Bojanowski and Joulin (2017) develop a scalable technique to cluster by predicting noise. Other techniques such as Schmidhuber (1992), Hochreiter and Schmidhuber (1999), and Olshausen and Field (1997) define various desirable properties about the latent representation of the input, such as predictability, complexity of encoding mapping, independence, or sparsity, and optimize to achieve these properties.</mark>

<mark style="background-color: #e56eee">In order to train the base model, information is propagated backwards by the UnsupervisedUpdate in a manner analogous to backprop. Unlike in backprop however, the backward weights V are decoupled from the forward weights W . Additionally, unlike backprop, there is no explicit error signal as there is no loss. Instead at each layer, and for each neuron, a learning signal is injected by a meta-learned MLP parameterized by θ, with hidden state h. Weight updates are again analogous to those in backprop, and depend on the hidden state of the pre- and postsynaptic neurons for each weight.</mark>

<mark style="background-color: #f19837">Evolved Policy Gradient Houthooft et al. (2018)  performing gradient descent on a learned loss  parameters of a learned loss function  reward Evolutionary Strategies  new environment configurations, both in and not in meta-training distribution.</mark>

<mark style="background-color: #f19837">Learning synaptic learning rules (Bengio et al., 1990; 1992)  run a synapse-local learning rule  parametric learning rule  supervised loss, or similarity to biologicallymotivated network  gradient descent, simulated annealing, genetic algorithms  similar domain tasks</mark>

<mark style="background-color: #f19837">Our work — metalearning for unsupervised representation learning  many applications of an unsupervised update rule  parametric update rule  few shot classification after unsupervised pretraining  SGD new base models (width, depth, nonlinearity), new datasets, new data modalities</mark>

<mark style="background-color: #ff6666">We train these meta-parameters by performing SGD on the sum of the MetaObjective over the course of (inner loop) training in order to find optimal parameters θ∗,  θ∗ = argmin  θ  Etask  [ ∑  t  MetaObjective(φt)  ]  , (1)  that minimize the meta-objective over a distribution of training tasks. Note that φt is a function of θ since θ affects the optimization trajectory</mark>

<mark style="background-color: #ff6666">Our base model consists of a standard fully connected multi-layer perceptron (MLP), with batch normalization (Ioffe and Szegedy, 2015), and ReLU nonlinearities</mark>

<mark style="background-color: #ff6666">The parameters are φ = {W 1, b1, V 1, · · · , W L, bL, V L}, where W l and bl are the weights and biases (applied after batch norm) for layer l, and V l are the corresponding weights used in the backward pass.</mark>

<mark style="background-color: #ff6666">our update rule to be neuron-local, so that updates are a function of pre- and post- synaptic neurons in the base model, and are defined for any base model architecture</mark>

<mark style="background-color: #ff6666">This has the added benefit that it makes the weight updates more similar to synaptic updates in biological neurons, which depend almost exclusively on local pre- and post-synaptic neuronal activity (Whittington and Bogacz, 2017). In practice, we relax this constraint and incorporate some cross neuron information to decorrelate neurons (see Appendix G.5 for more information)</mark>

<mark style="background-color: #ff6666">To build these updates, each neuron i in every layer l in the base model has an MLP, referred to as an update network, associated with it, with output hl  bi = MLP (xl  bi, zbil, V l+1, δl+1; θ) where  b indexes the training minibatch. The inputs to the MLP are the feedforward activations (xl & zl) defined above, and feedback weights and an error signal (V l and δl, respectively) which are defined below.</mark>

<mark style="background-color: #ff6666">All update networks share meta-parameters θ</mark>

<mark style="background-color: #ff6666">Evaluating the statistics of unit activation over a batch of data has proven helpful in supervised learning (Ioffe and Szegedy, 2015). It has similarly proven helpful in hand-designed unsupervised learning rules, such as sparse coding and clustering. We therefore allow hl  bi to accumulate statistics across examples in each training minibatch.</mark>

<mark style="background-color: #ff6666">As in supervised learning, an error signal δl  bi is then propagated backwards through the network. Unlike in supervised backprop, however, this error signal is generated by the corresponding update network for each unit.</mark>

<mark style="background-color: #ff6666">It is read out by linear projection of the per-neuron hidden state h, δl  bi = lin (hl  bi  ), and propogated backward using a set of learned ‘backward weights’ (V l)T , rather than the transpose of the forward weights (W l)T as would be the case in backprop (diagrammed in Figure 1). This is done to be more biologically plausible (Lillicrap et al., 2016).</mark>

<mark style="background-color: #ff6666">Again as in supervised learning, the weight updates (∆W l) are a product of pre- and post-synaptic signals. Unlike in supervised learning however, these signals are generated using the per-neuron update  1https://github.com/tensorflow/models/tree/master/research/learning_ unsupervised_learning networks: ∆W l  ij = func  (  hl  bi, hl−1  bj , Wij  )  .</mark>

<mark style="background-color: #ff6666">The meta-objective we use in this work is based on fitting a linear regression to labeled examples with a small number of data points</mark>

<mark style="background-color: #ff6666">We found that using a cosine distance, CosDist, rather than unnormalized squared error improved stability. Note this meta-objective is only used during meta-training and not used when applying the learned update rule.</mark>

<mark style="background-color: #ff6666">meta-optimize via SGD as opposed to reinforcement learning or other black box methods, due to the superior convergence properties of SGD in high dimensions, and the high dimensional nature of θ</mark>

<mark style="background-color: #ff6666">To improve stability and reduce the computational cost we approximate the gradients ∂[MetaObjective]  ∂θ via truncated backprop through time  (Shaban et al., 2018).</mark>

<mark style="background-color: #ff6666">We construct a set of training tasks consisting of CIFAR10 (Krizhevsky and Hinton, 2009) and multi-class classification from subsets of classes from Imagenet (Russakovsky et al., 2015) as well as from a dataset consisting of rendered fonts (Appendix H.1.1).</mark>

<mark style="background-color: #ff6666">We find that despite only training on networks with 2 to 5 layers and 64 to 512 units per layer, the learned rule generalizes to 11 layers and 10,000 units per layer.</mark>

<mark style="background-color: #5fb236">Training and computing derivatives through recurrent computation of this form is notoriously difficult Pascanu et al. (2013). Training parameters of recurrent systems in general can lead to chaos. We used the usual techniques such as gradient clipping (Pascanu et al., 2012), small learning rates, and adaptive learning rate methods (in our case Adam (Kingma and Ba, 2014))</mark>

<mark style="background-color: #5fb236">When training with truncated backprop the problem shifts from pure optimization to something more like optimizing on a Markov Decision Process where the state space is the base-model weights, φ, and the ‘policy’ is the learned optimizer.</mark>

<mark style="background-color: #5fb236">While traversing these states, the policy is constantly meta-optimized and changing, thus changing the distribution of states the optimizer reaches. This type of non-i.i.d training has been discussed at great length with respect to on and off-policy RL training algorithms (Mnih et al., 2013). Other works cast optimizer meta-learning as RL (Li and Malik, 2017) for this very reason, at a large cost in terms of gradient variance. In this work, we partially address this issue by training a large number of workers in parallel, to always maintain a diverse set of states when computing gradients.</mark>

<mark style="background-color: #5fb236">For similar reasons, the number of steps per truncation, and the total number of unsupervised training steps, are both sampled in an attempt to limit the bias introduced by truncation.</mark>

<mark style="background-color: #5fb236">We found restricting the maximum inner loop step size to be crucial for stability. Pearlmutter (1996) studied the effect of learning rates with respect to the stability of optimization and showed that as the learning rate increases gradients become chaotic. This effect was later demonstrated with respect to neural network training in Maclaurin et al. (2015). If learning rates are not constrained, we found that they rapidly grew and entered this chaotic regime.</mark>

<mark style="background-color: #5fb236">Another technique we found useful in addressing these problems is the use of batch norm in both the base model and in the UnsupervisedUpdate rule.</mark>

<mark style="background-color: #5fb236">Multi-layer perceptron training traditionally requires very precise weight initialization for learning to occur. Poorly scaled initialization can make learning impossible (Schoenholz et al., 2016). When applying a learned optimizer, especially early in meta-training of the learned optimizer, it is very easy for the learned optimizer to cause high variance weights in the base model, after which recovery is difficult. Batch norm helps solve this issues by making more of the weight space usable.</mark>

<mark style="background-color: #5fb236">We implement the described models in distributed Tensorflow (Abadi et al., 2016). We construct a cluster of 512 workers, each of which computes gradients of the meta-objective asynchronously. Each worker trains on one task by first sampling a dataset, architecture, and a number of training steps. Next, each worker samples k unrolling steps, does k applications of the UnsupervisedUpdate(·; θ), computes the MetaObjective on each new state, computes ∂[MetaObjective]  ∂θ and sends this gradient to a  parameter server. The final base-model state, φ, is then used as the starting point for the next unroll until the specified number of steps is reached. These gradients from different workers are batched and θ is updated with asynchronous SGD.</mark>

<mark style="background-color: #5fb236">By batching gradients as workers complete unrolls, we eliminate most gradient staleness while retaining the compute efficiency of asynchronous workers, especially given heterogeneous workloads which arise from dataset and model size variation. An overview of our training can be seen in algorithm F. Due to the small base models and the sequential nature of our compute workloads, we use multi core CPUs as opposed to GPUs. Training occurs over the course of ∼8 days with ∼200 thousand updates to θ with minibatch size 256.</mark>

<mark style="background-color: #ff6666">The base model, the model our learned update rule is training, is an L layer multi layer perception with batch norm.</mark>

<mark style="background-color: #ff6666">We define the MetaObjective to be a few shot linear regression. To increase stability and avoid undesirable loss landscapes, we additionally center, as well as normalize the predicted target before doing the loss computation.</mark>

<mark style="background-color: #ff6666">In our work, this error signal does not exist as there is no loss being optimized. Instead we have a learned top-down signal, dL, at the top of the network</mark>

<mark style="background-color: #5fb236">For a given layer, l, our weight updates are a mixture of multiple low rank readouts from hl and hl−1. These terms are then added together with a learnable weight, in θ to form a final update</mark>

<mark style="background-color: #5fb236">We update both the forward, W l, and the backward, V l, using the same update rule parameters θ</mark>

![[Pasted image 20241017235227.png]]
![[Pasted image 20241017235306.png]]
![[Pasted image 20241017234348.png]]
![[Pasted image 20241017234359.png]]
## Notes


%% Import Date: 2024-10-17T23:43:23.934+01:00 %%
