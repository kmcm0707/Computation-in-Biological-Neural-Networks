#Stanford
# Lecture 1 - What is Multi-task Learning
[[cs330_lecture1.pdf]]
[[cs330_intro_2022.pdf]]
## Datasets
![[Pasted image 20241008152405.png]]
Also impractical to learn a new model for every disease or person etc.
Few-shot learning - small dataset for new thing - only possible if have prior model. Wu: ProtoTransformer Flamingo: a Visual Language Model for Few-Shot Learning. 2022
A Generalist Agent: Reed et al. 2022

## What is a Task?
![[Pasted image 20241008152914.png]]
### Assumptions
Different tasks need to share some structure.
Even if seem different - laws of physics are same, people are organisms with intentions, rules of languages are similar.
## Informal Problem Definitions
The multi-task learning problem: Learn a set of tasks more quickly or more proficiently than learning them independently.
The transfer learning problem: Given data on previous task(s), learn a new task more quickly and/or more proficiently.
## Doesn’t Multi-task Learning Reduce to Single-task Learning?
![[Pasted image 20241008161644.png]]
It can do this - learning one model.
But The transfer learning problem doesn't do this.
