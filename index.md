This is the official website for the paper "Enhancing GANs with MMD Neural Architecture Search, PMish Activation Function, and Adaptive Rank Decomposition." [[Paper]](https://ieeexplore.ieee.org/document/10732016) [[Code]](https://github.com/PrasannaPulakurthi/MMD-NAS-Plus)

by [Prasanna Reddy Pulakurthi](https://prasannapulakurthi.com/), [Mahsa Mozaffari](https://mahsamozaffari.com/), [Sohail Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), [Jamison Heard](https://www.rit.edu/directory/jrheee-jamison-heard), [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600), and [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani).

**Overview of the Paper:** This research enhances Generative Adversarial Networks (GANs) by introducing a novel Parametric Mish (PMish) activation function, MMD-GAN repulsive loss incorporated neural architecture search strategy, and an adaptive rank decomposition method for network compression. The study improves GAN performance and stability with Modified MMD-GAN repulsive loss, demonstrating significant results on datasets like CIFAR-10, CIFAR-100, STL-10, and CelebA. Key contributions include the introduction of the PMish activation function, modified MMD-GAN rep loss, training strategy, and efficient model compression using ARD for deployment on resource-constrained devices.

## Image Generation Results

| CIFAR-10 (32x32) | CelebA (64x64) |
| :---: | :---: |
| <img src="assets/Sec6_Results/Cifar10_Classified.png" title="CIFAR-10"/> <br> <img src="assets/Sec6_Results/Cifar10_Grid.png" title="CIFAR-10"/> | <img src="assets/Sec6_Results/CelebA_Grid.png" title="CelebA"/> |

| CIFAR100 (32x32) | STL-10 (48x48) |
| :---: | :---: |
| <img src="assets/Sec6_Results/Cifar100_Grid.png" title="CIFAR-100" /> | <img src="assets/Sec6_Results/STL10_Grid.png" title="STL-10"/> |

## Parametric Mish (PMish) Activation Function
This is an implementation of the **PMish Activation** function using PyTorch. It combines the `Tanh` and `Softplus` functions with a learnable parameter, `beta`.

```python
import torch
import torch.nn as nn

class PMishActivation(nn.Module): 
    def __init__(self): 
        super(PMishActivation, self).__init__() 
        self.beta = nn.Parameter(torch.ones(1).type(torch.cuda.FloatTensor))  # Learnable parameter beta
        self.tanh_fn = nn.Tanh()
        self.softplus_fn = nn.Softplus()
        
    def forward(self, x): 
        return x * self.tanh_fn((1 / self.beta) * self.softplus_fn(self.beta * x))
```

| PMish | First-Order Derivative | Second-Order Derivative |
| :---: | :---: | :---: |
|<img src="assets/Sec3C_PMish/PMish.png"/> | <img src="assets/AppB/first_derivative_pmish.png"/> | <img src="assets/AppB/second_derivative_pmish.png"/> |

## PMish Output Landscape
<img src="assets/Sec3C_PMish/landscape_pmish.png"/> 

## Neural Architecture Search
![Architecture](assets/Sec3B_Arch/Arch.png)

## Training Strategy

| Small Upper Bound | Large Upper Bound | Increasing Upper Bound (Proposed) |
| :---: | :---: | :---: |
|<img src="assets/Sec3A_Loss/training_animation_R2.gif" title="Small Upper Bound"/> | <img src="assets/Sec3A_Loss/training_animation_R8.gif" title="Large Upper Bound" /> | <img src="assets/Sec3A_Loss/training_animation_R2_8.gif" title="Increasing Upper Bound"/> |
