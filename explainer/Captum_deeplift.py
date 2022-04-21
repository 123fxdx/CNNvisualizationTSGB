###captum
import torch
# import captum.insights.example #
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
baseline=torch.zeros_like(inp)
ig = IntegratedGradients(model)
saliency, delta = ig.attribute(inp, baseline, target=282, return_convergence_delta=True)
#
gs = GradientShap(model)
# We define a distribution of baselines and draw `n_samples` from that
# distribution in order to estimate the expectations of gradients across all baselines
baseline_dist = torch.randn(10, 3) * 0.001
saliency, delta = gs.attribute(inp, stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                   target=282, return_convergence_delta=True)
###
baseline=torch.zeros_like(inp)
# import cv2
# baseline=cv2.GaussianBlur
from skimage import transform, filters
baseline=filters.gaussian_filter(inp, 0.02*max(inp.shape[:2]))#
dl = DeepLift(model)
saliency, delta = dl.attribute(inp, baseline, target=282, return_convergence_delta=True)
#example for CIFAR10, captum.insights.example.py
######

