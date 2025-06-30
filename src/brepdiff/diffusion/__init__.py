from brepdiff.diffusion.base import Diffusion, DiffusionSample
from brepdiff.diffusion.gaussian_diffusion import GaussianDiffusion1D
from brepdiff.diffusion.separate_gaussian_diffusion import SeparateGaussianDiffusion1D

DIFFUSION_PROCESSES = {
    GaussianDiffusion1D.name: GaussianDiffusion1D,
    SeparateGaussianDiffusion1D.name: SeparateGaussianDiffusion1D,
}
