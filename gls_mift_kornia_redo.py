from kornia.core import Module, Tensor, stack, eye, concatenate
from kornia.core.check import KORNIA_CHECK_LAF
from kornia.feature.laf import (
    denormalize_laf,
    normalize_laf,
    get_laf_scale,
    laf_is_inside_image,
)
from kornia.filters import gaussian_blur2d
from kornia.geometry.subpix import ConvQuadInterp3d
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List


"""

Fan, Zhongli, Yingdong Pi, Mi Wang, Yifei Kang, and Kai Tan. "GLS-MIFT: A
modality invariant feature transform with global-to-local searching."
Information Fusion 105 (2024): 102252.

GLS-MIFT has several steps:

1. Generate a Scale Space with Gaussian blurring and downsampling (in Kornia)
    a. Octaves are downscaled by a factor of 1.5 (two thirds previous size)
    b. Within each octave, subsesquent images are downsampled by a factor of 2
    c. Gaussian blurring is applied to each image in the scale space
2. Convolve all images with Gaussian derivatives with ~5 sigma levels
3. Compute the rotated response at 6 different angles using step (2) results
4. For each image in the scale space, compute the feature response at each pixel
    a. Compute integrate the response magnitude over the sigma levels
    b. Compute sum response over sigma and divide by ((a)'s Result + epsilon)
    c. Compute median response magnitude on the first scale -> NT (noise
       threshold)
    d. Compute the fractional width: ((a / (max(response mag) + eps)) - 1) /
       (N_sigmas - 1)
    e. Freq spread weight: 1 / (1 + exp(0.5 - 10 * d))
    f. Compute clamped aggregated projected response minus noise threshold for
       each theta
    g. Weight by (e) results and divide by (a) result + epsilon
    h. The final response is the sum of the weighted responses over all thetas
5. Apply FAST to find salient points in the response map
6. Descriptor Computation
    a. Compute magnitudes for all filter responses at all feature points
    b. Over (sigma cross pixels in disk) compute the magnitude argmax histogram
       over theta
    c. Histogram over quadrants of primary direction counts and disambiguate
       phase
    d. Compute a histogram of argmax angles over each subregion in the disk
    e. Reorder every subregion histogram so the primary direction is first
    f. Concatenate all histograms together starting with the primary direction
    g. Normalize the concatenated histograms via L2
    h. RootSIFT mod (optional): L1 normalize the descriptor, square root, and L2
       normalize
7. Global Matching using RANSAC to fit an affine transformation
8. Warp the image using the affine transformation and repeat steps 1-6 on the
   warped image
9. For each reference keypoint, find the 20 nearest neighbors in the warped
   image
10. Redo the matching process with only the 20 nearest neighbors as possible
    matches
11. Final RANSAC to fit a planar homography

"""


class GLSMIFTScalePyramid(Module):
    r"""Create a scale pyramid of an image.

    Images are smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: Number of levels per octave.
        init_sigma: Initial blur level.
        min_size: Minimum size of the smallest octave.
        scale_factor: Scale factor between each level.
        double_image: Add 2x upscaled image as the first level of the pyramid.

    Returns:
        tuple: (output_pyr, sigmas, pixel_dists)
            - output_pyr: List of tensors, each containing an octave of the pyramid.
            - sigmas: Tensor of shape (n_levels,) containing the sigma values.
            - pixel_dists: Tensor of shape (n_octaves,) containing the pixel distances.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output pyramid: :math:`[(B, C, NL, H, W), (B, C, NL, H/scale_factor, W/scale_factor), ...]`
        - Output sigmas: :math:`(n_levels,)`
        - Output pixel_dists: :math:`(n_octaves,)`

    Example:
        >>> pyr = ScalePyramid(n_levels=3, init_sigma=1.6, min_size=15, scale_factor=1.5)
        >>> img = torch.rand(2, 3, 100, 100)
        >>> output_pyr, sigmas, pixel_dists = pyr(img)
    """

    def __init__(
        self,
        n_levels: int = 3,
        init_sigma: float = 1.6,
        min_size: int = 15,
        scale_factor: float = 2.0,
        double_image: bool = False,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.double_image = double_image

        self.sigma_step = scale_factor ** (1.0 / float(self.n_levels))

        sigmas = init_sigma * torch.tensor(
            [(self.sigma_step**o) for o in range(n_levels)]
        )
        self.register_buffer("sigmas", sigmas)

        pixel_dists = torch.tensor([(scale_factor**o) for o in range(30)])
        self.register_buffer("pixel_dists", pixel_dists)

        if double_image:
            self.sigmas *= 2.0
            self.pixel_dists /= 2.0

    def forward(self, x: Tensor) -> tuple[list[Tensor], Tensor, Tensor]:
        pyr = []
        cur_level = self._get_first_level(x)

        while min(cur_level.shape[-2:]) >= self.min_size:
            octave = [cur_level]

            for l in range(1, self.n_levels):
                cur_level = self._get_next_level(cur_level, l)
                octave.append(cur_level)

            pyr.append(stack(octave, dim=2))
            cur_level = self._downsample(cur_level)

        return pyr, self.sigmas, self.pixel_dists[: len(pyr)]

    def _get_first_level(self, x: Tensor) -> Tensor:
        if self.double_image:
            x = self._upsample_2x(x)
            cur_sigma = 1.0
        else:
            cur_sigma = 0.5

        # copying Kornia which copies OpenCV
        if self.init_sigma > cur_sigma:
            sigma = max((self.init_sigma**2 - cur_sigma**2) ** 0.5, 0.01)
            ksize = int(2.0 * 4.0 * sigma + 1.0)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            x = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
        return x

    def _get_next_level(self, x: Tensor, level: int) -> Tensor:
        sigma = self.sigmas[level]
        # don't know why we don't just use half sigma here
        # probably from here: https://en.wikipedia.org/wiki/Scale_space_implementation
        return self._blur(x, sigma * (self.sigma_step**2 - 1.0) ** 0.5)

    def _downsample(self, x: Tensor) -> Tensor:
        if self.scale_factor == 2.0:
            return x[:, :, ::2, ::2]
        else:
            return F.interpolate(
                x,
                size=(
                    int(x.shape[-2] / self.scale_factor),
                    int(x.shape[-1] / self.scale_factor),
                ),
                mode="bilinear",
                align_corners=False,
            )

    def _upsample_2x(self, x: Tensor) -> Tensor:
        return F.interpolate(
            x,
            size=(x.shape[-2] * 2, x.shape[-1] * 2),
            mode="bilinear",
            align_corners=False,
        )

    def _blur(self, x: Tensor, sigma: float) -> Tensor:
        ksize = int(2.0 * 4.0 * sigma + 1.0)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize

        # padding by larger than image not supported by PyTorch
        ksize = min(ksize, x.shape[-2], x.shape[-1])

        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        return gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))


def gaussian_derivative_kernels_2d(
    sigmas: Tensor,
    z_score_cutoff: int = 2,
):
    # establish the kernel size
    kernel_size = 2 * z_score_cutoff * sigmas.max().item() + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    # create the kernel
    x = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-kernel_size // 2, kernel_size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    # add placeholder dimension for broadcasting n_sigma x k_H x k_W
    sigmas = sigmas[:, None, None]

    # compute the gaussian kernel with derivative along x and y
    kernel_x = (
        -xx / (2 * torch.pi * sigmas**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigmas**2))
    )
    kernel_y = (
        -yy / (2 * torch.pi * sigmas**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigmas**2))
    )

    # the kernels are returned as two (n_sigma, k_H, k_W) tensors
    return kernel_x, kernel_y


class GaussianDerivativeLayer(nn.Module):
    def __init__(
        self,
        n_sigma: int = 4,
        tau: float = 0.8,
        n_angle: int = 6,
        reflect_padding: bool = True,
    ):
        super(GaussianDerivativeLayer, self).__init__()

        self.reflect_padding = reflect_padding

        sigmas = torch.arange(1, n_sigma + 1, dtype=torch.float32)
        sigmas *= tau

        # compute the gaussian derivative kernels
        kernel_x, kernel_y = gaussian_derivative_kernels_2d(sigmas)

        # concatenate the kernels along the channel dimension
        kernels = torch.cat([kernel_x, kernel_y], dim=0)[:, None, ...]
        # kernels = kernels.flip(
        #     dims=[
        #         -2,
        #     ]
        # )
        self.register_buffer("kernels", kernels)

        # store angles as a buffer (not a parameter)
        # thetas go (0, 30, 60, 90, 120, 150) degrees by default but need to generalize
        thetas = torch.linspace(0, torch.pi, (n_angle + 1))[:-1][:, None, None, None]
        self.register_buffer("thetas", thetas)

    def forward(self, x):
        # x is expected to be a (B = n_blurs, 1, H, W) tensor
        w_even = x.shape[-1] % 2 == 0
        h_even = x.shape[-2] % 2 == 0

        pad = (
            self.kernels.shape[-1] // 2,
            self.kernels.shape[-1] // 2 + int(w_even),
            self.kernels.shape[-2] // 2,
            self.kernels.shape[-2] // 2 + int(h_even),
        )

        # apply the padding
        if self.reflect_padding:
            x = F.pad(x, pad, mode="reflect")
        else:
            x = F.pad(x, pad, mode="constant", value=0)

        # apply the convolution
        x = F.conv2d(x, self.kernels, padding=0)

        # crop off the padding
        x = x[
            :,
            :,
            : -2 if h_even else -1,
            : -2 if w_even else -1,
        ]

        # split the output into x and y derivatives
        x, y = x.chunk(2, dim=1)

        # use R = cos(theta) * x + sin(theta) * y to compute the rotated responses
        # compute the rotated responses (B, n_sigma, H, W) and (B, n_sigma, H, W) -> (B, n_angle, n_sigma, H, W)
        resp = torch.cos(self.thetas) * x[:, None] + torch.sin(self.thetas) * y[:, None]

        return resp


class GLSMIFTFeatureMap(Module):
    """
    Module that computes the feature map for the GLS-MIFT algorithm.

    """

    def __init__(
        self,
        n_sigma: int = 4,
        tau: float = 0.8,
        n_angles: int = 6,
        pad_reflect: bool = True,
        eps: float = 5e-3,
    ):
        super(GLSMIFTFeatureMap, self).__init__()

        # store the parameters
        self.n_sigma = n_sigma
        self.tau = tau
        self.n_angles = n_angles
        self.eps = eps

        # create the GaussianDerivativeLayer
        self.gdl = GaussianDerivativeLayer(
            n_sigma=n_sigma,
            tau=tau,
            n_angle=n_angles,
            reflect_padding=pad_reflect,
        )

    def forward(self, x: list):
        # x is Gaussian scale space list across octaves: [(B, 1, H, W), (B, 1, H//2, W//2), ...]
        # B is the number of blurring operations that were originally done

        octave_fmaps = []
        octave_filter_mags = []

        # the noise threshold (NT) is the median response magnitude on the first scale
        # over all thetas and all sigmas and all pixels
        nt = torch.median(torch.abs(self.gdl(x[0])))

        # iterate over the images in the scale space
        for image in x:
            # apply the GaussianDerivativeLayer:
            # (B, 1, variable_H, variable_W) -> (B, n_angles, n_sigma, variable_H, variable_W)
            filter_vals = self.gdl(image)

            # compute the response magnitude
            resp_mag = torch.abs(filter_vals)

            # compute the average response magnitude over the sigma dimension
            avg_mag = torch.mean(resp_mag, dim=2, keepdim=True)

            # compute the sum of the responses over sigma and divide by the average magnitude
            avg_resp = torch.sum(filter_vals, dim=2, keepdim=True) / (
                avg_mag + self.eps
            )

            # compute the fractional width which has the max response magnitude in the denominator
            wid = (
                (avg_mag / (torch.max(resp_mag, dim=2, keepdim=True)[0] + self.eps)) - 1
            ) / (self.n_sigma - 1)

            # compute the frequency spread weight
            freq_spread = 1 / (1 + torch.exp(0.5 - 10 * wid))

            # compute the clamped aggregated projected response minus noise threshold for each theta
            clamped_resp = torch.clamp(
                torch.sum(filter_vals * avg_resp, dim=2, keepdim=True) - nt, min=0
            )

            # weight by the frequency spread and divide by the average magnitude
            weighted_resp = clamped_resp * freq_spread / (avg_mag + self.eps)

            # weighted_resp is shape (B, n_angles, 1, H, W) -> (B, n_angles, H, W)
            weighted_resp = weighted_resp.squeeze(2)

            # compute the feature map which is the sum of the responses over all thetas
            octave_fmap = torch.sum(weighted_resp, dim=1, keepdim=True)

            # the final response is the sum of the weighted responses over all thetas
            # however the descriptors will need the non-theta-aggregated responses
            octave_fmaps.append(octave_fmap)

            # store the filter magnitudes
            octave_filter_mags.append(resp_mag)

        # return filter response magnitude list [(B, n_angles, n_sigma, H, W), (B, n_angles, n_sigma, H//2, W//2), ...]
        # and the final feature map list [(B, 1, H, W), (B, 1, H//2, W//2), ...]
        # feature maps are computed by just averaging the responses over the theta dimension
        return octave_filter_mags, octave_fmaps


class FASTFeatureExtractor(Module):
    """
    Feature extractor based on the FAST (Features from Accelerated Segment Test) algorithm.

    Args:
        intensity_threshold (float): FAST pixel intensity threshold
        n_pixels_threshold (int): Min number of cicle pixels beyond threshold
        use_lookup_table (bool): Whether to use a lookup table for counting consecutive pixels.

    Returns:
        tuple: A tuple containing the mean intensity differences and the feature mask.

    Note:

    Defaults exactly reproduce cv2 4.9.0 output (nonmaxSuppression=False)

    """

    def __init__(
        self,
        intensity_threshold: float = (11.0 / 255.0),
        n_pixels_threshold: int = 9,
        use_lookup_table: bool = True,
        nms: bool = True,
        nms_kernel_size: int = 3,
    ):
        super(FASTFeatureExtractor, self).__init__()
        # Define the offset coordinates for the circle pixels
        self.offset_coords = torch.tensor(
            [
                [-3, 0],
                [-3, 1],
                [-2, 2],
                [-1, 3],
                [0, 3],
                [1, 3],
                [2, 2],
                [3, 1],
                [3, 0],
                [3, -1],
                [2, -2],
                [1, -3],
                [0, -3],
                [-1, -3],
                [-2, -2],
                [-3, -1],
            ],
            dtype=torch.long,
        )
        self.intensity_threshold = intensity_threshold
        self.n_pixels_threshold = n_pixels_threshold
        self.use_lookup_table = use_lookup_table
        self.nms = nms
        self.nms_kernel_size = nms_kernel_size

        if self.use_lookup_table:
            # make a tensor of all possible binary sequences in order from 0 to 2^16 - 1
            # https://stackoverflow.com/a/67463295
            numbers = torch.arange(2**16).reshape(2**16, 1)
            exponents = (2 ** torch.arange(16)).flip(dims=[0])
            table = ((exponents & numbers) > 0).bool()
            # count the number of consecutive 1s in each binary sequence
            lookup_table = self._count_ones_run_wraps(table)
            # we make the lookup table a byte tensor because max count is 16
            self.register_buffer("lookup_table", lookup_table.byte())

    def _count_ones_run_wraps(self, a, dtype=torch.int32) -> torch.Tensor:
        """
        Count the number of consecutive 1s in a binary sequence with wraparound. Assumes
        that each 1D slice along the final dimension is a binary sequence with wraparound.

        Args:
            a (torch.Tensor): Input binary or byte tensor of 1s and 0s
            dtype (torch.dtype): Data type for the output tensor (default: torch.int32)

        Returns:
            torch.Tensor: Tensor containing counts lacking final dimension

        """
        # from https://stackoverflow.com/a/59057567
        n = a.shape[-1]
        m = a.shape[:-1]
        return torch.minimum(
            torch.tensor(n, device=a.device, dtype=dtype),
            (
                torch.arange(1, 2 * n + 1, device=a.device, dtype=dtype)
                - torch.cummax(
                    torch.where(
                        a[..., None, :].bool(),
                        0,
                        torch.arange(
                            1, 2 * n + 1, device=a.device, dtype=dtype
                        ).reshape(
                            ((1,) * len(m))
                            + (
                                2,
                                n,
                            )
                        ),
                    ).reshape(*m, 2 * n),
                    dim=-1,
                )[0]
            )
            .max(-1)[0]
            .to(dtype),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1, "FASTFeatureExtractor only supports single-channel images"
        pad = torch.max(torch.abs(self.offset_coords))
        val_h = H - 2 * pad
        val_w = W - 2 * pad

        starts_i = pad + self.offset_coords[:, 0]
        ends_i = pad + self.offset_coords[:, 0] + val_h
        starts_j = pad + self.offset_coords[:, 1]
        ends_j = pad + self.offset_coords[:, 1] + val_w

        circle_pixels = torch.stack(
            [
                x[:, 0, starts_i[i] : ends_i[i], starts_j[i] : ends_j[i]]
                for i in range(16)
            ],
            dim=-1,
        )
        center_pixel = x[:, 0, pad : (pad + val_h), pad : (pad + val_w)].unsqueeze(-1)

        diffs = torch.abs(circle_pixels - center_pixel)
        mean_diffs = diffs.mean(dim=-1)[:, None, :, :]

        # Compute positive and negative intensity masks
        pos_mask = circle_pixels >= (center_pixel + self.intensity_threshold)
        neg_mask = circle_pixels <= (center_pixel - self.intensity_threshold)

        # Count consecutive occurrences for positive and negative masks
        if self.use_lookup_table:
            # convert the 16 binary values to a decimal number via an array of powers of 2
            powers_of_2 = 2 ** torch.arange(16, device=x.device, dtype=torch.int32)
            powers_of_2 = powers_of_2.flip(dims=[0])[None, None, None, :]
            pos_mask_ind = torch.sum(pos_mask.to(torch.int32) * powers_of_2, dim=-1)
            neg_mask_ind = torch.sum(neg_mask.to(torch.int32) * powers_of_2, dim=-1)
            # use the lookup table to get the counts of contiguous 1s with wraparound
            pos_counts = self.lookup_table[pos_mask_ind]
            neg_counts = self.lookup_table[neg_mask_ind]
        else:
            # count the number of consecutive 1s in each binary sequence with wraparound
            pos_counts = self._count_ones_run_wraps(pos_mask)
            neg_counts = self._count_ones_run_wraps(neg_mask)

        # Combine positive and negative counts and (B, H, W) -> (B, 1, H, W)
        features_counts = torch.max(pos_counts, neg_counts)[:, None, :, :]

        # Check n_pixels threshold
        feat_mask = (features_counts >= self.n_pixels_threshold).to(torch.uint8)
        feat_mask = F.pad(feat_mask, (pad,) * 4, mode="constant", value=0)
        mean_diffs = F.pad(mean_diffs, (pad,) * 4, mode="constant", value=0)

        # apply 3x3 non-maximum suppression using the mean_diffs to update the feat_mask
        if self.nms:
            local_optima_mask = (
                F.max_pool2d(
                    mean_diffs,
                    kernel_size=self.nms_kernel_size,
                    stride=1,
                    padding=self.nms_kernel_size // 2,
                )
                == mean_diffs
            )
            feat_mask = feat_mask * local_optima_mask

        return mean_diffs, feat_mask


def radial_grid(
    n_ang: int,
    n_rad: int,
):
    """
    Args:
        n_ang: The number of angular parts
        n_rad: The number of radial parts

    Returns:
        A (n_ang, n_rad, 2) tensor

    """
    # create the radial grid
    thetas = torch.linspace(0, 2 * torch.pi, (n_ang + 1))[:-1]
    radii = torch.linspace(0, 1, (n_rad + 1))[1:]

    # create the grid
    grid = torch.stack(
        [
            radii[None, :] * torch.cos(thetas[:, None]),
            radii[None, :] * torch.sin(thetas[:, None]),
        ],
        dim=-1,
    )
    return grid


def coords_mift(
    n_ang_parts: int = 6,
    ang_rate: int = 4,
    n_rad_parts: int = 3,
    rad_rate: int = 24,
    # expand: bool = True,
):
    """
    Create a radial grid for the MIFT descriptor.

    Args:
        :n_ang_parts: The number of angular parts
        :ang_rate: The number of angles per part
        :n_rad_parts: The number of radial parts
        :rad_rate: The number of radii per part
        # expand: Whether to expand patches by R/(N_r^2) - Removed because it
        makes inconsistent pixels/patch ... we will see how important this is.

    Returns:
        Shape (1, n_ang_parts * n_rad_parts * ang_rate * rad_rate, 2)

    Notes:
        The grid is sized at the unit circle and will need to be scaled by the LAF.

    """
    grid = radial_grid(n_ang_parts * ang_rate, n_rad_parts * rad_rate)
    grid = grid.reshape(n_ang_parts, ang_rate, n_rad_parts, rad_rate, 2)
    grid = torch.swapdims(grid, 1, 2)
    grid = grid.reshape(n_ang_parts * n_rad_parts, ang_rate * rad_rate, 2).contiguous()[
        None, ...
    ]
    return grid


def generate_mift_patch_grid_from_nlaf(
    img: Tensor,
    nlaf: Tensor,
    patch_size: int = 32,
    n_ang_parts: int = 6,
    ang_rate: int = 4,
    n_rad_parts: int = 3,
    rad_rate: int = 24,
) -> Tensor:
    """Helper function for affine grid generation.

    Args:
        img: image tensor of shape :math:`(B, CH, H, W)`.
        LAF: local affine frame with shape :math:`(B, N, 2, 3)`.
        PS: patch size to be extracted.

    Returns:
        grid :math:`(B*N, PS, PS, 2)`

    Notes:

        Really opaque documentation so to clear things up:
        1) LAF 2x2 portion has determinant storing the scale
        2) Need scale of 1.0 to be a patch size of 'patch_size' in original image
        3) LAF 2x1 portion stores the translation
    """
    KORNIA_CHECK_LAF(nlaf)
    B, N, _, _ = nlaf.size()
    _, ch, h, w = img.size()

    # norm, then renorm is needed for allowing detection on one resolution
    # and extraction at arbitrary other
    dlaf = denormalize_laf(nlaf, img)

    # mift grid of shape (1, n_ang_parts * n_rad_parts, ang_rate * rad_rate, 2)
    grid = coords_mift(
        n_ang_parts=n_ang_parts,
        ang_rate=ang_rate,
        n_rad_parts=n_rad_parts,
        rad_rate=rad_rate,
    ).to(dlaf.device)

    # only scale and translation are leveraged in the LAF
    scale = 8 * get_laf_scale(dlaf)

    # rescale grid while its centered at (0, 0) then shift all via broadcasting
    grid = grid * scale.view(B * N, 1, 1, 1)

    grid[..., :, 0] /= float(w - 1)
    grid[..., :, 1] /= float(h - 1)

    grid[..., :, 0:1] += 2 * (dlaf[0, :, 0, 2][:, None, None, None] / float(w - 1)) - 1
    grid[..., :, 1:2] += 2 * (dlaf[0, :, 1, 2][:, None, None, None] / float(h - 1)) - 1

    return grid


def blur_and_downsample_image(
    img: Tensor,
    scale_factor: float = 2.0,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    """
    Blur and downsample an image.

    Args:
        img: image tensor of shape :math:`(B, CH, H, W)`.
        scale_factor: scale factor between each level.

    Returns:
        tensor of shape :math:`(B, CH, H//scale_factor, W//scale_factor)`.

    Notes:
        Pyramid reduce uses a Gaussian kernel with a standard deviation of 2 * downscale / 6.0
        https://scikit-image.org/docs/stable/api/skimage.transform.html

    """
    B, C, H, W = img.size()
    downscale = 1.0 / scale_factor
    sigma = 2.0 * downscale / 6.0
    ksize = int(2.0 * 4.0 * sigma + 1.0)
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    # # hack to get around padding issues
    # ksize = min(ksize, H, W)
    # ksize = ksize + 1 if ksize % 2 == 0 else ksize
    blurred = gaussian_blur2d(img, (ksize, ksize), (sigma, sigma))
    return F.interpolate(
        blurred,
        size=(int(H * downscale), int(W * downscale)),
        mode=mode,
        align_corners=align_corners,
    )


def extract_mift_patches_from_pyramid(
    img: Tensor,
    laf: Tensor,
    scale_factor: float = 2.0,
    n_angle: int = 6,
    n_sigma: int = 4,
    tau: float = 0.8,
    reflect_padding: bool = True,
    n_ang_parts: int = 6,
    ang_rate: int = 4,
    n_rad_parts: int = 3,
    rad_rate: int = 24,
    patch_radius: float = 72.0,
    normalize_lafs_before_extraction: bool = True,
) -> Tensor:
    """
    Extract patches defined by LAFs from image tensor. MIFT patches are polar
    coordinate patches that are segmented into angular and radial parts.

    Args:
        img: image tensor of shape :math:`(B, CH, H, W)`.
        laf: tensor of local affine frames
        scale_factor: resizing factor between octaves of the pyramid
        n_angle: number of angles in the feature map
        n_sigma: number of scales in the feature map
        tau: factor for the sigma values in the feature map
        n_ang_parts: number of angular parts in the patches
        ang_rate: number of angles per part in the patches
        n_rad_parts: number of radial parts in the patches
        rad_rate: number of radii per part in the patches
        normalize_lafs_before_extraction: whether to normalize LAFs before extraction

    Returns:
        patches extracted from the image pyramid

    Shape:
        - Input: :math:`(B, CH, H, W)`
        - LAFs: :math:`(B, N, 2, 3)`
        - Output: :math:`(B, N, n_filters, n_parts, n_pixs_per_part)`
    """

    KORNIA_CHECK_LAF(laf)
    if normalize_lafs_before_extraction:
        # normalize based on the first image of the first octave
        nlaf = normalize_laf(laf, img)
    else:
        nlaf = laf

    B, N, _, _ = laf.size()
    _, CH, H, W = img.size()
    assert CH == 1, "MIFT patches only support single-channel images"

    # scale is the scale of the LAF in the original image
    scale = 2.0 * get_laf_scale(denormalize_laf(nlaf, img)) / patch_radius

    # max_level is the maximum level of the pyramid
    max_level = (
        (
            torch.log(torch.tensor([min(H, W)], device=img.device, dtype=img.dtype))
            / torch.log(torch.tensor([scale_factor], device=img.device))
        )
        .ceil()
        .long()
    )

    # pyr_idx is the index of the pyramid level for each LAF
    pyr_idx = (
        (torch.log(scale) / torch.log(torch.tensor(scale_factor)))
        .clamp(min=0.0, max=max(0, max_level.item() - 1))
        .long()
    )

    # create a module to do the convolution
    gdl = GaussianDerivativeLayer(
        n_sigma=n_sigma,
        tau=tau,
        n_angle=n_angle,
        reflect_padding=reflect_padding,
    ).to(img.device)

    # create the output tensor
    out = torch.empty(
        B,
        N,
        n_angle * n_sigma,
        n_ang_parts * n_rad_parts,
        ang_rate * rad_rate,
        device=nlaf.device,
        dtype=nlaf.dtype,
    )

    # iteratively downsample the image and extract patches when at the correct level
    cur_img = img
    cur_pyr_level = 0
    while min(cur_img.size(2), cur_img.size(3)) >= patch_radius:
        # apply the GaussianDerivativeLayer to the image
        filter_vals = gdl(cur_img).abs()
        b, n_ang, n_sig, h, w = filter_vals.size()
        # reshape for grid sampling
        filter_vals = filter_vals.view(b, n_ang * n_sig, h, w)
        # for loop temporarily, to be refactored
        for i in range(b):
            scale_mask = (pyr_idx[i] == cur_pyr_level).squeeze()
            # print(f"Scale mask sum: {scale_mask.float().sum().item()}")
            if (scale_mask.float().sum().item()) == 0:
                continue
            scale_mask = (scale_mask > 0).view(-1)
            grid = generate_mift_patch_grid_from_nlaf(
                cur_img[i : i + 1],
                nlaf[i : i + 1, scale_mask, :, :],
                patch_size=patch_radius,
                n_ang_parts=n_ang_parts,
                ang_rate=ang_rate,
                n_rad_parts=n_rad_parts,
                rad_rate=rad_rate,
            )
            patches = F.grid_sample(
                filter_vals[i : i + 1].expand(grid.shape[0], n_ang * n_sig, h, w),
                grid,
                padding_mode="border",
                align_corners=False,
            )
            out[i].masked_scatter_(scale_mask.view(-1, 1, 1, 1), patches)
        cur_img = blur_and_downsample_image(
            cur_img,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        cur_pyr_level += 1
    return out


class GLSMIFTDetector(Module):
    """
    Module that computes the GLS-MIFT feature map and extracts patches using the keypoint locations.

    Args:
        n_keypoints: Number of keypoints to detect.
        also_minima: Whether to detect minima as well as maxima.
        pyr_n_levels: Number of levels in the scale pyramid.
        pyr_init_sigma: Initial sigma for the scale pyramid.
        pyr_min_size: Minimum size of the smallest octave.
        pyr_scale_factor: Scale factor between each level.
        fmap_n_angles: Number of angles in the feature map.
        fmap_n_sigma: Number of scales in the feature map.
        fmap_tau: Tau parameter for the feature map.
        fmap_eps: Epsilon parameter for the feature map.
        desc_n_ang_parts: Number of angular parts in the patches.
        desc_ang_rate: Number of angles per part in the patches.
        desc_n_rad_parts: Number of radial parts in the patches.
        desc_rad_rate: Number of radii per part in the patches.

    """

    def __init__(
        self,
        n_keypoints: int = 5000,
        mr_size: float = 6.0,
        pyr_n_levels: int = 6,
        pyr_init_sigma: float = 1.6,
        pyr_min_size: int = 32,
        pyr_scale_factor: float = 2.0,
        fmap_n_angles: int = 6,
        fmap_n_sigma: int = 4,
        fmap_tau: float = 0.8,
        fmap_pad_reflect: bool = True,
        fmap_eps: float = 5e-3,
        fast_val_thresh: float = 11.0 / 255.0,
        fast_num_thresh: int = 9,
        fast_use_table: bool = True,
    ) -> None:
        super().__init__()

        self.n_keypoints = n_keypoints
        self.mr_size = mr_size

        self.pyr_scale_factor = pyr_scale_factor
        self.pyr_n_levels = pyr_n_levels

        self.pyr = GLSMIFTScalePyramid(
            n_levels=pyr_n_levels,
            init_sigma=pyr_init_sigma,
            min_size=pyr_min_size,
            scale_factor=pyr_scale_factor,
        )

        self.fmap = GLSMIFTFeatureMap(
            n_sigma=fmap_n_sigma,
            tau=fmap_tau,
            n_angles=fmap_n_angles,
            pad_reflect=fmap_pad_reflect,
            eps=fmap_eps,
        )

        self.resp = FASTFeatureExtractor(
            intensity_threshold=fast_val_thresh,
            n_pixels_threshold=fast_num_thresh,
            use_lookup_table=fast_use_table,
        )

        self.nms = ConvQuadInterp3d()

    def forward(self, x: Tensor) -> Tensor:
        # x is the input image tensor (must be grayscale) of shape (B, 1, H, W)

        # compute the scale pyramid -> [(B, 1, NL, H, W), (B, 1, NL, H/factor, W/factor), ...]
        pyr, sigmas, pixel_dists = self.pyr(x)

        B, _, NL, _, _ = pyr[0].size()

        # reshape so that the feature map can be computed
        # [(B, NL, H, W), (B, NL, H/factor, W/factor), ...] ->
        # [(B*NL, 1, H, W), (B*NL, 1, H/factor, W/factor), ...]
        pyr = [p[:, 0].view(B * NL, 1, p.size(-2), p.size(-1)) for p in pyr]

        # compute the feature maps
        # octave_filter_mags is [(B * NL, n_angles, n_sigma, H, W) ... ]
        # octave_fmaps is [(B * NL, 1, H, W) ... ]
        octave_filter_mags, octave_fmaps = self.fmap(pyr)

        # run FAST on each octave_fmap
        # mean_diffs is [(B * NL, 1, H, W) ... ]
        # feat_mask is [(B * NL, 1, H, W) ... ]
        mean_diffs, feat_mask = zip(*[self.resp(fmap) for fmap in octave_fmaps])

        # for each mean_diff divide by the median of the mean_diff
        # this is to normalize the responses across scales
        # mean_diffs = [
        #     m.view(B, NL, m.size(-2), m.size(-1)) / torch.mean(m) for m in mean_diffs
        # ]

        # reshape from [(B * NL, 1, H, W) ... ] to [(B, NL, H, W) ... ]
        # multiply by the mask to kill off non-maxima
        processed_maps = [
            m.view(B, NL, m.size(-2), m.size(-1))
            * f.view(B, NL, m.size(-2), m.size(-1))
            for m, f in zip(mean_diffs, feat_mask)
        ]

        # run non-maximum suppression
        # coords is [(B, 2, n_keypoints) ... ]
        # scores is [(B, n_keypoints) ... ]
        coords_max, scores_max = zip(*[self.nms(maps[None]) for maps in processed_maps])

        # scores_max = scores_max.view(scores_max.size(0), -1)
        # coords_max_flat = coords_max.view(scores_max.size(0), 3, -1).permute(0, 2, 1)

        scores_max_flat = [s.view(s.size(0), -1) for s in scores_max]
        coords_max_flat = [
            c.view(c.size(0), 3, -1).permute(0, 2, 1) for c in coords_max
        ]

        # if len(coords_max_flat) > self.n_keypoints:
        #     scores_max, indices = torch.topk(scores_max, self.n_keypoints, dim=1)
        #     coords_max_flat = torch.gather(
        #         coords_max_flat, 1, indices.unsqueeze(-1).repeat(1, 1, 3)
        #     )

        for i in range(len(coords_max_flat)):
            if coords_max_flat[i].size(0) > self.n_keypoints:
                scores_max_flat[i], indices = torch.topk(
                    scores_max_flat[i], self.n_keypoints, dim=1
                )
                coords_max_flat[i] = torch.gather(
                    coords_max_flat[i],
                    1,
                    indices.unsqueeze(-1).repeat(1, 1, 3),
                )

        B, N = scores_max_flat[0].size()

        # # take the coordinate along the NL blurring dimension and convert it to a scale
        # coords_max_flat[:, :, 0] = sigmas[0] * self.pyr_scale_factor ** (
        #     coords_max_flat[:, :, 0] / self.pyr_n_levels
        # )
        coords_max_flat = [
            torch.cat(
                [
                    (
                        sigmas[0]
                        * self.pyr_scale_factor
                        ** (c[:, :, 0:1] / float(self.pyr_n_levels))
                    ),
                    c[:, :, 1:],
                ],
                dim=-1,
            )
            for c in coords_max_flat
        ]
        # # Create local affine frames (LAFs)
        # rotmat = eye(2, dtype=x.dtype, device=x.device).view(1, 1, 2, 2)
        # current_lafs = torch.concatenate(
        #     [
        #         self.mr_size * coords_max_flat[:, :, 0].view(B, N, 1, 1) * rotmat,
        #         coords_max_flat[:, :, 1:3].view(B, N, 2, 1),
        #     ],
        #     3,
        # )
        rotmat = eye(2, dtype=x.dtype, device=x.device)[None, None]
        current_lafs = [
            torch.cat(
                [
                    self.mr_size * c[:, :, 0:1][:, :, :, None] * rotmat,
                    c[:, :, 1:3][:, :, :, None],
                ],
                -1,
            )
            for c in coords_max_flat
        ]

        # # Zero response lafs, which touch the boundary
        # good_mask = laf_is_inside_image(current_lafs, octave[:, 0])
        # resp_flat_best = resp_flat_best * good_mask.to(dev, dtype)

        good_masks = [laf_is_inside_image(l, p) for l, p in zip(current_lafs, pyr)]

        # apply the masks to the scores
        scores_max_flat = [s * m for s, m in zip(scores_max_flat, good_masks)]

        # make the LAFs in terms of original image pixels
        current_lafs = [c * pixel_dists[i] for i, c in enumerate(current_lafs)]

        # concatenate the LAFs and the scores
        lafs = concatenate(current_lafs, 1)
        responses = concatenate(scores_max_flat, 1)

        # Sort the responses
        responses, idxs = torch.topk(responses, k=self.n_keypoints, dim=1)
        lafs = torch.gather(
            lafs, 1, idxs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, 3)
        )
        return responses, lafs


class GLSMIFTDescriptor(Module):
    """
    Module that computes the GLS-MIFT descriptors from extracted patches.

    Args:
        n_ang_parts: Number of angular parts in the patches.
        ang_rate: Number of angles per part in the patches.
        n_rad_parts: Number of radial parts in the patches.
        rad_rate: Number of radii per part in the patches.
        root_sift: Whether to apply RootSIFT normalization.
    """

    def __init__(
        self,
        n_angle: int = 6,
        n_sigma: int = 4,
        n_ang_parts: int = 6,
        ang_rate: int = 4,
        n_rad_parts: int = 3,
        rad_rate: int = 24,
        root_sift: bool = True,
    ):
        super(GLSMIFTDescriptor, self).__init__()

        """
        6. Descriptor Computation
            a. Compute magnitudes for all filter responses at all feature points
            b. Over (sigma cross pixels in disk) compute the magnitude argmax histogram
            over theta
            c. Histogram over quadrants of primary direction counts and disambiguate
            phase
            d. Compute a histogram of argmax angles over each subregion in the disk
            e. Reorder every subregion histogram so the primary direction is first
            f. Concatenate all histograms together starting with the primary direction
            g. Normalize the concatenated histograms via L2
            h. RootSIFT mod (optional): L1 normalize the descriptor, square root, and L2
            normalize
        """

        self.n_angle = n_angle
        self.n_sigma = n_sigma
        self.n_ang_parts = n_ang_parts
        self.ang_rate = ang_rate
        self.n_rad_parts = n_rad_parts
        self.rad_rate = rad_rate
        self.root_sift = root_sift

    def forward(self, patches: Tensor) -> Tensor:
        """
        Compute GLS-MIFT descriptors from patches.

        Args:
            patches: Tensor of shape (B, N, n_filters, n_parts, n_pixs_per_part)
                     where B: Batch size (number of images)
                           N: Number of patches (default up to 5000 per image)
                           n_filters: Number of conv filters (n_angles * n_sigma)
                           n_parts: Number of parts (n_ang_parts * n_rad_parts)
                           n_pixs_per_part: Number of pixels per part (ang_rate * rad_rate)

        Returns:
            descriptors: Tensor of shape (B, N, D)
                         where D: Descriptor dimension
        """
        B, N, _, _, _ = patches.size()

        # we need to make a histogram of the argmax over the angles
        patches_by_ang_filter = patches.reshape(B * N, self.n_angle, -1)

        # compute the argmax over the angles:
        # (B * N, n_angle, n_sigma * n_ang_parts * n_rad_parts * n_pixs_per_part) ->
        # (B * N, n_sigma * n_ang_parts * n_rad_parts * n_pixs_per_part)
        top_angles = torch.argmax(patches_by_ang_filter, dim=1, keepdim=False)

        # histogram with n_angle bins forall parts (B, N, n_ang_parts, n_rad_parts, n_angle)
        parts_hists = torch.zeros(
            (B * N, self.n_ang_parts, self.n_rad_parts, self.n_angle),
            device=patches.device,
            dtype=torch.float32,
        )

        # scatter the counts of the top angles into the parts histogram
        parts_hists.scatter_(
            dim=-1,
            index=top_angles.view(
                B * N,
                self.n_sigma,
                self.n_ang_parts,
                self.n_rad_parts,
                self.ang_rate * self.rad_rate,
            )
            .movedim(1, -1)
            .reshape(
                B * N,
                self.n_ang_parts,
                self.n_rad_parts,
                self.n_sigma * self.ang_rate * self.rad_rate,
            ),  # (B * N, n_ang_parts, n_rad_parts, n_sigma * n_pixs_per_part)
            value=1,
            reduce="add",
        )  # final hists (B * N, n_ang_parts, n_rad_parts, n_angle)

        # primary direction histogram using scatter_add
        prim_dir_hists = torch.sum(parts_hists, dim=(1, 2))

        # compute the primary direction (B * N, n_angle) -> (B * N,)
        # pd is [0, 1, 2, 3, 4, 5] for [0, 30, 60, 90, 120, 150] degrees
        # but that generalizes if n_angle is not 6
        prim_dirs = torch.argmax(prim_dir_hists, dim=1, keepdim=False)

        # instead of by quadrant we just do it by n_ang_parts
        pd_mask = (top_angles == prim_dirs[:, None]).float()

        # view to expose each dimension
        pd_mask = pd_mask.view(B * N, self.n_sigma, self.n_ang_parts, -1)

        # reduced over everything except (B * N, n_ang_parts,)
        disambig_hist = pd_mask.sum(dim=(1, 3))

        # argmax over the disambiguated histogram (B * N, n_ang_parts) -> (B * N,)
        prim_ang_part = torch.argmax(disambig_hist, dim=1, keepdim=False)

        # now we apply shifts along the ang_parts dimension and the hist dimension
        parts_hists = parts_hists.view(
            B * N, self.n_ang_parts, self.n_rad_parts, self.n_angle
        )

        # use meshgrid approach instead of gather so can do both reorderings at once
        ii, jj, kk, ll = torch.meshgrid(
            torch.arange(B * N, device=patches.device),
            torch.arange(self.n_ang_parts, device=patches.device),
            torch.arange(self.n_rad_parts, device=patches.device),
            torch.arange(self.n_angle, device=patches.device),
            indexing="ij",
        )

        # apply the primary direction overall to the angular parts dimension
        jj = (jj + prim_ang_part.view(B * N, 1, 1, 1)) % self.n_ang_parts

        # apply the primary direction to the angle dimension
        ll = (ll + prim_dirs.view(B * N, 1, 1, 1)) % self.n_angle

        # reindex into the parts_hists tensor
        parts_hists = parts_hists[ii, jj, kk, ll]

        # we have to normalize each part histogram to be a probability distribution
        parts_hists = parts_hists / parts_hists.sum(dim=-1, keepdim=True)

        # now we flatten the parts histograms to be a descriptor
        parts_hists = parts_hists.view(B, N, -1)

        # normalize the descriptor
        if self.root_sift:
            # L1 normalize
            parts_hists = F.normalize(parts_hists, p=1, dim=-1)

            # square root
            parts_hists = torch.sqrt(parts_hists)

        # L2 normalize
        parts_hists = F.normalize(parts_hists, p=2, dim=-1)

        return parts_hists.reshape(B, N, -1)


# # test out the detector on a sample image called "mift_pic_fig.png"

# # load the image
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # img = cv2.imread("mift_pic_fig.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("IN718_EBSD.png", cv2.IMREAD_GRAYSCALE)
# # img = cv2.imread("IN718_BSE.png", cv2.IMREAD_GRAYSCALE)

# # do clahe
# clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
# img = clahe.apply(img)

# img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
# img = img / 255.0
# img = img.to(device)

# # parameters
# scale_factor = 1.5
# n_angle = 6
# n_sigma = 4
# tau = 0.8
# reflect_padding = True
# n_ang_parts = 6
# ang_rate = 8
# n_rad_parts = 6
# rad_rate = 8
# normalize_lafs_before_extraction = True
# n_keypoints = 5000
# mr_size = 6.0
# pyr_n_levels = 4
# pyr_scale_factor = 1.5
# fast_val_thresh = 0.001
# root_sift = True

# # create the detector
# detector = GLSMIFTDetector(
#     n_keypoints=n_keypoints,
#     mr_size=mr_size,
#     pyr_n_levels=pyr_n_levels,
#     pyr_scale_factor=pyr_scale_factor,
#     fast_val_thresh=fast_val_thresh,
# ).to(device)

# # run the detector
# responses, lafs = detector(img)

# # print the shapes
# print(f"responses shape: {responses.shape}")
# print(f"lafs shape: {lafs.shape}")

# # use kornia_moons to visualize the keypoints
# from kornia_moons.viz import visualize_LAF

# visualize_LAF(img.cpu(), lafs[:, :5000, :, :].cpu())
# plt.show()
# plt.savefig("viz_laf.png")

# # extract the patches
# patches = extract_mift_patches_from_pyramid(
#     img,
#     lafs,
#     scale_factor=scale_factor,
#     n_angle=n_angle,
#     n_sigma=n_sigma,
#     tau=tau,
#     n_ang_parts=n_ang_parts,
#     ang_rate=ang_rate,
#     n_rad_parts=n_rad_parts,
#     rad_rate=rad_rate,
#     normalize_lafs_before_extraction=normalize_lafs_before_extraction,
# )

# print(f"patches shape: {patches.shape}")

# # patches shape (B, N, n_filters, n_parts, n_pixs_per_part)
# # print(patches[0, 0, 0].view(48, 48).shape)

# # visualize the top patch part by part manually using matplotlib
# patch = patches[0, -1, 0].view(6, 6, 8, 8).cpu().numpy()
# fig, ax = plt.subplots(6, 6, figsize=(6, 6))
# for i in range(6):
#     for j in range(6):
#         ax[i, j].imshow(patch[i, j])
#         ax[i, j].axis("off")
# plt.show()
# plt.tight_layout()
# plt.savefig("viz_patch.png")


# # compute the descriptors
# descriptor = GLSMIFTDescriptor(
#     n_angle=n_angle,
#     n_sigma=n_sigma,
#     n_ang_parts=n_ang_parts,
#     ang_rate=ang_rate,
#     n_rad_parts=n_rad_parts,
#     rad_rate=rad_rate,
#     root_sift=root_sift,
# ).to(device)
# descriptors = descriptor(patches)

# print(f"descriptors shape: {descriptors.shape}")
