# test out the detector on a sample image called "mift_pic_fig.png"

# load the image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from gls_mift_kornia_redo import (
    GLSMIFTDetector,
    GLSMIFTDescriptor,
    extract_mift_patches_from_pyramid,
)
from kornia.feature import match_snn, match_smnn
from kornia.geometry.ransac import RANSAC
from kornia_moons.viz import draw_LAF_matches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# img1 = cv2.imread("V2_a.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("V2_b.png", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread("pd_10.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("t1_10.png", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread("pair1-1.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("pair1-2.jpg", cv2.IMREAD_GRAYSCALE)

img2 = cv2.imread("IN718_EBSD.png", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("IN718_BSE.png", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread("test_IN100_ebsd.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("test_IN100_ise.png", cv2.IMREAD_GRAYSCALE)

# img1 = cv2.imread("mift_pic_fig.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("mift_pic_fig.png", cv2.IMREAD_GRAYSCALE)

# # crop them to be square
# img1 = img1[: min(img1.shape[0], img1.shape[1]), : min(img1.shape[0], img1.shape[1])]
# img2 = img2[100:-100, 200:-100]

# do clahe
clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
img1 = clahe.apply(img1)
img2 = clahe.apply(img2)

img1 = torch.tensor(img1).unsqueeze(0).unsqueeze(0).float().to(device)
# img1 = img1 / 255.0
img1 = (img1 - img1.min()) / (img1.max() - img1.min())


img2 = torch.tensor(img2).unsqueeze(0).unsqueeze(0).float().to(device)
# img2 = img2 / 255.0
img2 = (img2 - img2.min()) / (img2.max() - img2.min())

# img2 = ((0.5 - img2) * 2.0) ** 2
# img2 = torch.rot90(img2, 1, [2, 3])

n_angle = 6
n_sigma = 6
n_ang_parts = 6
ang_rate = 4
n_rad_parts = 6
rad_rate = 4
reflect_padding = True
patch_radius = 32
tau = 0.8
scale_factor = 2.0
normalize_lafs_before_extraction = True
root_sift = True

mr_size = 6.0
n_keypoints = 30000
thresh = 0.99
thresh_inlier = 25.0

# create the detector
detector = GLSMIFTDetector(
    n_keypoints=n_keypoints,
    mr_size=mr_size,
    pyr_n_levels=4,
    pyr_init_sigma=1.6,
    pyr_min_size=32,
    pyr_scale_factor=scale_factor,
    fmap_n_angles=n_angle,
    fmap_n_sigma=n_sigma,
    fmap_tau=tau,
    fmap_pad_reflect=reflect_padding,
    fmap_eps=5e-3,
    fast_val_thresh=0.01,
    fast_num_thresh=9,
    fast_use_table=True,
).to(device)
descriptor = GLSMIFTDescriptor(
    n_angle=n_angle,
    n_sigma=n_sigma,
    n_ang_parts=n_ang_parts,
    ang_rate=ang_rate,
    n_rad_parts=n_rad_parts,
    rad_rate=rad_rate,
    root_sift=root_sift,
).to(device)

# detect keypoints
resp1, lafs1 = detector(img1)
resp2, lafs2 = detector(img2)

print(f"Number of keypoints in image 1: {lafs1.shape[1]}")
print(f"Number of keypoints in image 2: {lafs2.shape[1]}")

# lafs1 = lafs1[:, [10]]
# lafs2 = lafs2[:, [10]]

# extract patches
patches1 = extract_mift_patches_from_pyramid(
    img1,
    lafs1,
    scale_factor=scale_factor,
    tau=tau,
    reflect_padding=reflect_padding,
    n_angle=n_angle,
    n_sigma=n_sigma,
    n_ang_parts=n_ang_parts,
    ang_rate=ang_rate,
    n_rad_parts=n_rad_parts,
    rad_rate=rad_rate,
    patch_radius=patch_radius,
    normalize_lafs_before_extraction=normalize_lafs_before_extraction,
)
patches2 = extract_mift_patches_from_pyramid(
    img2,
    lafs2,
    scale_factor=scale_factor,
    tau=tau,
    reflect_padding=reflect_padding,
    n_angle=n_angle,
    n_sigma=n_sigma,
    n_ang_parts=n_ang_parts,
    ang_rate=ang_rate,
    n_rad_parts=n_rad_parts,
    rad_rate=rad_rate,
    patch_radius=patch_radius,
    normalize_lafs_before_extraction=normalize_lafs_before_extraction,
)

print(f"Patches1 shape: {patches1.shape}")
print(f"Patches2 shape: {patches2.shape}")

# compute descriptors and remove the batch dimension
descs1 = descriptor(patches1)[0]
descs2 = descriptor(patches2)[0]

print(f"Descs1 shape: {descs1.shape}")
print(f"Descs2 shape: {descs2.shape}")

# match descriptors
scores, matches = match_smnn(descs1, descs2, th=thresh)

print(f"Number of tentative matches: {matches.shape[0]}")

# get the pts from the LAFS (and remove batch dimension)
src_pts = lafs1[0, matches[:, 0], :, 2]
dst_pts = lafs2[0, matches[:, 1], :, 2]

ransac = RANSAC(
    "homography",
    inl_th=thresh_inlier,
    batch_size=4096 * 128,
    max_iter=100,
    confidence=0.9999,
    max_lo_iters=100,
)

H, mask = ransac(src_pts, dst_pts)
print(f"Number of inliers: {mask.sum()}")

draw_LAF_matches(
    lafs1,
    lafs2,
    matches.cpu().numpy(),
    img1.squeeze().cpu().numpy(),
    img2.squeeze().cpu().numpy(),
    mask.cpu().numpy(),
    # inlier_mask=np.array(
    #     [
    #         True,
    #     ]
    #     * matches.shape[0]
    # ),
    draw_dict={
        "inlier_color": (1, 0.2, 0.2),
        # "tentative_color": (0.2, 0.2, 1),
        # "feature_color": (1, 1, 1),
        "vertical": False,
    },
)

plt.show()
plt.tight_layout()
plt.savefig("mift_pic_fig_matches.png")
plt.close()


# # visualize all of the filter images for each patch
# # start with the first patch
# # plot each part
# patches1_reordered = (
#     patches1[0, 0]
#     .abs()
#     .reshape(n_angle, n_sigma, n_ang_parts, n_rad_parts, ang_rate, rad_rate)
#     .permute(0, 1, 2, 4, 3, 5)
#     .reshape(n_angle, n_sigma, n_ang_parts * ang_rate, n_rad_parts * rad_rate)
#     .cpu()
#     .numpy()
# )

# fig, axes = plt.subplots(n_angle, n_sigma, figsize=(n_sigma, n_angle))
# for i in range(n_angle):
#     for j in range(n_sigma):
#         axes[i, j].imshow(patches1_reordered[i, j], cmap="gray")
#         axes[i, j].axis("off")
# plt.show()
# plt.tight_layout()
# fig.savefig("mift_pic_patches1_reordered.png")
# plt.clf()

# patches2_reordered = (
#     patches2[0, 0]
#     .abs()
#     .reshape(n_angle, n_sigma, n_ang_parts, n_rad_parts, ang_rate, rad_rate)
#     .permute(0, 1, 2, 4, 3, 5)
#     .reshape(n_angle, n_sigma, n_ang_parts * ang_rate, n_rad_parts * rad_rate)
#     .cpu()
#     .numpy()
# )


# fig, axes = plt.subplots(n_angle, n_sigma, figsize=(n_sigma, n_angle))
# for i in range(n_angle):
#     for j in range(n_sigma):
#         small_img = patches2_reordered[i, j]
#         # use numpy to roll the image by half it's height
#         small_img = np.roll(small_img, small_img.shape[0] // 2, axis=0)
#         axes[i, j].imshow(small_img, cmap="gray")
#         axes[i, j].axis("off")
# plt.show()
# plt.tight_layout()
# fig.savefig("mift_pic_patches2_reordered.png")
# plt.clf()
