"""
case3: particles missing
"""

# %%
import torch
import torchdiffract

import torchgdm as tg
import numpy as np

import matplotlib.pyplot as plt
import random

tg.use_cuda()


# %%
# load the baseline
geo_pos = torch.load(
    "optimized/14_optimized_positions_N30_D_area17500_iter3000_adam.pt"
)
z0_metasurface = 0
pos_part = torch.ones((len(geo_pos), 3)) * z0_metasurface
pos_part[:, 0:2] = geo_pos  # torch.Size([900, 3])


# %%
# some parameters

z0_metasurface = 0.0
wavelengths = torch.tensor([700.0])  # wavelength
wl_nm = float(wavelengths[0].item())

calc_zone = wl_nm
shape = 121 * 4

# N_particles = 30  # construct grid of N x N particles
D_area = 1.0 * wl_nm * 25
z_focus = -D_area

x_m = int(shape / 2 - calc_zone * shape / (2 * D_area))
x_p = int(shape / 2 + calc_zone * shape / (2 * D_area)) + 1

# evironment and illumination field(s)
env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)  # environment
e_inc_list = [tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=torch.pi)]
#  structure: Mie-theory based particle
r_core = 100.0
d_shell = 30.0
r_shell = r_core + d_shell
mat_core = tg.materials.MatDatabase("Au")
mat_shell = tg.materials.MatDatabase("sio2")

# %%
# Calculation Ideal lens phase distribution (@ z=z position of the lens)
calc_zone_IL_pos = tg.tools.geometry.coordinate_map_2d_square(
    D_area / 2, n=shape, r3=z_focus
)["r_probe"].reshape((shape, shape, 3))
Ideal_Phase = torch.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        Ideal_Phase[i][j] = (
            2
            * torch.pi
            * (
                torch.sqrt(
                    calc_zone_IL_pos[i][j][0] ** 2
                    + calc_zone_IL_pos[i][j][1] ** 2
                    + z_focus**2
                )
                - (-z_focus)
            )
            / (wl_nm)
        )

# Propagation to the focal distance
e_in = torch.as_tensor(np.exp(-1j * tg.to_np(Ideal_Phase)))
difflay = torchdiffract.layers.PropagationLayer(
    Nx=shape,
    Ny=shape,
    Dx=D_area * 1e-9,
    Dy=D_area * 1e-9,
    wl=wl_nm * 1e-9,
    propag_z=(-z_focus) * 1e-9,
)  # 10e-10 because torchdiffract uses meters


E_propa = difflay(e_in)
I_Ideal_lens = np.abs(tg.to_np(E_propa)) ** 2

# When we propagate we need to give the information of the phase of our full lens
# but when we calculate the efficiency we only look at the wl x wl zone around the focal point
x_m = int(shape / 2 - calc_zone * shape / (2 * D_area))
x_p = int(shape / 2 + calc_zone * shape / (2 * D_area)) + 1
I_Ideal_lens = I_Ideal_lens[x_m:x_p, x_m:x_p]


# %%
# missing particles
def missing_particles(error_percentage, seed=0):
    print(f"error level: {error_percentage}")
    torch.manual_seed(seed)
    random.seed(seed)

    N = len(geo_pos)
    missing_frac = error_percentage / 100.0  # convert percentage → fraction
    num_missing = int(N * missing_frac)

    # Randomly select missing particle indices
    all_indices = list(range(N))
    missing_indices = random.sample(all_indices, num_missing)
    rest_indices = list(set(all_indices) - set(missing_indices))

    # Keep remaining particles
    pos_part_missing = pos_part[rest_indices]

    struct_alpha = tg.struct3d.StructMieSphereEffPola3D(
        wavelengths=[wl_nm],
        radii=[r_core, r_shell],
        materials=[mat_core, mat_shell],
        environment=env,
    )
    struct_combined = struct_alpha.copy(pos_part_missing)

    # Run simulation
    sim = tg.simulation.Simulation(
        structures=[struct_combined],
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=[wl_nm],
        device=torch.device("cuda"),
    )
    sim.run(verbose=False, progress_bar=False)

    # Near-field at focal plane
    r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(
        D_area / 2, n=shape, r3=z_focus
    )
    nf_res_xy = tg.postproc.fields.nf(sim, wl_nm, r_probe=r_probe_xy)
    I_lens = (
        nf_res_xy["tot"]
        .get_efield_intensity()
        .cpu()
        .reshape((shape, shape))[x_m:x_p, x_m:x_p]
    )

    # Relative efficiency
    eff = 100.0 * I_lens.mean() / I_Ideal_lens.mean()
    return eff


error_levels = [0, 1, 2, 5, 10, 15, 20, 30, 40, 50]  # in %
seeds_per_level = 3

eff_means = []
eff_stds = []

for e in error_levels:
    vals = [missing_particles(e, seed=k) for k in range(seeds_per_level)]
    eff_means.append(np.mean(vals))
    eff_stds.append(np.std(vals))

# %%
# Plot
plt.figure(figsize=(6, 4))
plt.errorbar(error_levels, eff_means, yerr=eff_stds, fmt="o-", capsize=4)
plt.xlabel("Missing particles (%)")
plt.ylabel("Relative efficiency to ideal lens (%)")
plt.title("Efficiency vs. Missing particles (Au@SiO2)")
plt.grid(True)
plt.tight_layout()
plt.show()

for e, m, s in zip(error_levels, eff_means, eff_stds):
    print(f"{e:>5.1f}%  ->  {m:6.2f}%  ± {s:4.2f}%")


# %%
# %%
# near field plot
r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(D_area / 2, n=shape, r3=z_focus)

nf_res_xy = tg.postproc.fields.nf(sim, wavelengths[0], r_probe=r_probe_xy)

r_probe_xz = tg.tools.geometry.coordinate_map_2d(
    [-D_area, D_area], [-1.5 * z_focus, 1.5 * z_focus], 31, 101, r3=0, projection="xz"
)
nf_res_xz = tg.postproc.fields.nf(sim, wavelengths[0], r_probe=r_probe_xz)

I_lens = (
    nf_res_xy["tot"]
    .get_efield_intensity()
    .cpu()
    .reshape((shape, shape))[x_m:x_p, x_m:x_p]
)

# - plot
plt.figure(figsize=(12, 6))

plt.subplot(121, title=f"XY - $|E|/|E_0|^2$ at z={z_focus}")
im = tg.visu.visu2d.field_intensity(nf_res_xy["tot"])
plt.colorbar(im)
tg.visu2d.structure(sim, projection="xy", alpha=0.1, color="w", legend=False)

plt.subplot(122, title=f"XZ - $|E|/|E_0|^2$ at y=0")
im = tg.visu.visu2d.field_intensity(nf_res_xz["tot"])
plt.colorbar(im)
tg.visu2d.structure(sim, projection="xz", alpha=0.1, color="w", legend=False)

plt.tight_layout()
plt.show()

im = plt.imshow(
    I_lens, extent=[-calc_zone / 2, calc_zone / 2, -calc_zone / 2, calc_zone / 2]
)
plt.title("Lens - Intensity around focal point")
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.colorbar(im)
plt.show()

# %%
# Calculation of the phase distribution for a ideal lens (@ z=z position of the lens)
calc_zone_IL_pos = tg.tools.geometry.coordinate_map_2d_square(
    D_area / 2, n=shape, r3=z_focus
)["r_probe"].reshape((shape, shape, 3))
Ideal_Phase = torch.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        Ideal_Phase[i][j] = (
            2
            * torch.pi
            * (
                torch.sqrt(
                    calc_zone_IL_pos[i][j][0] ** 2
                    + calc_zone_IL_pos[i][j][1] ** 2
                    + z_focus**2
                )
                - (-z_focus)
            )
            / (wavelengths[0])
        )

# Propagation to the focal distance
e_in = torch.as_tensor(np.exp(-1j * tg.to_np(Ideal_Phase)))
difflay = torchdiffract.layers.PropagationLayer(
    Nx=shape,
    Ny=shape,
    Dx=D_area * 1e-9,
    Dy=D_area * 1e-9,
    wl=wavelengths[0] * 1e-9,
    propag_z=(-z_focus) * 1e-9,
)  # 10e-10 because torchdiffract uses meters


E_propa = difflay(e_in)
I_Ideal_lens = np.abs(tg.to_np(E_propa)) ** 2

# When we propagate we need to give the information of the phase of our full lens
# but when we calculate the efficiency we only look at the wl x wl zone around the focal point
x_m = int(shape / 2 - calc_zone * shape / (2 * D_area))
x_p = int(shape / 2 + calc_zone * shape / (2 * D_area)) + 1
I_Ideal_lens = I_Ideal_lens[x_m:x_p, x_m:x_p]

# Plot of the Ideal lens Intensity
# im = plt.imshow(
#     I_Ideal_lens, extent=[-calc_zone / 2, calc_zone / 2, -calc_zone / 2, calc_zone / 2]
# )
# plt.title("Ideal Lens - Intensity around focal point")
# plt.xlabel("x [nm]")
# plt.ylabel("y [nm]")
# plt.colorbar(im)
# plt.show()


# %%
# compare to ideal lens
Eff = 100 * I_lens.mean() / I_Ideal_lens.mean()
print(f"efficiency compare to the ideal lens: {Eff}")
extent = [-calc_zone / 2, calc_zone / 2, -calc_zone / 2, calc_zone / 2]

plt.figure(figsize=(12, 5))

# Ideal lens
plt.subplot(1, 2, 1)
plt.imshow(I_Ideal_lens, extent=extent)
plt.title("Ideal Lens Intensity")
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.colorbar()

# Baseline optimized lens
plt.subplot(1, 2, 2)
plt.imshow(I_lens, extent=extent)
plt.title("Baseline Optimized Lens Intensity")
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.colorbar()


plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 5))
plt.bar(["Ideal Lens", "Baseline Optimized"], [100, Eff], color=["gray", "blue"])
plt.ylabel("Relative Efficiency (%)")
plt.ylim(0, 110)
plt.title("Efficiency Comparison")
plt.show()


# %%
# check the separation between particles and particles
def nearest_neighbor_distance(r_pos):
    diffs = r_pos.unsqueeze(1) - r_pos.unsqueeze(0)
    dists = torch.sqrt((diffs**2).sum(dim=-1) + 1e-9)

    # Ignore self-distances
    mask = torch.eye(len(r_pos), device=r_pos.device)
    dists = dists + mask * 1e6

    # Take the closest neighbor for each particle
    nearest_distances, _ = torch.min(dists, dim=1)
    return nearest_distances.mean(), nearest_distances


mean_nn_dist, all_nn_distances = nearest_neighbor_distance(pos_part_missing)
big_distance = (all_nn_distances > mean_nn_dist).sum()
print("Average nearest-neighbor distance:", mean_nn_dist.item(), "nm")
print("Min nearest-neighbor distance:", all_nn_distances.min().item(), "nm")
print("Max nearest-neighbor distance:", all_nn_distances.max().item(), "nm")
print(f"count the distance: {big_distance}")

# %%
