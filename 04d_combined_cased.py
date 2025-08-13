"""
combined cases
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
# Combined cases
def combined_cases(missing_error, position_error, size_error, seed=0):
    """
    Create a combined case with missing particles, position variations, and size variations.
    :param missing_error: Percentage of particles to be removed (0-1).
    :param position_error: Standard deviation for position noise.
    :param size_error: Standard deviation for radius variation.
    :param seed: Random seed for reproducibility.
    :return: List of particle positions and radii.
    """
    print(
        f"missing_error:{missing_error}, position_error:{position_error}, size_error:{size_error},"
    )
    random.seed(seed)
    np.random.seed(seed)

    ## missing particles ##
    N = len(geo_pos)
    missing_frac = missing_error / 100.0  # convert percentage → fraction
    num_missing = int(N * missing_frac)

    # Randomly select missing particle indices
    all_indices = list(range(N))
    missing_indices = random.sample(all_indices, num_missing)
    rest_indices = list(set(all_indices) - set(missing_indices))
    pos_part_missing = pos_part[rest_indices]  # Keep remaining particles
    N_remaining = len(pos_part_missing)

    ## position variation ##
    geo_pos_varied = (
        pos_part_missing[:, 0:2] + torch.randn((N_remaining, 2)) * position_error
    )
    pos_part_varied = torch.ones((N_remaining, 3)) * z0_metasurface
    pos_part_varied[:, 0:2] = geo_pos_varied

    ## size variation ##
    relative_error = r_core * (size_error / 100.0)  # standard deviation
    variation = torch.randn(N_remaining) * relative_error  # size error ±5%
    r_cores_varied = torch.clamp(r_core + variation, min=5.0)
    r_shells_varied = torch.maximum(r_cores_varied + d_shell, r_cores_varied + 1.0)

    structs = []
    for i in range(N_remaining):
        struct_alpha = tg.struct3d.StructMieSphereEffPola3D(
            wavelengths=wl_nm,
            radii=[r_cores_varied[i].item(), r_shells_varied[i].item()],
            materials=[mat_core, mat_shell],
            environment=env,
        )
        struct_alpha = struct_alpha + pos_part_varied[i]
        structs.append(struct_alpha)

    sim = tg.simulation.Simulation(
        structures=structs,
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=[wl_nm],
        device=torch.device("cuda"),
    )
    sim.run(verbose=False, progress_bar=True)

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
    eff = 100.0 * I_lens.mean() / I_Ideal_lens.mean()
    return eff


missing_error = 5.0
position_error = 20.0
size_error = 5.0
seeds_per_level = 3

vals = [
    combined_cases(missing_error, position_error, size_error, seed=k)
    for k in range(seeds_per_level)
]
eff_mean = float(np.mean(vals))
eff_std = float(np.std(vals))
print(
    f"{missing_error}% missing, {position_error} nm σ pos, {size_error}% σ size → "
    f"{eff_mean:.2f}% ± {eff_std:.2f}%"
)


# scenarios = [
#     (5.0, 20.0, 5.0),  # good scenario
#     (5.0, 10.0, 10.0),  # middle
#     (5.0, 30.0, 20.0),  # bad scenario
# ]
# seeds_per_level = 3

# eff_means, eff_stds = [], []
# for m, p, s in scenarios:
#     vals = [combined_cases(m, p, s, seed=k) for k in range(seeds_per_level)]
#     eff_means.append(float(np.mean(vals)))
#     eff_stds.append(float(np.std(vals)))

# %%
eff_mean_all = [18.09, 16.30, 15.22]
eff_mean_all.append(eff_mean)
labels = ["size 5%", "position 20nm", "missing 5%", "combined"]  # x-axis labels

# Plot the bar chart
plt.bar(
    labels,  # x-axis labels
    eff_mean_all,  # Heights of the bars
    color=["tab:blue", "tab:orange", "tab:green", "tab:gray"],  # Colors for each bar
)

# Add values on top of each bar
for i, value in enumerate(eff_mean_all):
    plt.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=10)

plt.ylabel("Relative Efficiency (%)")
plt.ylim(0, 20)
plt.title("Efficiency Comparison")
plt.show()

# %%
# Plot
plt.figure(figsize=(8.5, 4))
plt.errorbar(scenarios, eff_means, yerr=eff_stds, fmt="o-", capsize=4)
plt.xlabel("Missing particles (%)")
plt.ylabel("Relative efficiency to ideal lens (%)")
plt.title("Efficiency vs. Missing particles (Au@SiO2)")
plt.grid(True)
plt.tight_layout()
plt.show()

for e, m, s in zip(error_levels, eff_means, eff_stds):
    print(f"{e:>5.1f}%  ->  {m:6.2f}%  ± {s:4.2f}%")


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


mean_nn_dist, all_nn_distances = nearest_neighbor_distance(pos_part_varied)
big_distance = (all_nn_distances > mean_nn_dist).sum()
print("Average nearest-neighbor distance:", mean_nn_dist.item(), "nm")
print("Min nearest-neighbor distance:", all_nn_distances.min().item(), "nm")
print("Max nearest-neighbor distance:", all_nn_distances.max().item(), "nm")

print(f"count the distance: {big_distance}")
# %%
