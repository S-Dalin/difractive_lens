"""
case1: size variation
"""

# %%
import torch
import torchdiffract

import torchgdm as tg
import numpy as np

import matplotlib.pyplot as plt

tg.use_cuda()


# %%
# load the baseline
geo_pos = torch.load(
    "optimized/14_optimized_positions_N30_D_area17500_iter3000_adam.pt"
)
z0_metasurface = 0
pos_part = torch.ones((len(geo_pos), 3)) * z0_metasurface
pos_part[:, 0:2] = geo_pos

# %%
# some parameters
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
# I lens size variations
"""
gaussian distribution error with standard deviation σ expressed as the core radius
relative_error = 5 nm (this is 1σ)
mean = 100 nm
±1σ → 95-105 nm contains ~68% of particles
±2σ → 90-110 nm contains ~95% of particles
±3σ → 85-115 nm contains ~99.7% of particles

σ (%)	σ (nm)	68% of particles (±1σ)	 95% of particles (±2σ)	  99.7% of particles (±3σ)
1%	    1 nm	    99 - 101 nm	            98 - 102 nm	            97 - 103 nm
5%	    5 nm	    95 - 105 nm	            90 - 110 nm	            5 - 115 nm

Each particel has different size, so that why we need to vary the size of each particle and 
that's why we use for loop of N particles. 
"""


def size_variation(error_percentage, seed=0):
    print(f"error level: {error_percentage}")
    torch.manual_seed(seed)
    N = len(geo_pos)
    error = error_percentage
    relative_error = r_core * (error / 100.0)  # standard deviation
    variation = torch.randn(N) * relative_error  # size error ±5%
    r_cores_varied = r_core + variation
    r_cores_varied = torch.clamp(r_cores_varied, min=5.0)

    r_shells_varied = r_cores_varied + d_shell
    r_shells_varied = torch.maximum(r_shells_varied, r_cores_varied + 1.0)

    structs = []
    for i in range(N):
        struct_alpha = tg.struct3d.StructMieSphereEffPola3D(
            wavelengths=wl_nm,
            radii=[r_cores_varied[i].item(), r_shells_varied[i].item()],
            materials=[mat_core, mat_shell],
            environment=env,
        )
        struct_alpha = struct_alpha + pos_part[i]
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


error_levels = [0, 1, 2, 3, 4, 5, 6, 7]  # % relative core-radius error
seeds_per_level = 3

eff_means = []
eff_stds = []

for e in error_levels:
    vals = [size_variation(e, seed=k) for k in range(seeds_per_level)]
    eff_means.append(np.mean(vals))
    eff_stds.append(np.std(vals))

# %%
# Plot
plt.figure(figsize=(6, 4))

plt.errorbar(error_levels, eff_means, yerr=eff_stds, fmt="o-", capsize=4)
plt.xlabel("Core size error (%)")
plt.ylabel("Relative efficiency to ideal lens (%)")
plt.title("Efficiency vs. Size Variation (Au@SiO2)")
plt.grid(True)
plt.tight_layout()
plt.show()

for e, m, s in zip(error_levels, eff_means, eff_stds):
    print(f"{e:>5.1f}%  ->  {m:6.2f}%  ± {s:4.2f}%")


# %%
# mixed variation
def mixed_size_variation(error_levels, seed=0):
    torch.manual_seed(seed)

    N = len(pos_part)  # use the same positions you add below
    errs = torch.as_tensor(error_levels, dtype=torch.float32)

    # per-particle sigma (nm)
    relative_error = r_core * (errs / 100.0)
    variation = torch.randn(N) * relative_error

    # radii with safety clamps
    r_core_var = torch.clamp(r_core + variation, min=5.0)
    r_shell_var = torch.maximum(r_core_var + d_shell, r_core_var + 1.0)

    # build structures
    structs = []
    for i in range(N):
        s = tg.struct3d.StructMieSphereEffPola3D(
            wavelengths=[wl_nm],
            radii=[float(r_core_var[i]), float(r_shell_var[i])],
            materials=[mat_core, mat_shell],
            environment=env,
        )
        structs.append(s + pos_part[i])

    # simulate
    sim = tg.simulation.Simulation(
        structures=structs,
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=[wl_nm],
        device=torch.device("cuda"),
    )
    with torch.inference_mode():
        sim.run(verbose=False, progress_bar=False)

        r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(
            D_area / 2, n=shape, r3=z_focus
        )
        nf_res_xy = tg.postproc.fields.nf(sim, wl_nm, r_probe=r_probe_xy)
        I = (
            nf_res_xy["tot"]
            .get_efield_intensity()
            .cpu()
            .reshape((shape, shape))[x_m:x_p, x_m:x_p]
        )

    eff = 100.0 * (I.mean().item() / I_Ideal_lens.mean().item())
    return eff


N = len(pos_part)
errors = [2.0] * (N // 2) + [5.0] * (N - N // 2)  # per-particle list

seeds_per_level = 2
vals = [mixed_size_variation(errors, seed=k) for k in range(seeds_per_level)]
eff_mean_mixed = float(np.mean(vals))
eff_std_mixed = float(np.std(vals))
print(eff_mean_mixed, eff_std_mixed)

# print(eff_mean_mixed)


# %%
error_levels_new = []
eff_means_new = []
eff_stds_new = []
error_levels_new = [f"{e}%" for e in error_levels]  # make all % into strings
error_levels_new.append("mixed 2% and 5%")

eff_means_new = eff_means.copy()
eff_means_new.append(eff_mean_mixed)

eff_stds_new = eff_stds.copy()
eff_stds_new.append(eff_std_mixed)

plt.figure(figsize=(8.5, 4))
plt.errorbar(error_levels_new, eff_means_new, yerr=eff_stds_new, fmt="o-", capsize=4)
plt.xlabel("Core size error")
plt.ylabel("Relative efficiency to ideal lens (%)")
plt.title("Efficiency vs. Size Variation (Au@SiO₂)")
plt.grid(True)
plt.tight_layout()
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


# Example usage:
mean_nn_dist, all_nn_distances = nearest_neighbor_distance(geo_pos)
big_distance = (all_nn_distances > mean_nn_dist).sum()
print("Average nearest-neighbor distance:", mean_nn_dist.item(), "nm")
print("Min nearest-neighbor distance:", all_nn_distances.min().item(), "nm")
print(f"count the distance: {big_distance}")
# %%
