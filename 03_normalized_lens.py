# %%
import torch
import torchdiffract

import torchgdm as tg
import numpy as np

import matplotlib.pyplot as plt

torch.cuda.empty_cache()


# %%
# Load optimized positions
pos_part_loaded = torch.load(
    "optimized/11_optimized_positions_N30_D_area17500_iter5000_adam.pt"
)

geo_part = pos_part_loaded
z0_metasurface = 0
pos_part = torch.ones((len(pos_part_loaded), 3)) * z0_metasurface
pos_part[:, 0:2] = pos_part_loaded

print("Loaded positions:", geo_part.shape)

# %%
wavelengths = torch.tensor([700.0])

calc_zone = wavelengths[
    0
]  # We are going to calculate the mean of the efficiency around the focal point (wl x wl zone)
shape = 121 * 4

N_particles = 30  # construct grid of N x N particles
D_area = 1.0 * wavelengths[0] * 25
z_focus = -D_area

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
im = plt.imshow(
    I_Ideal_lens, extent=[-calc_zone / 2, calc_zone / 2, -calc_zone / 2, calc_zone / 2]
)
plt.title("Ideal Lens - Intensity around focal point")
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.colorbar(im)
plt.show()

##################################################################################

# %%
# environment
env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)

# illumination field(s)
e_inc_list = [tg.env.freespace_3d.PlaneWave(e0p=1.0, e0s=0.0, inc_angle=torch.pi)]

#  structure: Mie-theory based particle
r_core = 100.0  # nm
d_shell = 30.0  # nm
mat_core = tg.materials.MatDatabase("Au")
mat_shell = tg.materials.MatDatabase("sio2")
struct_alpha = tg.struct3d.StructMieSphereEffPola3D(
    wavelengths=wavelengths,
    radii=[r_core, r_core + d_shell],
    materials=[mat_core, mat_shell],
    environment=env,
)
# create and run simulation
sim = tg.simulation.Simulation(
    structures=[struct_alpha.copy(pos_part)],
    environment=env,
    illumination_fields=e_inc_list,
    wavelengths=[wavelengths[0]],
    device=torch.device("cuda"),
)
sim.run(verbose=False, progress_bar=False)

# To compare with the ideal lens we also calculate the field above the full lens but only keep the wl x wl zone
r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(D_area / 2, n=shape, r3=z_focus)
nf_res_xy = tg.postproc.fields.nf(sim, wavelengths[0], r_probe=r_probe_xy)
I_lens = (
    nf_res_xy["tot"]
    .get_efield_intensity()
    .cpu()
    .reshape((shape, shape))[x_m:x_p, x_m:x_p]
)

im = plt.imshow(
    I_lens, extent=[-calc_zone / 2, calc_zone / 2, -calc_zone / 2, calc_zone / 2]
)
plt.title("Lens - Intensity around focal point")
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.colorbar(im)
plt.show()

Eff = 100 * I_lens.mean() / I_Ideal_lens.mean()
# Eff = 100 * I_lens.max() / I_Ideal_lens.max()

print(f"Efficiency = {Eff} %")
# %%
# plot efficiency opt to the ideal lens
import matplotlib.pyplot as plt

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


# %%
# compare the efficiency

plt.figure(figsize=(4, 5))
plt.bar(["Ideal Lens", "Baseline Optimized"], [100, Eff], color=["gray", "blue"])
plt.ylabel("Relative Efficiency (%)")
plt.ylim(0, 110)
plt.title("Efficiency Comparison")
plt.show()

# %%
I_Ideal_lens.shape
# %%
I_lens.shape

# %%
