# encoding: utf-8
"""
autodiff: diffractive lens design
=================================

adaptation of torchgdm example: Optimize a diffractive lens using dipolar core-shell particles

goal: use the design to study impact of
 - size-variations
 - position variations
 - missing particles

author: P. Wiecha, 07/2025
"""
# %%
# imports
# -------
import matplotlib.pyplot as plt
import torch

import torchgdm as tg

tg.use_cuda()

# %%
# simulation setup
# ----------------
# define a meta-atom particle as core-shell Mie-based eff. pola particle

# environment
env = tg.env.freespace_3d.EnvHomogeneous3D(env_material=1.0)

# illumination field(s)
wavelengths = torch.tensor([700.0])
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


# %%
# define fitness function
# -----------------------
# the goal will be to design a lens by positioning the scatterers on a plane.
# the fitness function therefore will calculate the field enhancement at a
# target position (the focus of the lens). The free parameters are the (x,y)
# coordinates of many identical nanostructures (using a list of their positions).


def func(r_pos, r_target, struct_alpha, env, e_inc_list, wavelength, z0=0):
    # create assembly of many same structures at `r_pos` positions
    r_pos = torch.concatenate((r_pos, z0 * torch.ones_like(r_pos)[:, :1]), dim=1)
    struct_combined = struct_alpha.copy(r_pos)

    # struct1 = Mie-struct with random size
    # struct1 = struct1 + [x,y,z]  # move to one of the positions in metasurface

    # all structures combined:
    # struct_full = struct1 + struct2 + ...  # pos. 1
    # struct_list = [struct1, struct2, ...]  # pos. 2 --> pass this to sim class

    # create and run simulation
    sim = tg.simulation.Simulation(
        structures=[struct_combined],
        environment=env,
        illumination_fields=e_inc_list,
        wavelengths=[wavelength],
        device=struct_alpha.device,
        copy_structures=False,  # required: copy not allowed for autograd
    )
    sim.run(verbose=False, progress_bar=False)

    # calculate intensity at target position
    nf_target = sim.get_nf(
        wavelength=wavelength,
        r_probe=r_target,
        illumination_index=0,
        progress_bar=False,
    )
    I_center = nf_target["tot"].get_efield_intensity()  # field intensity

    return sim, I_center


# %%
# setup the optimization
# ----------------------


# initialize geometry: uniform grid
# Note: it's required to specify the device before setting `require_grad``
z0_metasurface = 0
N_particles = 25  # construct grid of N x N particles
D_area = 1.5 * wavelengths[0] * N_particles
z_focus = -D_area  # for the test: use focal distance == lens size

# optimization target
r_target = torch.as_tensor([[0, 0, z_focus]], device=struct_alpha.device)

# init: regular grid
geo_pos_init = tg.tools.geometry.coordinate_map_2d_square(
    D_area / 2, N_particles, z0_metasurface
)
geo_pos_init = geo_pos_init["r_probe"][:, :2]  # optimize x,y coordinates
geo_pos = geo_pos_init.clone().detach().to(struct_alpha.device)
geo_pos.requires_grad = True

# here we simply use a torch optimizer
# Note: the learning rate will be highly problem specific
optimizer = torch.optim.Adam([geo_pos], lr=10)


# %%
# run the optimization loop
# -------------------------

for i in tg.tqdm(range(50)):
    optimizer.zero_grad()

    # evaluate fitness: maximize intensity at focal pos.
    sim, I_center = func(
        geo_pos,
        r_target,
        struct_alpha,
        env,
        e_inc_list,
        wavelength=wavelengths[0],
        z0=z0_metasurface,
    )
    fitness = -1 * I_center

    # backpropagate gradients for total loss function
    loss = fitness
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(i, loss.data, fitness.data)

print("Final intensity enhancement at target:", I_center.data)


# %%
# plot the results
# ----------------

# - calc NF maps
r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(D_area, n=121, r3=z_focus)
nf_res_xy = tg.postproc.fields.nf(sim, wavelengths[0], r_probe=r_probe_xy)

r_probe_xz = tg.tools.geometry.coordinate_map_2d(
    [-D_area, D_area], [-1.5 * z_focus, 1.5 * z_focus], 51, 151, r3=0, projection="xz"
)
nf_res_xz = tg.postproc.fields.nf(sim, wavelengths[0], r_probe=r_probe_xz)


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


# %%
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
print("Average nearest-neighbor distance:", mean_nn_dist.item(), "nm")
print("Min nearest-neighbor distance:", all_nn_distances.min().item(), "nm")
print("Max nearest-neighbor distance:", all_nn_distances.max().item(), "nm")


# %%


# %%
geo_pos.data
# %%
# save the geo optimization
torch.save(geo_pos.detach().cpu(), "optimized/optimized_positions.pt")


# %%
