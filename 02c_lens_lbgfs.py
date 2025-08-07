"""
autodiff: diffractive lens design
=================================

adaptation of torchgdm example: Optimize a diffractive lens using dipolar core-shell particles

goal: use the design to study impact of
 - size-variations
 - position variations
 - missing particles
"""

# %%
# imports
# -------
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchgdm as tg
import numpy as np
import torchdiffract


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
# function
# -----------------------
# the goal will be to design a lens by positioning the scatterers on a plane.
# the fitness function therefore will calculate the field enhancement at a
# target position (the focus of the lens). The free parameters are the (x,y)
# coordinates of many identical nanostructures (using a list of their positions).


def func(r_pos, r_target, struct_alpha, env, e_inc_list, wavelength, z0=0):
    # create assembly of many same structures at `r_pos` positions
    r_pos = torch.concatenate((r_pos, z0 * torch.ones_like(r_pos)[:, :1]), dim=1)
    struct_combined = struct_alpha.copy(r_pos)

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


def separation_penalty(r_pos, min_dist=500.0):
    diffs = r_pos.unsqueeze(1) - r_pos.unsqueeze(0)
    dists = torch.sqrt((diffs**2).sum(dim=-1) + 1e-9)

    # Ignore self-distances
    mask = torch.eye(len(r_pos), device=r_pos.device)
    dists = dists + mask * 1e6

    penalty = torch.relu(min_dist - dists).sum()
    return penalty


# %%
# setup the optimization
# ----------------------

# initialize geometry: uniform grid
# Note: it's required to specify the device before setting `require_grad``
z0_metasurface = 0
N_particles = 25  # construct grid of N x N particles
D_area = 1.0 * wavelengths[0] * N_particles  # 26250
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
# optimizer = torch.optim.Adam([geo_pos], lr=10)
# optimizer_lbfgs = torch.optim.LBFGS([geo_pos], lr=1.0, max_iter=50, history_size=10)


# %%
# L-BFGS function
# ------------------------------------------
# geo_pos - parameter to optimize
# lr = learnign rate
# history_size = Size of history for storing past updates (used to approximate Hessian).
# line_search_fn =
# optimizer_lbfgs = optim.LBFGS(
#     [geo_pos], lr=20, history_size=10, max_iter=20, line_search_fn="strong_wolfe"
# )
optimizer_lbfgs = optim.LBFGS(
    [geo_pos],
    lr=10,
    max_iter=100,
    history_size=20,
    line_search_fn="strong_wolfe",
    tolerance_grad=1e-10,
    tolerance_change=1e-12,
)

loss_history = []
final_sim = None
final_I_center = None


# Closure
def closure():
    global final_sim, final_I_center
    optimizer_lbfgs.zero_grad()
    sim, I_center = func(
        geo_pos,
        r_target,
        struct_alpha,
        env,
        e_inc_list,
        wavelength=wavelengths[0],
        z0=0,
    )
    loss = -I_center + 0.1 * separation_penalty(geo_pos)
    loss.backward()

    # Store last simulation & intensity
    final_sim = sim
    final_I_center = I_center.detach()

    loss_history.append(loss.item())
    return loss


# Run L-BFGS
for i in range(100):  # outer iterations
    optimizer_lbfgs.step(closure)
    if i % 5 == 0:
        print(f"Iter {i}, Loss: {loss_history[-1]}")

# Plot loss
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Loss (-Intensity)")
plt.title("L-BFGS Lens Optimization Test")
plt.grid(True)
plt.show()

print("Final intensity:", final_I_center.item())

#########################################################################################


# %%
# plot the results
# ----------------

# - calc NF maps
r_probe_xy = tg.tools.geometry.coordinate_map_2d_square(D_area, n=121, r3=z_focus)
nf_res_xy = tg.postproc.fields.nf(final_sim, wavelengths[0], r_probe=r_probe_xy)

r_probe_xz = tg.tools.geometry.coordinate_map_2d(
    [-D_area, D_area], [-1.5 * z_focus, 1.5 * z_focus], 31, 101, r3=0, projection="xz"
)
nf_res_xz = tg.postproc.fields.nf(final_sim, wavelengths[0], r_probe=r_probe_xz)


# - plot
plt.figure(figsize=(12, 6))

plt.subplot(121, title=f"XY - $|E|/|E_0|^2$ at z={z_focus}")
im = tg.visu.visu2d.field_intensity(nf_res_xy["tot"])
plt.colorbar(im)
tg.visu2d.structure(final_sim, projection="xy", alpha=0.1, color="w", legend=False)

plt.subplot(122, title=f"XZ - $|E|/|E_0|^2$ at y=0")
im = tg.visu.visu2d.field_intensity(nf_res_xz["tot"])
plt.colorbar(im)
tg.visu2d.structure(final_sim, projection="xz", alpha=0.1, color="w", legend=False)


plt.tight_layout()
plt.show()


# %%
torch.save(
    geo_pos.detach().cpu(),
    "optimized/07_optimized_positions_N25_D_area1_iter10000_lfbgs.pt",
)

# %%
# Load optimized positions
# geo_pos = torch.load("optimized/03_optimized_positions_N25_D_area1_iter1000.pt")

# geo_part = pos_part_loaded
# z0_metasurface = 0
# pos_part = torch.ones((len(pos_part_loaded), 3)) * z0_metasurface
# pos_part[:, 0:2] = pos_part_loaded

# print("Loaded positions:", geo_part.shape)


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


# %%
