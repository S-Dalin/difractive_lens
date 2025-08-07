"""
author: P. Wiecha, 04/2025
"""

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
import pymiecs as mie

# - setup a core-shell sphere
wavelengths = np.linspace(400, 900, 300)  # wavelength in nm
k0 = 2 * np.pi / wavelengths

r_core = 90.0
d_shell = 50
r_shell = r_core + d_shell

n_env = 1
mat_core = mie.materials.MaterialDatabase("Au")
mat_shell = mie.materials.MaterialDatabase("sio2")
n_core = mat_core.get_refindex(wavelength=wavelengths)
n_shell = mat_shell.get_refindex(wavelength=wavelengths)


## %% efficiencies

# - calculate efficiencies
q_res = mie.Q(k0, r_core=r_core, n_core=n_core, r_shell=r_shell, n_shell=n_shell)

# # - plot
# plt.plot(wavelengths, q_res["qsca"], label="scat")
# plt.plot(wavelengths, q_res["qabs"], label="abs.")
# plt.plot(wavelengths, q_res["qext"], label="extinct")

# plt.legend()
# plt.xlabel("wavelength (nm)")
# plt.ylabel(r"efficiency (1/$\sigma_{geo}$)")
# plt.tight_layout()
# plt.show()


plt.plot(wavelengths, q_res["qback"], label="backward")
plt.plot(wavelengths, q_res["qfwd"], label=r"forward")

for i in range(2):
    plt.plot(wavelengths, q_res["qsca_m"][i], label=rf"M{i+1}", dashes=[2, 2])
    plt.plot(wavelengths, q_res["qsca_e"][i], label=rf"E{i+1}", dashes=[2, 2])

plt.legend()
plt.xlabel("wavelength (nm)")
plt.ylabel("directional scattering intensity (a.u.)")
plt.title("Si @ Ag, r_core={:.1f}nm, d_shell={:.1f}nm".format(r_core, d_shell))
plt.legend()
plt.show()


# %%
