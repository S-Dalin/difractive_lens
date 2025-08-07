# %%
import pymiecs as mie
import time

import numpy as np
import matplotlib.pyplot as plt

# from PyMieScatt import MieQCoreShell, CoreShellScatteringFunction


wavelengths = np.linspace(400, 900, 100)  # wavelength in nm
k0 = 2 * np.pi / wavelengths

r_core = 100.0
r_shell = r_core + 30.0
n_env = 1
mat_core = mie.materials.MaterialDatabase("Au")
mat_shell = mie.materials.MaterialDatabase("SiO2")
n_core = mat_core.get_refindex(wavelength=wavelengths)
n_shell = mat_shell.get_refindex(wavelength=wavelengths)


# - calculate efficiencies
q_res = mie.Q(k0, r_core=r_core, n_core=n_core, r_shell=r_shell, n_shell=n_shell)

# - plot
plt.plot(wavelengths, q_res["qsca"], label="scat")
# plt.plot(wavelengths, q_res["qabs"], label="abs.")
# plt.plot(wavelengths, q_res["qext"], label="extinct")

plt.legend()
plt.xlabel("wavelength (nm)")
plt.ylabel(r"efficiency (1/$\sigma_{geo}$)")
plt.tight_layout()
plt.show()


for i, spec in enumerate(q_res["qsca_e"]):
    colors_blue = plt.cm.Blues(np.linspace(0.4, 1, len(q_res["qsca_e"])))
    plt.plot(wavelengths, spec.real, label=f"Re(a_{i+1})", color=colors_blue[i])
for i, spec in enumerate(q_res["qsca_m"]):
    colors_red = plt.cm.Reds(np.linspace(0.4, 1, len(q_res["qsca_m"])))
    plt.plot(wavelengths, spec.real, label=f"Re(b_{i+1})", color=colors_red[i])
plt.legend(ncol=2)
plt.xlabel("wavelength (nm)")

# %% Mie coefficients
a, b = mie.mie_coeff.core_shell_ab(
    k0, r_core=r_core, n_core=n_core, r_shell=r_shell, n_shell=n_shell, n_max=3
)

# - plot Mie coefficients
colors_blue = plt.cm.Blues(np.linspace(0.4, 1, len(a)))
colors_red = plt.cm.Reds(np.linspace(0.4, 1, len(a)))

plt.figure(figsize=(10, 4))

plt.subplot(121)
for i, spec in enumerate(a):
    plt.plot(wavelengths, spec.real, label=f"Re(a_{i+1})", color=colors_blue[i])
for i, spec in enumerate(b):
    plt.plot(wavelengths, spec.real, label=f"Re(b_{i+1})", color=colors_red[i])
plt.legend(ncol=2)
plt.xlabel("wavelength (nm)")

plt.subplot(122)
for i, spec in enumerate(a):
    plt.plot(wavelengths, spec.imag, label=f"Im(a_{i+1})", color=colors_blue[i])
for i, spec in enumerate(b):
    plt.plot(wavelengths, spec.imag, label=f"Im(b_{i+1})", color=colors_red[i])
plt.legend(ncol=2)
plt.xlabel("wavelength (nm)")
plt.tight_layout()
plt.show()


# %%
plt.figure()
plt.plot(wavelengths, np.angle(a[0]), label="Phase shift of $a_1$")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Phase (rad)")
plt.title("Scattering Phase Shift: Dipole Term $a_1$")
# plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
