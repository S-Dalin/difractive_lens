# %%
import numpy as np
import matplotlib.pyplot as plt
import pymiecs as mie


# %%
# Fixed wavelength
wavelength = 800.0  # nm
k0 = 2 * np.pi / wavelength

# Sweep core radius
r_core_values = np.linspace(60, 140, 100)
r_shell_thickness = 30.0
r_shell_values = r_core_values + r_shell_thickness

# Refractive indices at 700 nm
mat_core = mie.materials.MaterialDatabase("Au")
mat_shell = mie.materials.MaterialDatabase("SiO2")
n_core = mat_core.get_refindex(wavelength=wavelength)
n_shell = mat_shell.get_refindex(wavelength=wavelength)

# Calculate phase shift of a1
a1_phases = []
for r_core, r_shell in zip(r_core_values, r_shell_values):
    a, b = mie.mie_coeff.core_shell_ab(
        k0,
        r_core=r_core,
        n_core=n_core,
        r_shell=r_shell,
        n_shell=n_shell,
        n_max=1,
    )
    a1_phases.append(np.angle(a[0]))

# Plot
plt.plot(r_core_values, a1_phases, label="Phase shift of $a_1$")
plt.xlabel("Core radius (nm)")
plt.ylabel("Phase shift (rad)")
plt.title("Scattering Phase Shift of $a_1$ at 700 nm (Au@SiO₂)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
