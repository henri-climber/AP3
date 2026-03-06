import pandas as pd

# Konstanten
lam = 523.8e-9   # Laser Wellenlänge (m)
l = 0.05         # Küvettenlänge (m) -> ggf anpassen

data = {
    "delta_p_video1":[0.6,0.58,0.5,0.44,0.39,0.3,0.28,0.22,0.16,0.1,0.02,0],
    "N_video1":[0,3,6,9,12,15,18,21,24,27,30,33],

    "delta_p_video2":[0.6,0.56,0.5,0.44,0.38,0.29,0.26,0.2,0.16,0.08,0.01,0],
    "N_video2":[0,3,6,9,12,15,18,21,24,27,30,33]
}

df = pd.DataFrame(data)

# Δn berechnen
df["delta_n_video1"] = df["N_video1"] * lam / (2*l)
df["delta_n_video2"] = df["N_video2"] * lam / (2*l)

print(df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# -----------------------------
# Daten
# -----------------------------
df = pd.DataFrame({
    "delta_p_video1": [0.60,0.58,0.50,0.44,0.39,0.30,0.28,0.22,0.16,0.10,0.02,0.00],
    "N_video1":       [0,3,6,9,12,15,18,21,24,27,30,33],
    "delta_p_video2": [0.60,0.56,0.50,0.44,0.38,0.29,0.26,0.20,0.16,0.08,0.01,0.00],
    "N_video2":       [0,3,6,9,12,15,18,21,24,27,30,33]
})

# -----------------------------
# Konstanten
# -----------------------------
lam = 632.8e-9
l = 0.05
sigma_l = 0.5e-3       # 0.01 mm = 1e-5 m
sigma_dp = 0.02

# -----------------------------
# Δn berechnen
# -----------------------------
df["delta_n_video1"] = df["N_video1"] * lam / (2 * l)
df["delta_n_video2"] = df["N_video2"] * lam / (2 * l)

# -----------------------------
# Fehler Δn
# -----------------------------
df["sigma_n_video1"] = (df["N_video1"] * lam / (2 * l**2)) * sigma_l
df["sigma_n_video2"] = (df["N_video2"] * lam / (2 * l**2)) * sigma_l

# -----------------------------
# Fit Video 1
# -----------------------------
x1 = df["delta_p_video1"].values
y1 = df["delta_n_video1"].values

m1, b1 = np.polyfit(x1, y1, 1)
xfit1 = np.linspace(min(x1), max(x1), 100)
yfit1 = m1 * xfit1 + b1

# -----------------------------
# Fit Video 2
# -----------------------------
x2 = df["delta_p_video2"].values
y2 = df["delta_n_video2"].values

m2, b2 = np.polyfit(x2, y2, 1)
xfit2 = np.linspace(min(x2), max(x2), 100)
yfit2 = m2 * xfit2 + b2

# -----------------------------
# Plot Video 1
# -----------------------------
def format_power_of_ten(x, pos):
    if np.isclose(x, 0.0):
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10 ** exponent)
    if np.isclose(abs(coefficient), 1.0):
        sign = "-" if coefficient < 0 else ""
        return rf"${sign}10^{{{exponent}}}$"
    return rf"${coefficient:.1f}\cdot 10^{{{exponent}}}$"


plt.figure()

plt.errorbar(
    x1,
    y1,
    xerr=sigma_dp,
    yerr=df["sigma_n_video1"],
    fmt='o',
    mfc='none',      # ungefüllte Punkte
    capsize=3,
    label="Messwerte"
)

plt.plot(xfit1, yfit1, label=f"Fit: y = {m1:.2e}x + {b1:.2e}")

plt.xlabel(r"Druckunterschied $\Delta p$ / $\operatorname{bar}$ ")
plt.ylabel(r"Änderung des Brechungsindex $\Delta n$")
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("A3Messung1.pdf")
plt.show()

# -----------------------------
# Plot Video 2
# -----------------------------
plt.figure()

plt.errorbar(
    x2,
    y2,
    xerr=sigma_dp,
    yerr=df["sigma_n_video2"],
    fmt='o',
    mfc='none',      # ungefüllte Punkte
    capsize=3,
    label="Messwerte"
)

plt.plot(xfit2, yfit2, label=f"Fit: y = {m2:.2e}x + {b2:.2e}")

plt.xlabel(r"Druckunterschied $\Delta p$ / $\operatorname{bar}$ ")
plt.ylabel(r"Änderung des Brechungsindex $\Delta n$")
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_power_of_ten))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("A3Messung2.pdf")
plt.show()

# -----------------------------
# Fitparameter
# -----------------------------
print("Video 1:")
print("Steigung =", m1)
print("Intercept =", b1)

print("\nVideo 2:")
print("Steigung =", m2)
print("Intercept =", b2)
