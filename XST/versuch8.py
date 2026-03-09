import pandas as pd
import matplotlib.pyplot as plt
import shutil

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({'font.size': 14})

# if shutil.which("latex"):
#     plt.rcParams["text.usetex"] = True
#     # improve qualit



file = "data/xst.xlsx"
sheet1 = "A1.1"
df = pd.read_excel(file, sheet_name=sheet1)

Winkel = df["beta"]
Rate = df["R"]

plt.plot(Winkel, Rate)
plt.xlabel(r"Winkel $\theta \,/\, ^\circ$")
plt.ylabel(r" Zählerrate $R_Z$ 1/s")
plt.axvline(x=6.3, linestyle="--", label=r"$\mathrm{K}_{\beta}$", color="green")
plt.axvline(x=7.1, linestyle="--", label=r"$\mathrm{K}_{\alpha}$", color="magenta")
plt.axvline(x=12.9, linestyle="--", label=r"$\mathrm{K}_{\beta}$", color="green")
plt.axvline(x=14.5, linestyle="--", label=r"$\mathrm{K}_{\alpha}$", color="magenta")
plt.grid()
plt.tight_layout()
plt.savefig("Emissionsspektrum1.pdf")
plt.legend()
plt.show()


sheet2 = "A1.2"
df = pd.read_excel(file, sheet_name=sheet2)
Winkel = df["beta"]
Rate = df["R"]

plt.plot(Winkel, Rate)
plt.xlabel(r"Winkel $\theta \,/\, ^\circ$")
plt.ylabel(r"Zählerrate $R_Z$ 1/s")
plt.axvline(x=19.6, linestyle="--", label=r"$\mathrm{K}_{\beta}$", color="green")
plt.axvline(x=22.1, linestyle="--", label=r"$\mathrm{K}_{\alpha}$", color="magenta")
plt.legend()
plt.tight_layout()
plt.savefig("Emissionsspektrum2.pdf")
plt.grid()
plt.show()
