import pandas as pd
import matplotlib.pyplot as plt

file_path = "data/rad_cassy.xlsx"

df = pd.read_excel(file_path, sheet_name="Messung Untergrund 1")
df2 = pd.read_excel(file_path, sheet_name="Messung Untergrund 2")

Impulsrate1 = df["$N_A$"]
Kanaele1 = df["$n_A$"]

Impulsrate2 = df2["$N_A$"]
Kanaele2 = df2["$n_A$"]
plt.figure(figsize=(10,6))
plt.plot(Kanaele1, Impulsrate1, label = "Messung ohne Bleiabschirmung")
plt.plot(Kanaele2, Impulsrate2, label = "Messung mit Bleiabschrimung")
plt.xlabel("Kanäle")
plt.ylabel("Impulsrate")
plt.legend()
plt.tight_layout()
plt.savefig("Untergrundmessung.pdf")
plt.grid(True)

plt.show()


import pandas as pd

file_path = "data/rad_cassy.xlsx"

df1 = pd.read_excel(file_path, sheet_name="Messung Untergrund 1")
df2 = pd.read_excel(file_path, sheet_name="Messung Untergrund 2")

counts1 = df1["$N_A$"]
counts2 = df2["$N_A$"]

N1 = counts1.sum()
N2 = counts2.sum()

print("Gesamtzahl Quanten ohne Blei:", N1)
print("Gesamtzahl Quanten mit Blei:", N2)