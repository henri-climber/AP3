import numpy as np
import uncertainties
from uncertainties import ufloat

k = np.array([77, 78, 78, 76, 74])


mean_k = np.mean(k)

# Standardabweichung (Stichprobe)
std_k = np.std(k, ddof=1)

# Standardfehler des Mittelwerts
error_mean_k = std_k / np.sqrt(len(k))

print("k-Werte:", k)
print("Mittelwert k:", mean_k)
print("Standardabweichung:", std_k)
print("Fehler des Mittelwerts:", error_mean_k)




import numpy as np


l = np.array([197, 195, 191, 196, 196])


mean_l = np.mean(l)


std_l = np.std(l, ddof=1)


error_mean_l = std_k / np.sqrt(len(l))

print("k-Werte:", l)
print("Mittelwert lk:", mean_l)
print("Standardabweichung:", std_l)
print("Fehler des Mittelwerts:", error_mean_l)

d = np.array([282, 282, 285, 283, 280])


mean_d = np.mean(d)

# Standardabweichung (Stichprobe)
std_d = np.std(d, ddof=1)

# Standardfehler des Mittelwerts
error_mean_d = std_d / np.sqrt(len(d))

print("d-Werte:", d)
print("Mittelwert d:", mean_d)
print("Standardabweichung:", std_d)
print("Fehler des Mittelwerts:", error_mean_d)



# Messwerte
e = 670        # mm
sigma_e = 2    # mm

d = 282.4      # mm
sigma_d = 0.81 # mm

# Brennweite
f = 0.25 * (e - (d**2)/e)

# Ableitungen für Fehlerfortpflanzung
df_de = 0.25 * (1 + (d**2)/(e**2))
df_dd = - d / (2*e)

# Unsicherheit von f
sigma_f = np.sqrt((df_de * sigma_e)**2 + (df_dd * sigma_d)**2)

print("Brennweite f' =", f, "mm")
print("Unsicherheit =", sigma_f, "mm")



x = ufloat(1000, 1)
b = ufloat(330, 2)

print(x-b)



# Messwerte (mm)
e = 670.0
sigma_e = 2.0

d = 282.4
sigma_d = 0.81

k = 76.6
sigma_k = 0.75

l = 195.0
sigma_l = 0.75

# Ausdruck unter der Wurzel
A = (e - k - l)**2 - d**2
sqrtA = np.sqrt(A)

# Hauptebenenabstand
h = k + l - sqrtA

# Partielle Ableitungen
dh_de = -(e - k - l) / sqrtA
dh_dk = 1 + (e - k - l) / sqrtA
dh_dl = 1 + (e - k - l) / sqrtA
dh_dd = d / sqrtA

# Fehlerfortpflanzung
sigma_h = np.sqrt(
    (dh_de * sigma_e)**2 +
    (dh_dk * sigma_k)**2 +
    (dh_dl * sigma_l)**2 +
    (dh_dd * sigma_d)**2
)

print("h =", h, "mm")
print("Fehler von h =", sigma_h, "mm")
print(f"Ergebnis: h = ({h:.2f} ± {sigma_h:.2f}) mm")
