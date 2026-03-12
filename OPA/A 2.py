
#Fehlerrechnung zu A2

import numpy as np

# Positionen in cm
schirm_B = np.array([35.0, 34.8, 34.9, 34.9, 35.0])
schirm_G = np.array([37.9, 37.6, 37.6, 37.5, 37.6])
linse = 45.0

# Brennweiten berechnen (in cm)
f_B_cm = linse - schirm_B
f_G_cm = linse - schirm_G

# in mm umrechnen
f_B = f_B_cm * 10
f_G = f_G_cm * 10

# Mittelwert
mean_B = np.mean(f_B)
mean_G = np.mean(f_G)

# Standardabweichung
std_B = np.std(f_B, ddof=1)
std_G = np.std(f_G, ddof=1)

# Fehler des Mittelwerts
error_mean_B = std_B / np.sqrt(len(f_B))
error_mean_G = std_G / np.sqrt(len(f_G))

print("Linse B")
print("Messwerte (mm):", f_B)
print("Mittelwert (mm):", mean_B)
print("Fehler Mittelwert (mm):", error_mean_B)

print("\nLinse G")
print("Messwerte (mm):", f_G)
print("Mittelwert (mm):", mean_G)
print("Fehler Mittelwert (mm):", error_mean_G)
