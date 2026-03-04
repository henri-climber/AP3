"""
Visuelle Inspektion: Überprüfe ob die verschiedenen Shifts Sinn machen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Daten laden
dfA11 = pd.read_excel('data/xst.xlsx', sheet_name='A1.1')
dfA3 = pd.read_excel('data/xst.xlsx', sheet_name='A3')

# Glätten
dfA11['R_smooth'] = savgol_filter(dfA11['R'], window_length=11, polyorder=3)
dfA3['R_smooth'] = savgol_filter(dfA3['R'], window_length=9, polyorder=3)

# Visualisiere verschiedene Shifts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

test_shifts = [-0.2, 0.0, +0.2, +0.3, +0.5, +0.8]

for idx, shift in enumerate(test_shifts):
    ax = axes[idx]

    # Bereich 2-10°
    mask1 = (dfA11['beta'] >= 2) & (dfA11['beta'] <= 10)

    dfA3_shifted = dfA3.copy()
    dfA3_shifted['beta'] = dfA3['beta'] + shift
    mask3 = (dfA3_shifted['beta'] >= 2) & (dfA3_shifted['beta'] <= 10)

    ax.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
            linewidth=2.5, label='A1.1', alpha=0.9, color='blue')
    ax.plot(dfA3_shifted.loc[mask3, 'beta'], dfA3_shifted.loc[mask3, 'R_smooth'],
            linewidth=2.5, label=f'A3 (shift={shift:+.1f}°)',
            alpha=0.9, color='orange')

    ax.set_xlabel('Beta (°)')
    ax.set_ylabel('R (Counts)')
    ax.set_title(f'Shift = {shift:+.2f}°')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(-50, 1100)

plt.tight_layout()
plt.savefig('visual_shift_comparison.png', dpi=150, bbox_inches='tight')
print("Visueller Vergleich gespeichert: visual_shift_comparison.png")
plt.show()

# Analysiere spezifische markante Punkte
print("\nMARKANTE PUNKTE:")
print("=" * 60)

# 1. Erster steiler Anstieg (wo R deutlich ansteigt)
print("\n1. Anstiegspunkt (R steigt über ~50):")
idx_A11_rise = dfA11[dfA11['R_smooth'] > 50].index[0]
idx_A3_rise = dfA3[dfA3['R_smooth'] > 50].index[0]
beta_A11_rise = dfA11.loc[idx_A11_rise, 'beta']
beta_A3_rise = dfA3.loc[idx_A3_rise, 'beta']
print(f"   A1.1 bei: {beta_A11_rise:.2f}°")
print(f"   A3 bei:   {beta_A3_rise:.2f}°")
print(f"   Differenz: {beta_A11_rise - beta_A3_rise:+.4f}° (A3 liegt früher)")

# 2. Erstes Plateau (R > 400)
print("\n2. Plateau-Erreichen (R > 400):")
idx_A11_plat = dfA11[dfA11['R_smooth'] > 400].index[0]
idx_A3_plat = dfA3[dfA3['R_smooth'] > 400].index[0]
beta_A11_plat = dfA11.loc[idx_A11_plat, 'beta']
beta_A3_plat = dfA3.loc[idx_A3_plat, 'beta']
print(f"   A1.1 bei: {beta_A11_plat:.2f}°")
print(f"   A3 bei:   {beta_A3_plat:.2f}°")
print(f"   Differenz: {beta_A11_plat - beta_A3_plat:+.4f}° (A3 liegt früher)")

# 3. Haupt-Peak
mask_A11_peak = (dfA11['beta'] >= 6) & (dfA11['beta'] <= 8)
mask_A3_peak = (dfA3['beta'] >= 6) & (dfA3['beta'] <= 8)
idx_A11_max = dfA11.loc[mask_A11_peak, 'R_smooth'].idxmax()
idx_A3_max = dfA3.loc[mask_A3_peak, 'R_smooth'].idxmax()
beta_A11_max = dfA11.loc[idx_A11_max, 'beta']
beta_A3_max = dfA3.loc[idx_A3_max, 'beta']
print("\n3. Haupt-Peak (6-8°):")
print(f"   A1.1 Maximum bei: {beta_A11_max:.2f}° (R={dfA11.loc[idx_A11_max, 'R_smooth']:.1f})")
print(f"   A3 Maximum bei:   {beta_A3_max:.2f}° (R={dfA3.loc[idx_A3_max, 'R_smooth']:.1f})")
print(f"   Differenz: {beta_A11_max - beta_A3_max:+.4f}° (A3 liegt früher)")

print("\n" + "=" * 60)
print("FAZIT:")
print("Die Differenzen variieren zwischen verschiedenen Features,")
print("was die Unsicherheit der Kristallorientierung widerspiegelt.")
print("=" * 60)

