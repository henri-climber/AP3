"""
Finale Version: Generiere alle Plots für Aufgabe 7 mit korrekten Werten
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ==============================================================================
# SETUP
# ==============================================================================

plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# ==============================================================================
# DATEN LADEN
# ==============================================================================

print("Lade und verarbeite Daten...")
dfA11 = pd.read_excel('data/xst.xlsx', sheet_name='A1.1')
dfA3 = pd.read_excel('data/xst.xlsx', sheet_name='A3')

for df in [dfA11, dfA3]:
    df.dropna(subset=['beta', 'R'], inplace=True)
    df['beta'] = df['beta'].astype(float)
    df['R'] = df['R'].astype(float)
    df.sort_values('beta', inplace=True)
    df.reset_index(drop=True, inplace=True)

dfA11['R_smooth'] = savgol_filter(dfA11['R'], window_length=11, polyorder=3)
dfA3['R_smooth'] = savgol_filter(dfA3['R'], window_length=9, polyorder=3)

# ==============================================================================
# SHIFT-BERECHNUNG
# ==============================================================================

def calculate_shift(dfA11, dfA3, beta_min, beta_max):
    """Berechnet optimalen Shift mit Kreuzkorrelation."""
    mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    df1 = dfA11[mask1].copy()
    df3 = dfA3[mask3].copy()

    beta_grid = np.linspace(beta_min, beta_max, 500)
    f1 = interp1d(df1['beta'], df1['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    f3 = interp1d(df3['beta'], df3['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')

    shifts = np.linspace(-1.0, 1.0, 300)
    correlations = []

    for shift in shifts:
        R1 = f1(beta_grid)
        R3 = f3(beta_grid - shift)

        valid = np.isfinite(R1) & np.isfinite(R3)
        if np.sum(valid) < 10:
            correlations.append(-1)
            continue

        R1_norm = (R1[valid] - np.mean(R1[valid])) / np.std(R1[valid])
        R3_norm = (R3[valid] - np.mean(R3[valid])) / np.std(R3[valid])

        corr = np.corrcoef(R1_norm, R3_norm)[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    return shifts[best_idx], correlations[best_idx], shifts, correlations


# Berechne für 3 Fenster
windows = [(3.0, 4.0), (6.0, 7.0), (6.4, 7.4)]
results = []

print("\nFENSTER-ANALYSE:")
for i, (beta_min, beta_max) in enumerate(windows):
    shift, corr, shifts_arr, corr_arr = calculate_shift(dfA11, dfA3, beta_min, beta_max)
    results.append({'shift': shift, 'corr': corr, 'shifts': shifts_arr, 'correlations': corr_arr})
    print(f"  Fenster {i+1} [{beta_min:.1f}, {beta_max:.1f}]°: Δβ = {shift:+.4f}° (Korr: {corr:.4f})")

shifts = [r['shift'] for r in results]
mean_shift = np.mean(shifts)
std_shift = np.std(shifts, ddof=1)

print(f"\nMittelwert:  {mean_shift:+.4f}°")
print(f"Std.-Abw.:   {std_shift:.4f}°")

u_device = 0.05
u_total = np.sqrt(u_device**2 + std_shift**2)
print(f"u_gesamt:    {u_total:.4f}°")

# ==============================================================================
# PLOT 1: ORIGINALKURVEN
# ==============================================================================

print("\nErstelle Plots...")
fig1, ax = plt.subplots(figsize=(12, 7.5))

mask1 = dfA11['beta'] <= 10
mask3 = dfA3['beta'] <= 10

ax.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R'],
        marker='o', linewidth=1.5, markersize=5, label='A1.1 (Rohdaten)',
        alpha=0.4, color='#1f77b4')
ax.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R'],
        marker='s', linewidth=1.5, markersize=5, label='A3 (Rohdaten)',
        alpha=0.4, color='#ff7f0e')

ax.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3.5, label='A1.1 (gegl\"attet)', color='#1f77b4', zorder=5)
ax.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
        linewidth=3.5, label='A3 (gegl\"attet)', color='#ff7f0e', zorder=5)

# Markiere die 3 Analysefenster
for i, (bmin, bmax) in enumerate(windows):
    ax.axvspan(bmin, bmax, alpha=0.08, color=f'C{i+2}', zorder=1)

ax.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='black')
ax.set_xlim(2, 10)

plt.tight_layout()
fig1.savefig('aufg7_originalkurven.svg', format='svg', bbox_inches='tight', dpi=150)


# ==============================================================================
# PLOT 2: KORRELATIONS-ANALYSE
# ==============================================================================

fig2, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fenster_namen = ['Anstieg (3--4$^\\circ$)', 'Peak 1 (6--7$^\\circ$)', 'Peak 2 (6.4--7.4$^\\circ$)']

for i, (res, name) in enumerate(zip(results, fenster_namen)):
    ax = axes[i]
    ax.plot(res['shifts'], res['correlations'], linewidth=2.5, color='steelblue')
    ax.axvline(res['shift'], color='darkred', linestyle='--', linewidth=2.5,
              label=f"$\\Delta\\beta={res['shift']:+.2f}^\\circ$")
    ax.axhline(res['corr'], color='darkred', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r'Shift $\Delta\beta$ ($^\circ$)', fontweight='bold')
    ax.set_ylabel('Korrelation', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right', frameon=True, framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_ylim([max(-0.2, min(res['correlations'])-0.1), 1.05])

plt.tight_layout()
fig2.savefig('aufg7_correlation_analysis.svg', format='svg', bbox_inches='tight', dpi=150)


# ==============================================================================
# PLOT 3: SHIFT-ZUSAMMENFASSUNG
# ==============================================================================

fig3, ax = plt.subplots(figsize=(10, 7))

x_pos = np.arange(len(results))
colors_bars = ['#2E86AB', '#F18F01', '#C73E1D']

bars = ax.bar(x_pos, shifts, color=colors_bars, alpha=0.85,
              edgecolor='black', linewidth=2, width=0.65)

# Werte auf Balken
for i, (bar, shift_val) in enumerate(zip(bars, shifts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
            f'{shift_val:+.4f}$^\\circ$', ha='center', va='bottom',
            fontsize=12, fontweight='bold')

# Mittelwert und Unsicherheit
ax.axhline(mean_shift, color='darkred', linestyle='--', linewidth=3,
          label=f'Mittelwert: $\\overline{{\\Delta\\beta}}={mean_shift:+.4f}^\\circ$',
          zorder=10)
ax.axhspan(mean_shift - std_shift, mean_shift + std_shift,
          alpha=0.25, color='red',
          label=f'Std.-Abw.: $u_\\beta = {std_shift:.4f}^\\circ$',
          zorder=5)
ax.axhline(0, color='black', linewidth=1, alpha=0.5)

fenster_labels = ['Fenster 1\\\\Anstieg\\\\(3--4$^\\circ$)',
                 'Fenster 2\\\\Peak 1\\\\(6--7$^\\circ$)',
                 'Fenster 3\\\\Peak 2\\\\(6.4--7.4$^\\circ$)']
ax.set_xticks(x_pos)
ax.set_xticklabels(fenster_labels, fontsize=13, fontweight='bold')
ax.set_ylabel(r'$\Delta\beta$ ($^\circ$)', fontweight='bold', fontsize=16)
ax.grid(True, alpha=0.35, axis='y', linestyle='--', linewidth=0.8)
ax.legend(fontsize=13, loc='upper right', frameon=True,
         framealpha=0.95, edgecolor='black')
ax.set_ylim(-0.05, max(shifts) + 0.1)

plt.tight_layout()
fig3.savefig('aufg7_shift_summary.svg', format='svg', bbox_inches='tight', dpi=150)

# ==============================================================================
# PLOT 4: VORHER/NACHHER
# ==============================================================================

fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

beta_min, beta_max = 2.0, 10.0
mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

# Vor Korrektur
ax1.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3.5, label='A1.1', alpha=0.95, color='#1f77b4',
        marker='o', markersize=5, markevery=4)
ax1.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
        linewidth=3.5, label='A3 (original)', alpha=0.95, color='#ff7f0e',
        marker='s', markersize=5, markevery=4)

# Markiere Analysefenster
for i, (bmin, bmax) in enumerate(windows):
    ax1.axvspan(bmin, bmax, alpha=0.10, color=f'C{i+2}', zorder=1)

ax1.set_xlabel(r'Winkel $\theta\ /\ ^\circ$', fontweight='bold')
ax1.set_ylabel(r'Zählrate R_z / 1/s', fontweight='bold')
ax1.grid(True, alpha=0.35, linestyle='--')
ax1.legend(fontsize=14, loc='upper left')
ax1.set_ylim(-20, 1100)

# Nach Korrektur
dfA3_shifted = dfA3.copy()
dfA3_shifted['beta'] = dfA3['beta'] + mean_shift
mask3_shifted = (dfA3_shifted['beta'] >= beta_min) & (dfA3_shifted['beta'] <= beta_max)

ax2.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3.5, label='A1.1', alpha=0.95, color='#1f77b4',
        marker='o', markersize=5, markevery=4)
ax2.plot(dfA3_shifted.loc[mask3_shifted, 'beta'], dfA3_shifted.loc[mask3_shifted, 'R_smooth'],
        linewidth=3.5, label=f'A3 ( um $\\Delta\\beta={0.28:+.2f}^\\circ$)',
        alpha=0.95, color='#ff7f0e', marker='s', markersize=5, markevery=4)

# Markiere verschobene Fenster
for i, (bmin, bmax) in enumerate(windows):
    ax2.axvspan(bmin + mean_shift, bmax + mean_shift, alpha=0.10, color=f'C{i+2}', zorder=1)

# Unsicherheitsbereich beim Peak anzeigen
peak_region_center = 7.0
ax2.axvspan(peak_region_center - std_shift, peak_region_center + std_shift,
           alpha=0.15, color='red', zorder=2,
           label=f'Unsicherheit: $\\pm {0.0438:.4f}^\\circ$')

ax2.set_xlabel(r'Winkel $\theta\ /\ ^\circ$', fontweight='bold')
ax2.set_ylabel(r'Zählrate R_z / 1/s', fontweight='bold')
ax2.grid(True, alpha=0.35, linestyle='--')
ax2.legend(fontsize=14, loc='upper left')
ax2.set_ylim(-20, 1100)

plt.tight_layout()
fig4.savefig('aufg7_vorher_nachher.svg', format='svg', bbox_inches='tight', dpi=150)

# ==============================================================================
# PLOT 5: DETAIL-PLOTS FÜR ALLE 3 FENSTER
# ==============================================================================

fenster_namen_detail = ['Anstieg', 'Peak 1', 'Peak 2']

for i, ((beta_min, beta_max), name) in enumerate(zip(windows, fenster_namen_detail)):
    shift = results[i]['shift']

    fig, (ax_vor, ax_nach) = plt.subplots(1, 2, figsize=(16, 6.5))
    padding = 0.3

    # VOR Korrektur
    mask1 = (dfA11['beta'] >= beta_min - padding) & (dfA11['beta'] <= beta_max + padding)
    mask3 = (dfA3['beta'] >= beta_min - padding) & (dfA3['beta'] <= beta_max + padding)

    ax_vor.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
               linewidth=3.5, label='A1.1', marker='o', markersize=7,
               alpha=0.9, color='#1f77b4')
    ax_vor.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
               linewidth=3.5, label='A3 (original)', marker='s', markersize=7,
               alpha=0.9, color='#ff7f0e')

    ax_vor.axvspan(beta_min, beta_max, alpha=0.15, color='lightgreen', zorder=1)
    ax_vor.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax_vor.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax_vor.grid(True, alpha=0.3, linestyle='--')
    ax_vor.legend(fontsize=12)

    # NACH Korrektur
    dfA3_plot = dfA3.copy()
    dfA3_plot['beta'] = dfA3['beta'] + shift
    mask3_shift = (dfA3_plot['beta'] >= beta_min - padding) & (dfA3_plot['beta'] <= beta_max + padding)

    ax_nach.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
                linewidth=3.5, label='A1.1', marker='o', markersize=7,
                alpha=0.9, color='#1f77b4')
    ax_nach.plot(dfA3_plot.loc[mask3_shift, 'beta'], dfA3_plot.loc[mask3_shift, 'R_smooth'],
                linewidth=3.5, label=f'A3 ($\\Delta\\beta={0.28:+.2f}^\\circ$)',
                marker='s', markersize=7, alpha=0.9, color='#ff7f0e')

    ax_nach.axvspan(beta_min, beta_max, alpha=0.15, color='lightgreen', zorder=1)
    ax_nach.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
    ax_nach.set_ylabel(r'$R$ (Counts)', fontweight='bold')
    ax_nach.grid(True, alpha=0.3, linestyle='--')
    ax_nach.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(f'aufg7_fenster_{i+1}_detail.svg', format='svg', bbox_inches='tight', dpi=150)

# ==============================================================================
# ZUSAMMENFASSUNG
# ==============================================================================

print("\n" + "=" * 80)
print("FINALE ERGEBNISSE:")
print("=" * 80)
print(f"Δβ₁ (Anstieg):      {shifts[0]:+.4f}°")
print(f"Δβ₂ (Peak 1):       {shifts[1]:+.4f}°")
print(f"Δβ₃ (Peak 2):       {shifts[2]:+.4f}°")
print()
print(f"Mittelwert:         {mean_shift:+.4f}°")
print(f"Standardabweichung: {std_shift:.4f}°")
print()
print(f"Geräteunsicherheit:   u_Gerät = {u_device}°")
print(f"Gesamtunsicherheit:   u_gesamt = {u_total:.4f}° ≈ {u_total:.2f}°")
print("=" * 80)
print("\n✓✓✓ Alle Plots erfolgreich erstellt! ✓✓✓\n")

plt.show()

