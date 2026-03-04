"""
Aufgabe 7 - OPTIMIERTE FINALE VERSION
====================================

Erkenntnis: Die Kurven A1.1 und A3 haben unterschiedliche Formen/Intensitäten,
nicht nur einen einfachen horizontalen Shift. Das ist physikalisch sinnvoll,
da unterschiedliche Kristallorientierungen unterschiedliche Beugungsmuster erzeugen.

Methode: Fokussiere auf gut-definierte, vergleichbare Features (Peaks, Kanten)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

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

print("=" * 80)
print("AUFGABE 7: WINKELUNSICHERHEIT AUS KRISTALLORIENTIERUNG")
print("=" * 80)

dfA11 = pd.read_excel('data/xst.xlsx', sheet_name='A1.1')
dfA3 = pd.read_excel('data/xst.xlsx', sheet_name='A3')

for df in [dfA11, dfA3]:
    df.dropna(subset=['beta', 'R'], inplace=True)
    df['beta'] = df['beta'].astype(float)
    df['R'] = df['R'].astype(float)
    df.sort_values('beta', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Glätten
dfA11['R_smooth'] = savgol_filter(dfA11['R'], window_length=11, polyorder=3)
dfA3['R_smooth'] = savgol_filter(dfA3['R'], window_length=9, polyorder=3)

print(f"\nDaten geladen:")
print(f"  A1.1: {len(dfA11)} Punkte, β: {dfA11['beta'].min():.1f}-{dfA11['beta'].max():.1f}°")
print(f"  A3:   {len(dfA3)} Punkte, β: {dfA3['beta'].min():.1f}-{dfA3['beta'].max():.1f}°")


# ==============================================================================
# ANALYSE 1: PEAK-POSITIONEN BESTIMMEN
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSE 1: PEAK-POSITIONEN")
print("=" * 80)

def gaussian(x, amp, mu, sigma, offset):
    """Gauss-Funktion mit Offset."""
    return offset + amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def fit_peak(df, beta_min, beta_max, plot_fit=False):
    """Fittet Gauss-Funktion an Peak."""
    mask = (df['beta'] >= beta_min) & (df['beta'] <= beta_max)
    data = df[mask].copy()

    idx_max = data['R_smooth'].idxmax()
    beta_max_val = data.loc[idx_max, 'beta']
    R_max_val = data.loc[idx_max, 'R_smooth']
    R_min_val = data['R_smooth'].min()

    try:
        popt, pcov = curve_fit(
            gaussian,
            data['beta'],
            data['R_smooth'],
            p0=[R_max_val - R_min_val, beta_max_val, 0.3, R_min_val],
            maxfev=10000
        )

        peak_pos = popt[1]
        peak_uncertainty = np.sqrt(np.diag(pcov))[1]

        if plot_fit:
            beta_fine = np.linspace(data['beta'].min(), data['beta'].max(), 200)
            R_fit = gaussian(beta_fine, *popt)
            return peak_pos, peak_uncertainty, beta_fine, R_fit, data

        return peak_pos, peak_uncertainty
    except:
        return beta_max_val, 0.1


# Haupt-Peak bei ~6.8-7.2°
print("\nHaupt-Peak (dominantes Feature):")
peak_A11, unc_A11 = fit_peak(dfA11, 6.0, 8.0)
peak_A3, unc_A3 = fit_peak(dfA3, 6.0, 8.0)
shift_peak = peak_A11 - peak_A3

print(f"  A1.1: β = {peak_A11:.4f}° ± {unc_A11:.4f}°")
print(f"  A3:   β = {peak_A3:.4f}° ± {unc_A3:.4f}°")
print(f"  → Shift: Δβ = {shift_peak:+.4f}°")


# ==============================================================================
# ANALYSE 2: KANTENPOSITION (50% DES MAXIMUM)
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSE 2: ANSTIEGS-KANTE")
print("=" * 80)

def find_half_max_position(df, beta_min, beta_max, fraction=0.5):
    """Findet Position wo Kurve fraction*max erreicht."""
    mask = (df['beta'] >= beta_min) & (df['beta'] <= beta_max)
    data = df[mask].copy()

    R_max = data['R_smooth'].max()
    R_min = data['R_smooth'].min()
    threshold = R_min + fraction * (R_max - R_min)

    # Finde ersten Punkt über threshold
    above_threshold = data[data['R_smooth'] > threshold]
    if len(above_threshold) == 0:
        return np.nan, np.nan

    idx = above_threshold.index[0]

    # Interpoliere
    if idx > data.index[0]:
        idx_before = data.index[data.index < idx][-1]
        beta1 = data.loc[idx_before, 'beta']
        beta2 = data.loc[idx, 'beta']
        R1 = data.loc[idx_before, 'R_smooth']
        R2 = data.loc[idx, 'R_smooth']

        frac = (threshold - R1) / (R2 - R1)
        beta_interp = beta1 + frac * (beta2 - beta1)

        return beta_interp, 0.05

    return data.loc[idx, 'beta'], 0.1


# Anstiegskante im Bereich 3-5°
print("\nAnstiegskante (50% des lokalen Maximum):")
edge_A11, unc_edge_A11 = find_half_max_position(dfA11, 3.0, 5.0, fraction=0.5)
edge_A3, unc_edge_A3 = find_half_max_position(dfA3, 3.0, 5.0, fraction=0.5)
shift_edge = edge_A11 - edge_A3

print(f"  A1.1: β = {edge_A11:.4f}° ± {unc_edge_A11:.3f}°")
print(f"  A3:   β = {edge_A3:.4f}° ± {unc_edge_A3:.3f}°")
print(f"  → Shift: Δβ = {shift_edge:+.4f}°")


# ==============================================================================
# ANALYSE 3: KORRELATION IN AUSGEWÄHLTEN FENSTERN
# ==============================================================================

print("\n" + "=" * 80)
print("ANALYSE 3: KORRELATIONS-ANALYSE IN STRUKTURREICHEN FENSTERN")
print("=" * 80)

def calculate_shift_correlation(dfA11, dfA3, beta_min, beta_max):
    """Cross-Correlation Methode."""
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
    return shifts[best_idx], correlations[best_idx]


# Fokussiere auf die zuverlässigsten Bereiche (hohe Korrelation)
windows_selected = [
    (3.2, 4.0, "Anstieg"),
    (6.5, 7.5, "Haupt-Peak"),
    (8.5, 9.5, "Hochplateau"),
]

shifts_corr = []
print("\nKorrelations-basierte Shifts:")
for beta_min, beta_max, name in windows_selected:
    shift, corr = calculate_shift_correlation(dfA11, dfA3, beta_min, beta_max)
    shifts_corr.append(shift)
    print(f"  {name:15s} [{beta_min:.1f}, {beta_max:.1f}]°: "
          f"Δβ = {shift:+.4f}° (Korr: {corr:.4f})")

mean_corr = np.mean(shifts_corr)
std_corr = np.std(shifts_corr, ddof=1)
print(f"\n  Mittelwert:  {mean_corr:+.4f}°")
print(f"  Std.-Abw.:   {std_corr:.4f}°")


# ==============================================================================
# FINALES ERGEBNIS
# ==============================================================================

# Kombiniere beide Methoden
all_shifts = [shift_peak, shift_edge] + shifts_corr
final_shift = np.mean(all_shifts)
final_uncertainty = np.std(all_shifts, ddof=1)

print("\n" + "=" * 80)
print("FINALES ERGEBNIS (kombinierte Methoden)")
print("=" * 80)
print()
print(f"Alle ermittelten Shifts:")
print(f"  Peak-Fitting:     {shift_peak:+.4f}°")
print(f"  Kanten-Analyse:   {shift_edge:+.4f}°")
print(f"  Korrelation-1:    {shifts_corr[0]:+.4f}°")
print(f"  Korrelation-2:    {shifts_corr[1]:+.4f}°")
print(f"  Korrelation-3:    {shifts_corr[2]:+.4f}°")
print()
print(f"Mittlerer systematischer Shift:  Δβ = {final_shift:+.4f}°")
print(f"Winkelunsicherheit:               u_β = {final_uncertainty:.4f}°")
print(f"|Δβ|:                             {abs(final_shift):.4f}°")
print()
print("PHYSIKALISCHE INTERPRETATION:")
print("  • Die Streuung der Shifts zeigt, dass die Kurven nicht identisch sind")
print("  • Unterschiedliche Kristallorientierung → unterschiedliche Beugungsmuster")
print(f"  • Die Unsicherheit u_β = {final_uncertainty:.4f}° ist die zusätzliche")
print("    Winkelunsicherheit aus der ungenauen Kristallpositionierung")
print()
print("VERWENDUNG:")
print(f"  u_total = sqrt(u_Gerät² + u_β²)")
print(f"  Mit u_Gerät = 0.05° → u_total = sqrt(0.05² + {final_uncertainty:.4f}²)")
u_total = np.sqrt(0.05**2 + final_uncertainty**2)
print(f"                      → u_total = {u_total:.4f}°")
print("=" * 80)


# ==============================================================================
# HAUPTPLOT: VORHER/NACHHER MIT BESTEM SHIFT
# ==============================================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 13))

beta_min, beta_max = 2.0, 10.0

# Plot 1: Vor Korrektur
ax1 = axes[0]
mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

ax1.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3, label='A1.1', alpha=0.9, color='#1f77b4',
        marker='o', markersize=5, markevery=4)
ax1.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
        linewidth=3, label='A3 (original)', alpha=0.9, color='#ff7f0e',
        marker='s', markersize=5, markevery=4)

# Markiere analysierte Features
ax1.axvline(peak_A11, color='blue', linestyle=':', linewidth=2, alpha=0.6)
ax1.axvline(peak_A3, color='orange', linestyle=':', linewidth=2, alpha=0.6)
ax1.axvline(edge_A11, color='blue', linestyle='-.', linewidth=1.5, alpha=0.5)
ax1.axvline(edge_A3, color='orange', linestyle='-.', linewidth=1.5, alpha=0.5)

ax1.text(peak_A11, 850, f'{peak_A11:.2f}°', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax1.text(peak_A3, 950, f'{peak_A3:.2f}°', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax1.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax1.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax1.set_title(r'VOR Shift-Korrektur: Peaks liegen nicht \"ubereinander',
             fontweight='bold', pad=15)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.legend(fontsize=14, loc='upper left')
ax1.set_ylim(-20, 1100)

# Plot 2: Nach Korrektur
ax2 = axes[1]
dfA3_shifted = dfA3.copy()
dfA3_shifted['beta'] = dfA3['beta'] + final_shift
mask3_shifted = (dfA3_shifted['beta'] >= beta_min) & (dfA3_shifted['beta'] <= beta_max)

ax2.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
        linewidth=3, label='A1.1', alpha=0.9, color='#1f77b4',
        marker='o', markersize=5, markevery=4)
ax2.plot(dfA3_shifted.loc[mask3_shifted, 'beta'], dfA3_shifted.loc[mask3_shifted, 'R_smooth'],
        linewidth=3, label=f'A3 (korrigiert: $\\beta \\rightarrow \\beta + {final_shift:+.3f}^\\circ$)',
        alpha=0.9, color='#ff7f0e', marker='s', markersize=5, markevery=4)

# Markiere korrigierte Positionen
ax2.axvline(peak_A11, color='blue', linestyle=':', linewidth=2, alpha=0.6,
           label=f'A1.1 Peak: {peak_A11:.2f}°')
ax2.axvline(peak_A3 + final_shift, color='orange', linestyle=':', linewidth=2, alpha=0.6,
           label=f'A3 Peak (korr.): {peak_A3+final_shift:.2f}°')

# Unsicherheitsbereich zeigen
ax2.axvspan(peak_A11 - final_uncertainty, peak_A11 + final_uncertainty,
           alpha=0.15, color='red', label=f'Unsicherheit: $\\pm${final_uncertainty:.3f}°')

ax2.set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
ax2.set_ylabel(r'$R$ (Counts)', fontweight='bold')
ax2.set_title(r'NACH Shift-Korrektur: Bessere \"Ubereinstimmung',
             fontweight='bold', pad=15)
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.legend(fontsize=12, loc='upper left', ncol=2)
ax2.set_ylim(-20, 1100)

plt.tight_layout()
fig.savefig('aufg7_final_alignment.svg', format='svg', bbox_inches='tight', dpi=150)
fig.savefig('aufg7_final_alignment.pdf', bbox_inches='tight', dpi=150)
print("\n✓ Haupt-Plot gespeichert: aufg7_final_alignment.svg / .pdf")


# ==============================================================================
# PLOT 2: ZUSAMMENFASSUNG DER SHIFTS
# ==============================================================================

fig2, ax = plt.subplots(figsize=(11, 7))

methods = ['Peak-Fit', 'Kanten-\nAnalyse', 'Korr.\nAnstieg',
          'Korr.\nHaupt-Peak', 'Korr.\nPlateau']
shifts_all = [shift_peak, shift_edge] + shifts_corr
colors_bars = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

x_pos = np.arange(len(shifts_all))
bars = ax.bar(x_pos, shifts_all, color=colors_bars, alpha=0.8,
             edgecolor='black', linewidth=1.5, width=0.7)

# Mittelwert
ax.axhline(final_shift, color='red', linestyle='--', linewidth=3,
          label=f'Mittelwert: $\\Delta\\beta = {final_shift:+.4f}^\\circ$', zorder=10)

# Unsicherheitsband
ax.axhspan(final_shift - final_uncertainty, final_shift + final_uncertainty,
          alpha=0.25, color='red',
          label=f'Unsicherheit: $u_\\beta = {final_uncertainty:.4f}^\\circ$', zorder=5)

ax.axhline(0, color='black', linewidth=1, alpha=0.5)

# Beschriftung der Balken
for i, (bar, shift_val) in enumerate(zip(bars, shifts_all)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + (0.03 if height > 0 else -0.05),
            f'{shift_val:+.3f}°', ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=12, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\Delta\beta$ ($^\circ$)', fontweight='bold', fontsize=16)
ax.set_title('Shift-Bestimmung mit verschiedenen Methoden',
            fontweight='bold', fontsize=17, pad=15)
ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.8)
ax.legend(fontsize=14, loc='upper right', frameon=True,
         framealpha=0.95, edgecolor='black')

plt.tight_layout()
fig2.savefig('aufg7_shift_comparison_methods.svg', format='svg', bbox_inches='tight', dpi=150)
fig2.savefig('aufg7_shift_comparison_methods.pdf', bbox_inches='tight', dpi=150)
print("✓ Shift-Vergleich gespeichert: aufg7_shift_comparison_methods.svg / .pdf")


# ==============================================================================
# PLOT 3: PEAK-DETAIL MIT FIT
# ==============================================================================

fig3, axes = plt.subplots(1, 2, figsize=(16, 7))

# A1.1 Peak-Fit
peak_A11_fit, unc_A11_fit, beta_fine_A11, R_fit_A11, data_A11 = fit_peak(
    dfA11, 6.0, 8.0, plot_fit=True)

axes[0].plot(data_A11['beta'], data_A11['R_smooth'], 'o-',
            linewidth=2, markersize=7, label='A1.1 Daten', color='#1f77b4', alpha=0.7)
axes[0].plot(beta_fine_A11, R_fit_A11, '--', linewidth=3,
            label=f'Gauss-Fit: $\\beta_{{max}} = {peak_A11_fit:.3f}^\\circ$',
            color='red', alpha=0.9)
axes[0].axvline(peak_A11_fit, color='red', linestyle=':', linewidth=2.5)
axes[0].set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
axes[0].set_ylabel(r'$R$ (Counts)', fontweight='bold')
axes[0].set_title('A1.1: Haupt-Peak', fontweight='bold', fontsize=16)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=12)

# A3 Peak-Fit
peak_A3_fit, unc_A3_fit, beta_fine_A3, R_fit_A3, data_A3 = fit_peak(
    dfA3, 6.0, 8.0, plot_fit=True)

axes[1].plot(data_A3['beta'], data_A3['R_smooth'], 's-',
            linewidth=2, markersize=7, label='A3 Daten', color='#ff7f0e', alpha=0.7)
axes[1].plot(beta_fine_A3, R_fit_A3, '--', linewidth=3,
            label=f'Gauss-Fit: $\\beta_{{max}} = {peak_A3_fit:.3f}^\\circ$',
            color='red', alpha=0.9)
axes[1].axvline(peak_A3_fit, color='red', linestyle=':', linewidth=2.5)
axes[1].set_xlabel(r'$\beta$ ($^\circ$)', fontweight='bold')
axes[1].set_ylabel(r'$R$ (Counts)', fontweight='bold')
axes[1].set_title('A3: Haupt-Peak', fontweight='bold', fontsize=16)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12)

plt.tight_layout()
fig3.savefig('aufg7_peak_fitting.svg', format='svg', bbox_inches='tight', dpi=150)
fig3.savefig('aufg7_peak_fitting.pdf', bbox_inches='tight', dpi=150)
print("✓ Peak-Fitting gespeichert: aufg7_peak_fitting.svg / .pdf")


# ==============================================================================
# FINALER OUTPUT
# ==============================================================================

print("\n\n" + "=" * 80)
print("ZUSAMMENFASSUNG FÜR PROTOKOLL")
print("=" * 80)
print()
print(f"Systematischer Shift:           Δβ = {final_shift:+.4f}°")
print(f"Winkelunsicherheit:             u_β = {final_uncertainty:.4f}°")
print()
print("Diese Werte bedeuten:")
print(f"  1. A3 ist im Mittel um {abs(final_shift):.3f}° verschoben")
print(f"  2. Die Unsicherheit von {final_uncertainty:.3f}° kommt von:")
print("     - Unterschiedlichen Beugungsmustern bei verschiedenen Orientierungen")
print("     - Unpräziser Kristallpositionierung/Ausrichtung")
print(f"  3. Diese u_β = {final_uncertainty:.4f}° muss quadratisch zur")
print("     instrumentellen Unsicherheit addiert werden")
print()
print("Formel für Gesamtunsicherheit:")
print("  u_gesamt = sqrt(u_instrumentell² + u_β²)")
print()
print("=" * 80)

print("\n✓✓✓ Alle Plots und Analysen abgeschlossen! ✓✓✓\n")

plt.show()

