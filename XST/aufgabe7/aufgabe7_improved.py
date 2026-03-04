"""
Aufgabe 7: Winkelunsicherheit - Verbesserte Version
Mit detaillierter Analyse und visueller Validierung
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter, correlate
from typing import Tuple, List, Dict

# ==============================================================================
# SETUP
# ==============================================================================

plt.rcParams["text.usetex"] = False  # Deaktiviere LaTeX erstmal für debugging
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11


# ==============================================================================
# DATEN LADEN UND VORBEREITEN
# ==============================================================================

def load_data():
    """Lädt und bereitet Daten vor."""
    dfA11 = pd.read_excel('data/xst.xlsx', sheet_name='A1.1')
    dfA3 = pd.read_excel('data/xst.xlsx', sheet_name='A3')

    # Bereinigung
    for df in [dfA11, dfA3]:
        df.dropna(subset=['beta', 'R'], inplace=True)
        df['beta'] = df['beta'].astype(float)
        df['R'] = df['R'].astype(float)
        df.sort_values('beta', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Glätten
    dfA11['R_smooth'] = savgol_filter(dfA11['R'], window_length=11, polyorder=3)
    dfA3['R_smooth'] = savgol_filter(dfA3['R'], window_length=9, polyorder=3)

    return dfA11, dfA3


# ==============================================================================
# SHIFT-BERECHNUNG MIT VERSCHIEDENEN METHODEN
# ==============================================================================

def calculate_shift_cross_correlation(
    dfA11: pd.DataFrame,
    dfA3: pd.DataFrame,
    beta_min: float,
    beta_max: float,
    shift_range: Tuple[float, float] = (-1.0, 1.0),
    n_shifts: int = 200
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Berechnet Shift mit Cross-Correlation auf gleichmäßigem Gitter.

    Returns:
        optimal_shift, shift_array, correlation_array
    """
    # Daten im Fenster
    mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    df1 = dfA11[mask1].copy()
    df3 = dfA3[mask3].copy()

    # Interpolation auf feines Gitter
    beta_grid = np.linspace(beta_min, beta_max, 500)
    f1 = interp1d(df1['beta'], df1['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    f3 = interp1d(df3['beta'], df3['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')

    R1_grid = f1(beta_grid)

    # Normalisiere für Formvergleich
    R1_norm = (R1_grid - np.mean(R1_grid)) / np.std(R1_grid)

    # Teste verschiedene Shifts
    shifts = np.linspace(shift_range[0], shift_range[1], n_shifts)
    correlations = []

    for shift in shifts:
        # Shift bedeutet: A3 um +shift nach rechts → R3(beta - shift)
        R3_grid = f3(beta_grid - shift)

        valid_mask = np.isfinite(R1_grid) & np.isfinite(R3_grid)
        if np.sum(valid_mask) < 10:
            correlations.append(-1)
            continue

        R3_norm = (R3_grid - np.mean(R3_grid)) / np.std(R3_grid)

        # Korrelation berechnen
        corr = np.corrcoef(R1_norm[valid_mask], R3_norm[valid_mask])[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)
    best_idx = np.argmax(correlations)
    optimal_shift = shifts[best_idx]

    return optimal_shift, shifts, correlations


def calculate_shift_mse(
    dfA11: pd.DataFrame,
    dfA3: pd.DataFrame,
    beta_min: float,
    beta_max: float,
    shift_range: Tuple[float, float] = (-1.0, 1.0)
) -> Tuple[float, float]:
    """
    Berechnet Shift durch MSE-Minimierung.
    """
    mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)
    mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)

    df1 = dfA11[mask1].copy()
    df3 = dfA3[mask3].copy()

    if len(df1) < 5 or len(df3) < 5:
        return np.nan, np.inf

    # Interpolation
    beta_grid = np.linspace(beta_min, beta_max, 500)
    f1 = interp1d(df1['beta'], df1['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    f3 = interp1d(df3['beta'], df3['R_smooth'], kind='cubic',
                  bounds_error=False, fill_value='extrapolate')

    def loss(shift):
        R1 = f1(beta_grid)
        R3 = f3(beta_grid - shift)

        valid = np.isfinite(R1) & np.isfinite(R3)
        if np.sum(valid) < 10:
            return 1e10

        # Normalisieren
        R1_norm = (R1[valid] - np.mean(R1[valid])) / np.std(R1[valid])
        R3_norm = (R3[valid] - np.mean(R3[valid])) / np.std(R3[valid])

        return np.mean((R1_norm - R3_norm)**2)

    result = minimize_scalar(loss, bounds=shift_range, method='bounded')
    return result.x, result.fun


# ==============================================================================
# MULTI-FENSTER ANALYSE
# ==============================================================================

def analyze_multiple_windows(dfA11, dfA3, windows):
    """Analysiert mehrere Fenster."""
    results = []

    print("\nFENSTER-ANALYSE:")
    print("=" * 80)

    for i, (beta_min, beta_max) in enumerate(windows):
        # Methode 1: MSE
        shift_mse, loss_mse = calculate_shift_mse(
            dfA11, dfA3, beta_min, beta_max, shift_range=(-1.0, 1.0)
        )

        # Methode 2: Cross-Correlation
        shift_corr, shifts, correlations = calculate_shift_cross_correlation(
            dfA11, dfA3, beta_min, beta_max, shift_range=(-1.0, 1.0)
        )

        results.append({
            'window': (beta_min, beta_max),
            'shift_mse': shift_mse,
            'loss_mse': loss_mse,
            'shift_corr': shift_corr,
            'max_corr': np.max(correlations),
            'shifts': shifts,
            'correlations': correlations
        })

        print(f"Fenster {i+1}: [{beta_min:.1f}, {beta_max:.1f}]°")
        print(f"  MSE-Methode:     Δβ = {shift_mse:+.4f}° (Loss: {loss_mse:.4f})")
        print(f"  Korrelation:     Δβ = {shift_corr:+.4f}° (Max-Corr: {np.max(correlations):.4f})")
        print()

    return results


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_correlation_analysis(results, windows):
    """Plottet Korrelation vs. Shift für jedes Fenster."""
    n_windows = len(results)
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_windows > 1 else [axes]

    for i, (res, window) in enumerate(zip(results, windows)):
        ax = axes[i]
        ax.plot(res['shifts'], res['correlations'], linewidth=2, color='steelblue')
        ax.axvline(res['shift_corr'], color='red', linestyle='--', linewidth=2,
                  label=f"Δβ = {res['shift_corr']:+.4f}°")
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Shift (°)')
        ax.set_ylabel('Korrelation')
        ax.set_title(f"Fenster [{window[0]:.1f}, {window[1]:.1f}]°")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Leere Subplots ausblenden
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def plot_aligned_comparison(dfA11, dfA3, shift, title_suffix=""):
    """Plottet Originalkurven und verschobene Kurven zum Vergleich."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Gemeinsamer Bereich
    beta_min, beta_max = 2.0, 10.0
    mask1 = (dfA11['beta'] >= beta_min) & (dfA11['beta'] <= beta_max)

    # Plot 1: Original
    ax1.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
            linewidth=2.5, label='A1.1', alpha=0.9, color='#1f77b4')

    mask3 = (dfA3['beta'] >= beta_min) & (dfA3['beta'] <= beta_max)
    ax1.plot(dfA3.loc[mask3, 'beta'], dfA3.loc[mask3, 'R_smooth'],
            linewidth=2.5, label='A3 (original)', alpha=0.9, color='#ff7f0e')

    ax1.set_xlabel('Beta (°)')
    ax1.set_ylabel('R (Counts, geglättet)')
    ax1.set_title(f'Vor Shift-Korrektur {title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Nach Korrektur
    dfA3_shifted = dfA3.copy()
    dfA3_shifted['beta'] = dfA3['beta'] + shift
    mask3_shifted = (dfA3_shifted['beta'] >= beta_min) & (dfA3_shifted['beta'] <= beta_max)

    ax2.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
            linewidth=2.5, label='A1.1', alpha=0.9, color='#1f77b4')
    ax2.plot(dfA3_shifted.loc[mask3_shifted, 'beta'], dfA3_shifted.loc[mask3_shifted, 'R_smooth'],
            linewidth=2.5, label=f'A3 (verschoben um Δβ = {shift:+.4f}°)',
            alpha=0.9, color='#ff7f0e')

    ax2.set_xlabel('Beta (°)')
    ax2.set_ylabel('R (Counts, geglättet)')
    ax2.set_title(f'Nach Shift-Korrektur (sollten besser übereinanderliegen) {title_suffix}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_window_detail(dfA11, dfA3, window, shift, idx):
    """Detailansicht für ein Fenster."""
    beta_min, beta_max = window
    padding = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, use_shift, title in zip(axes, [False, True],
                                     ['Vor Korrektur', 'Nach Korrektur']):
        mask1 = (dfA11['beta'] >= beta_min - padding) & (dfA11['beta'] <= beta_max + padding)

        # A1.1 plotten
        ax.plot(dfA11.loc[mask1, 'beta'], dfA11.loc[mask1, 'R_smooth'],
                linewidth=3, label='A1.1', alpha=0.8, color='#1f77b4',
                marker='o', markersize=6)

        # A3 (verschoben oder nicht)
        if use_shift:
            dfA3_plot = dfA3.copy()
            dfA3_plot['beta'] = dfA3['beta'] + shift
            mask3 = (dfA3_plot['beta'] >= beta_min - padding) & (dfA3_plot['beta'] <= beta_max + padding)
            label3 = f'A3 (Δβ={shift:+.3f}°)'
        else:
            dfA3_plot = dfA3
            mask3 = (dfA3_plot['beta'] >= beta_min - padding) & (dfA3_plot['beta'] <= beta_max + padding)
            label3 = 'A3 (original)'

        ax.plot(dfA3_plot.loc[mask3, 'beta'], dfA3_plot.loc[mask3, 'R_smooth'],
                linewidth=3, label=label3, alpha=0.8, color='#ff7f0e',
                marker='s', markersize=6)

        # Fenster markieren
        if use_shift:
            ax.axvspan(beta_min + shift, beta_max + shift, alpha=0.15, color='orange')
        else:
            ax.axvspan(beta_min, beta_max, alpha=0.15, color='gray')

        ax.set_xlabel('Beta (°)', fontweight='bold')
        ax.set_ylabel('R (Counts)', fontweight='bold')
        ax.set_title(f'Fenster {idx}: {title}', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    return fig


# ==============================================================================
# HAUPTANALYSE
# ==============================================================================

def main():
    print("=" * 80)
    print("AUFGABE 7: WINKELUNSICHERHEIT - VERBESSERTE ANALYSE")
    print("=" * 80)

    # Daten laden
    print("\n1. Lade Daten...")
    dfA11, dfA3 = load_data()
    print(f"   A1.1: {len(dfA11)} Punkte, Beta: {dfA11['beta'].min():.1f}-{dfA11['beta'].max():.1f}°")
    print(f"   A3:   {len(dfA3)} Punkte, Beta: {dfA3['beta'].min():.1f}-{dfA3['beta'].max():.1f}°")

    # Fenster definieren (strukturreiche Bereiche)
    windows = [
        (3.2, 4.0),   # Steiler Anstieg - Start
        (3.8, 4.8),   # Anstieg - Hauptbereich
        (4.5, 5.5),   # Plateau-Bereich
        (5.8, 6.8),   # Peak-Bereich (1. Peak bei A3)
        (6.5, 7.5),   # Haupt-Peak (bei A1.1)
        (7.5, 8.5),   # Abstieg/zweiter Peak
        (8.5, 9.5),   # Plateau oben
    ]

    print("\n2. Analysiere Fenster...")
    results = analyze_multiple_windows(dfA11, dfA3, windows)

    # Statistik über beide Methoden
    shifts_mse = np.array([r['shift_mse'] for r in results])
    shifts_corr = np.array([r['shift_corr'] for r in results])

    valid_mse = shifts_mse[np.isfinite(shifts_mse)]
    valid_corr = shifts_corr[np.isfinite(shifts_corr)]

    mean_mse = np.mean(valid_mse)
    std_mse = np.std(valid_mse, ddof=1)

    mean_corr = np.mean(valid_corr)
    std_corr = np.std(valid_corr, ddof=1)

    print("=" * 80)
    print("STATISTISCHE AUSWERTUNG:")
    print("=" * 80)
    print("\nMSE-Methode:")
    print(f"  Mittlerer Shift:  Δβ = {mean_mse:+.4f}°")
    print(f"  Standardabw.:     u_β = {std_mse:.4f}°")
    print(f"  Einzelwerte:      {', '.join([f'{s:+.3f}' for s in valid_mse])}")

    print("\nKorrelations-Methode:")
    print(f"  Mittlerer Shift:  Δβ = {mean_corr:+.4f}°")
    print(f"  Standardabw.:     u_β = {std_corr:.4f}°")
    print(f"  Einzelwerte:      {', '.join([f'{s:+.3f}' for s in valid_corr])}")

    # Wähle Korrelations-Methode als Hauptergebnis (robuster)
    final_shift = mean_corr
    final_uncertainty = std_corr

    print("\n" + "=" * 80)
    print("FINALES ERGEBNIS (Korrelations-Methode):")
    print("=" * 80)
    print(f"  Systematischer Shift:        Δβ = {final_shift:+.4f}°")
    print(f"  Winkelunsicherheit:          u_β = {final_uncertainty:.4f}°")
    print(f"  |Δβ|:                        {abs(final_shift):.4f}°")
    print()
    print("INTERPRETATION:")
    print(f"  → A3 liegt systematisch bei kleineren Beta-Werten als A1.1")
    print(f"  → A3 muss um {final_shift:+.4f}° nach rechts verschoben werden")
    print(f"  → Die Streuung von {final_uncertainty:.4f}° ist die zusätzliche")
    print(f"    Winkelunsicherheit aus der Kristallorientierung")
    print()
    print("Gesamtunsicherheit (quadratische Addition):")
    u_device = 0.05  # Beispiel: gerätebdingte Unsicherheit
    u_total = np.sqrt(u_device**2 + final_uncertainty**2)
    print(f"  u_gesamt = sqrt({u_device}² + {final_uncertainty:.4f}²) = {u_total:.4f}°")
    print("=" * 80)

    # Plots erstellen
    print("\n3. Erstelle Plots...")

    # Plot: Korrelations-Kurven
    print("   - Korrelations-Analyse pro Fenster")
    fig_corr = plot_correlation_analysis(results, windows)
    fig_corr.savefig('shift_correlation_analysis.png', dpi=150, bbox_inches='tight')
    print("      → shift_correlation_analysis.png")

    # Plot: Vergleich mit finalem Shift
    print("   - Vorher/Nachher-Vergleich (Korrelations-Shift)")
    fig_comp = plot_aligned_comparison(dfA11, dfA3, final_shift,
                                       title_suffix=f"(Δβ={final_shift:+.4f}°)")
    fig_comp.savefig('shift_alignment_comparison.png', dpi=150, bbox_inches='tight')
    print("      → shift_alignment_comparison.png")

    # Detail-Plots für ausgewählte Fenster
    print("   - Detail-Plots für ausgewählte Fenster")
    for i in [1, 3, 4]:  # Fenster mit interessanten Features
        if i < len(windows):
            fig_det = plot_window_detail(dfA11, dfA3, windows[i],
                                         results[i]['shift_corr'], i+1)
            fig_det.savefig(f'shift_detail_fenster_{i+1}.png', dpi=150, bbox_inches='tight')
            print(f"      → shift_detail_fenster_{i+1}.png")

    # Zusammenfassung als Balkendiagramm
    fig_summary = plot_shift_summary(results, windows, final_shift, final_uncertainty)
    fig_summary.savefig('shift_summary.png', dpi=150, bbox_inches='tight')
    print("      → shift_summary.png")

    print("\n✓ Analyse abgeschlossen!")
    plt.show()

    return results, final_shift, final_uncertainty


def plot_shift_summary(results, windows, mean_shift, std_shift):
    """Zusammenfassung der Shifts."""
    fig, ax = plt.subplots(figsize=(12, 7))

    shifts_mse = [r['shift_mse'] for r in results]
    shifts_corr = [r['shift_corr'] for r in results]

    x = np.arange(len(windows))
    width = 0.35

    bars1 = ax.bar(x - width/2, shifts_mse, width, label='MSE-Methode',
                   alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, shifts_corr, width, label='Korrelations-Methode',
                   alpha=0.8, color='coral', edgecolor='black')

    # Mittelwert und Unsicherheitsbereich
    ax.axhline(mean_shift, color='red', linestyle='--', linewidth=2,
              label=f'Mittelwert (Korr.): {mean_shift:+.4f}°')
    ax.axhspan(mean_shift - std_shift, mean_shift + std_shift,
              alpha=0.2, color='red', label=f'±1σ: ±{std_shift:.4f}°')
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('Fenster-Index', fontweight='bold')
    ax.set_ylabel('Δβ (°)', fontweight='bold')
    ax.set_title('Geschätzte Shifts pro Fenster und Methode', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}\n[{w[0]:.1f},{w[1]:.1f}]"
                        for i, w in enumerate(windows)], fontsize=9)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    results, final_shift, final_uncertainty = main()

