#!/usr/bin/env python3
import math

print("╔════════════════════════════════════════════════════════════════════════════╗")
print("║  BERECHNUNG DER STRAHLENBELASTUNG VON CS-137 KALIBRIERQUELLE (1M ENTFERNT)  ║")
print("╚════════════════════════════════════════════════════════════════════════════╝\n")

# SCHRITT 1: Aktivität zum jetzigen Zeitpunkt berechnen
print("SCHRITT 1: AKTIVITÄT ZUM JETZIGEN ZEITPUNKT")
print("─" * 75)
T_half = 30.2  # Jahre
lbd = math.log(2) / (T_half * 365.25 * 24 * 3600)
A_0 = 37 * 1000  # Bq (1997)
t = 28  # Jahre (von 1997 zu 2025)
A = A_0 * math.exp(-lbd * t * 365.25 * 24 * 3600)

print(f"Halbwertszeit Cs-137: T₁/₂ = {T_half} Jahre")
print(f"Zerfallskonstante: λ = ln(2)/(T₁/₂) = {lbd:.4e} s⁻¹")
print(f"Aktivität 1997: A₀ = 37 kBq = {A_0:,} Bq")
print(f"Zeitraum: t = {t} Jahre")
print(f"Aktivität 2025: A = A₀ × e^(-λt) = {A:.2f} Bq")
print(f"Reduktion: {A/A_0*100:.1f}% der Originalaktivität\n")

# SCHRITT 2: Raumwinkel abschätzen
print("SCHRITT 2: RAUMWINKEL DES KÖRPERS IN 1M ENTFERNUNG")
print("─" * 75)
Omega = 0.72  # sr
Omega_full = 4 * math.pi
fraction = Omega / Omega_full

print(f"Körperoberfläche von vorne in 1m Abstand: ~0.5m × 1.7m ≈ 0.85 m²")
print(f"Kugelfläche in 1m Radius: 4πr² = 4π m² ≈ 12.6 m²")
print(f"Geschätzter Raumwinkel: Ω ≈ {Omega} sr")
print(f"Vollständige Sphäre: 4π ≈ {Omega_full:.3f} sr")
print(f"Strahlungsanteil auf Körper: {Omega}/{Omega_full:.3f} = {fraction:.4e}\n")

# SCHRITT 3: Energieübertragung pro Jahr
print("SCHRITT 3: ENERGIEÜBERTRAGUNG PRO JAHR")
print("─" * 75)
E_gamma_MeV = 0.662
E_gamma_J = E_gamma_MeV * 1.602e-13
seconds_per_year = 365.25 * 24 * 3600
photons_per_year = A * seconds_per_year
energy_per_year = photons_per_year * E_gamma_J * fraction

print(f"Energie eines γ-Quants (Cs-137): E_γ = {E_gamma_MeV} MeV = {E_gamma_J:.4e} J")
print(f"Sekunden pro Jahr: {seconds_per_year:.3e} s")
print(f"Anzahl γ-Quanten pro Jahr: N = A × t = {photons_per_year:.3e}")
print(f"Energie auf Körper pro Jahr: E = N × E_γ × (Ω/4π)")
print(f"                             E = {energy_per_year:.3e} J\n")

# SCHRITT 4: Effektive Dosis
print("SCHRITT 4: EFFEKTIVE DOSIS")
print("─" * 75)
m_body = 70  # kg
D = energy_per_year / m_body  # Gy
Q = 1  # Qualitätsfaktor für Gammastrahlung
H = D * Q  # Sv

print(f"Körpergewicht: m = {m_body} kg")
print(f"Energiedosis: D = E/m = {energy_per_year:.3e} J / {m_body} kg = {D:.6f} Gy")
print(f"            D = {D*100:.4f} rad")
print(f"Qualitätsfaktor (γ-Strahlung): Q = {Q}")
print(f"Effektive Dosis: H = Q × D = {H:.4e} Sv")
print(f"                H = {H*1000:.4f} mSv\n")

# ZUSAMMENFASSUNG UND VERGLEICH
print("╔════════════════════════════════════════════════════════════════════════════╗")
print("║                            ZUSAMMENFASSUNG                                 ║")
print("╚════════════════════════════════════════════════════════════════════════════╝\n")
print(f"Zusätzliche jährliche Strahlenbelastung: {H*1000:.3f} mSv/Jahr")
print(f"\nZum Vergleich:")
print(f"  • Natürliche jährliche Strahlenbelastung:  ~2-3 mSv/Jahr")
print(f"  • Grenzwert für Strahlenarbeiter:          20-50 mSv/Jahr")
print(f"  • Akute Strahlenkrankheit:                 >100 mSv")
print(f"\nDiese Cs-137 Quelle würde in 1m Entfernung zusätzlich")
print(f"{H*1000:.3f} mSv/Jahr zur Strahlenbelastung hinzufügen.")
print(f"Das ist etwa {H*1000/2.5:.1f}x die natürliche Strahlenbelastung.\n")

