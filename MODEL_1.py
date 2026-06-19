#!/usr/bin/env python3
"""
Exoplanet Detection Pipeline - ISRO Hackathon
USING LIGHTKURVE'S BUILT-IN BLS 
Fixed for version compatibility
"""

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

print("="*50)
print("ISRO BAH 2026 - Exoplanet Detection Pipeline")
print("Using Lightkurve's Box-Least Squares (BLS)")
print("="*50)

# ============================================
# STEP 1: Download data
# ============================================
print("\n[1/5] Downloading TESS data...")

# Try L 98-59 first
search_result = lk.search_lightcurve('L 98-59', mission='TESS')
print(f"Found {len(search_result)} total sectors")

# Download only the first sector
lc = search_result[0].download()
print(f"✅ Downloaded successfully!")

# ============================================
# STEP 2: Clean the data
# ============================================
print("\n[2/5] Cleaning data...")

# Remove outliers
lc_clean = lc.remove_outliers(sigma=5.0)

# Remove long-term trends
lc_flat = lc_clean.flatten(window_length=401)

print(f"✅ Data cleaned: {len(lc_flat.flux)} points")

# ============================================
# STEP 3: Run BLS detection (compatible version)
# ============================================
print("\n[3/5] Running BLS planet search...")
print("Searching for periods between 0.5 and 20 days...")

# Create an array of periods to search (0.5 to 10 days, 10000 steps)
periods = np.linspace(0.5, 10, 10000)

# Create an array of transit durations (as fraction of period)
# Typical transit duration is 0.01 to 0.1 times the period
durations = np.linspace(0.01, 0.1, 10)

# Run BLS
bls = lc_flat.to_periodogram(method='bls', period=periods, duration=durations)

# Get the best period
best_period = bls.period_at_max_power
best_power = bls.power.max()

print(f"\n✅ BLS complete!")
print(f"Best period found: {best_period:.5f} days")
print(f"Detection power: {best_power:.2f}")

# ============================================
# STEP 4: Fold the light curve
# ============================================
print("\n[4/5] Creating phase-folded transit...")

# Fold at the best period
folded_lc = lc_flat.fold(period=best_period)

# Bin to see transit clearly
binned_folded = folded_lc.bin(time_bin_size=0.01)

# Calculate transit depth
transit_flux = binned_folded.flux.value
transit_depth = (1 - np.min(transit_flux)) * 100

print(f"Transit depth: {transit_depth:.4f}%")

# ============================================
# STEP 5: Calculate planet radius
# ============================================
print("\n[5/5] Calculating physical properties...")

# L 98-59 star radius = 0.32 solar radii
solar_radius_km = 695700
earth_radius_km = 6371
stellar_radius_km = 0.32 * solar_radius_km

depth_decimal = transit_depth / 100
planet_radius_km = np.sqrt(depth_decimal) * stellar_radius_km
planet_radius_earth = planet_radius_km / earth_radius_km

print("="*50)
print("DETECTION RESULTS")
print("="*50)
print(f"📅 Orbital Period: {best_period:.5f} days")
print(f"📉 Transit Depth: {transit_depth:.5f}%")
print(f"🪐 Planet Radius: {planet_radius_earth:.2f} Earth radii ({planet_radius_km:.0f} km)")
print("="*50)

# Signal strength interpretation
print("\n📊 Signal Strength:")
if best_power > 10:
    print("   ✅ Very strong detection - Real planet very likely!")
elif best_power > 5:
    print("   ✅ Good detection - Promising candidate")
elif best_power > 3:
    print("   ⚠️ Weak signal - Possible but needs validation")
else:
    print("   ❌ No significant detection - Try a different star")

# ============================================
# STEP 6: Plot results
# ============================================
print("\nGenerating plots...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Plot 1: Raw light curve
ax1.plot(lc_flat.time.value, lc_flat.flux.value, 'k.', markersize=1, alpha=0.5)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Relative Flux')
ax1.set_title('L 98-59 Light Curve (Cleaned)')

# Plot 2: BLS Periodogram
periods_plot = bls.period.value
powers_plot = bls.power.value
ax2.plot(periods_plot, powers_plot, 'b-', lw=0.5)
ax2.axvline(best_period.value, color='red', linestyle='--', linewidth=2, 
            label=f'Best period: {best_period.value:.3f} days')
ax2.set_xlabel('Period (days)')
ax2.set_ylabel('BLS Power')
ax2.set_title('BLS Periodogram')
ax2.legend()
ax2.set_xlim(0, 15)

# Plot 3: Phase-folded transit
folded_lc.plot(ax=ax3, marker='.', markersize=2, linestyle='none', alpha=0.3)
binned_folded.plot(ax=ax3, color='red', linewidth=2, label='Binned transit')
ax3.set_title(f'Phase-Folded Transit at P = {best_period.value:.3f} days')
ax3.set_xlabel('Phase')
ax3.set_ylabel('Relative Flux')
ax3.legend()

plt.tight_layout()
plt.savefig('exoplanet_detection.png', dpi=150)
print("✅ Plot saved as 'exoplanet_detection.png'")

plt.show()

print("\n🎉 Pipeline complete!")

# ============================================
# Try another star if needed
# ============================================
print("\n" + "="*50)
print("TIPS")
print("="*50)
print("If the folded transit doesn't show a clear dip:")
print("   Edit line 25 and change 'L 98-59' to 'TRAPPIST-1'")
print("   OR change to 'HD 209458'")
print("\nTo save time in the future, you can decrease:")
print("   periods = np.linspace(0.5, 10, 5000)  # fewer = faster")
