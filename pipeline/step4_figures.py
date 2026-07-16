# =============================================================================
# PUBLICATION 1 — STEP 4: CORE FIGURES
# Run AFTER step3c.
# Output: figure1_mortality_by_race.png, figure2_icu_los_by_race.png,
#         figure3_readmission.png, figure4_forest_plot_mortality.png
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "matplotlib", "seaborn"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')
FIG  = os.path.join(OUT, 'figures/')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family':'Arial','font.size':11,'axes.titlesize':13,'axes.labelsize':12,
    'xtick.labelsize':10,'ytick.labelsize':10,'figure.dpi':300,
    'savefig.dpi':300,'savefig.bbox':'tight',
})

PALETTE = {
    'White':'#2166AC','Black':'#D6604D','Hispanic':'#4DAC26','Asian':'#7B3294',
    'Native American/Alaska Native':'#E08214','Pacific Islander':'#1A9641','Other':'#878787'
}

try:
    cohort = pd.read_csv(OUT + 'neurosurg_cohort_enriched.csv', low_memory=False)
except FileNotFoundError:
    cohort = pd.read_csv(OUT + 'neurosurg_cohort_raw.csv', low_memory=False)

RACE_ORDER = ['White','Black','Hispanic','Asian','Native American/Alaska Native','Pacific Islander','Other']
RACE_ORDER = [r for r in RACE_ORDER if r in cohort['race_group'].unique()]
INS_ORDER  = [i for i in ['Private','Medicare','Medicaid','Self-Pay','Other']
              if i in cohort['insurance_group'].unique()]

# -----------------------------------------------------------------------------
# FIGURE 1: In-hospital mortality by race
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9,5))
data = []
for r in RACE_ORDER:
    s = cohort[cohort['race_group']==r]['in_hospital_mortality'].dropna()
    p = s.mean(); n = len(s)
    data.append({'race':r,'n':n,'val':p*100,'ci':1.96*np.sqrt(p*(1-p)/n)*100})
df = pd.DataFrame(data)
ax.bar(df['race'], df['val'], color=[PALETTE.get(r,'#878787') for r in df['race']],
       edgecolor='white', linewidth=0.8, width=0.6)
ax.errorbar(df['race'], df['val'], yerr=df['ci'], fmt='none', color='black', capsize=4, linewidth=1.2)
for i, row in df.iterrows():
    ax.text(i, row['val']+row['ci']+0.5, f"n={row['n']:,}", ha='center', va='bottom', fontsize=8.5, color='#444')
ax.set_ylabel('In-Hospital Mortality (%)'); ax.set_title('In-Hospital Mortality by Race/Ethnicity\nNeurosurgical ICU Cohort (MIMIC-IV)', pad=12)
plt.xticks(rotation=20, ha='right'); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, df['val'].max()*1.35); plt.tight_layout()
plt.savefig(FIG+'figure1_mortality_by_race.png'); plt.close(); print("Figure 1 saved.")

# -----------------------------------------------------------------------------
# FIGURE 2: ICU LOS boxplot by race
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9,5))
bp = ax.boxplot([cohort[cohort['race_group']==r]['icu_los_days'].dropna().values for r in RACE_ORDER],
                labels=RACE_ORDER, patch_artist=True,
                medianprops=dict(color='black',linewidth=2),
                flierprops=dict(marker='o',markersize=2,alpha=0.3))
for patch, r in zip(bp['boxes'], RACE_ORDER):
    patch.set_facecolor(PALETTE.get(r,'#878787')); patch.set_alpha(0.8)
ax.set_ylabel('ICU Length of Stay (Days)'); ax.set_title('ICU Length of Stay by Race/Ethnicity\nNeurosurgical ICU Cohort (MIMIC-IV)', pad=12)
plt.xticks(rotation=20, ha='right'); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, np.percentile(cohort['icu_los_days'].dropna(),95)*1.2)
plt.tight_layout(); plt.savefig(FIG+'figure2_icu_los_by_race.png'); plt.close(); print("Figure 2 saved.")

# -----------------------------------------------------------------------------
# FIGURE 3: 30-day readmission + time-to-ICU by race (2-panel)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1,2,figsize=(14,5))
ins_colors = ['#2166AC','#4DAC26','#D6604D','#E08214','#878787']

# Panel A: readmission by race
data_r = []
for r in RACE_ORDER:
    s = cohort[(cohort['race_group']==r)&(cohort['in_hospital_mortality']==0)]['readmit_30day'].dropna()
    if len(s)>10:
        p=s.mean(); data_r.append({'g':r,'val':p*100,'ci':1.96*np.sqrt(p*(1-p)/len(s))*100})
df_r = pd.DataFrame(data_r)
axes[0].bar(df_r['g'],df_r['val'],color=[PALETTE.get(r,'#878787') for r in df_r['g']],edgecolor='white',width=0.6)
axes[0].errorbar(df_r['g'],df_r['val'],yerr=df_r['ci'],fmt='none',color='black',capsize=4,linewidth=1.2)
axes[0].set_title('30-Day Readmission by Race/Ethnicity')
axes[0].set_ylabel('30-Day Readmission Rate (%)'); axes[0].spines[['top','right']].set_visible(False)
plt.setp(axes[0].get_xticklabels(),rotation=20,ha='right')

# Panel B: time to ICU by race (if available)
if 'time_to_icu_hours' in cohort.columns:
    data_t = []
    for r in RACE_ORDER:
        s = cohort[(cohort['race_group']==r)&(cohort['direct_icu_admission']==0)]['time_to_icu_hours'].dropna()
        if len(s)>10:
            data_t.append({'g':r,'val':s.median(),'q25':s.quantile(0.25),'q75':s.quantile(0.75)})
    df_t = pd.DataFrame(data_t)
    axes[1].bar(df_t['g'],df_t['val'],color=[PALETTE.get(r,'#878787') for r in df_t['g']],edgecolor='white',width=0.6)
    axes[1].errorbar(df_t['g'],df_t['val'],
                     yerr=[df_t['val']-df_t['q25'], df_t['q75']-df_t['val']],
                     fmt='none',color='black',capsize=4,linewidth=1.2)
    axes[1].set_title('Time to ICU Admission by Race/Ethnicity\n(Median [IQR], Non-Direct Admissions)')
    axes[1].set_ylabel('Hours from Hospital Admission to ICU')
    axes[1].spines[['top','right']].set_visible(False)
    plt.setp(axes[1].get_xticklabels(),rotation=20,ha='right')
else:
    axes[1].axis('off'); axes[1].text(0.5,0.5,'Time-to-ICU\nnot available',ha='center',va='center',transform=axes[1].transAxes)

plt.suptitle('Readmission and Time-to-ICU Disparities — Neurosurgical ICU Cohort (MIMIC-IV)', fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(FIG+'figure3_readmission_tti.png'); plt.close(); print("Figure 3 saved.")

# -----------------------------------------------------------------------------
# FIGURE 4: Forest plot — fully adjusted OR for mortality by race
# -----------------------------------------------------------------------------
try:
    mort_or  = pd.read_csv(OUT+'fully_adjusted_mortality.csv')
    race_or  = mort_or[mort_or['Variable'].str.contains('race_group')].copy()
    race_or['Label'] = race_or['Variable'].str.extract(r"\.'([^']+)'")
    race_or  = race_or.dropna(subset=['Label']).sort_values('OR')
    if len(race_or)>0:
        fig, ax = plt.subplots(figsize=(9, max(4, len(race_or)*0.8+2)))
        y_pos   = range(len(race_or))
        colors  = [PALETTE.get(l,'#878787') for l in race_or['Label']]
        ax.scatter(race_or['OR'], list(y_pos), color=colors, s=80, zorder=3)
        ax.hlines(list(y_pos), race_or['CI_95_low'], race_or['CI_95_high'], color=colors, linewidth=2.5, zorder=2)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_yticks(list(y_pos)); ax.set_yticklabels(race_or['Label'].tolist())
        for i, (_, row) in enumerate(race_or.iterrows()):
            p_str = "<0.001" if row['p_value']<0.001 else f"{row['p_value']:.3f}"
            ax.text(race_or['CI_95_high'].max()*1.05, i,
                    f"aOR={row['OR']:.2f} [{row['CI_95_low']:.2f}–{row['CI_95_high']:.2f}], p={p_str}",
                    va='center', fontsize=9)
        ax.set_xlabel('Adjusted Odds Ratio (ref: White)')
        ax.set_title('Adjusted OR for In-Hospital Mortality by Race/Ethnicity\n(Fully Adjusted Model)', pad=12)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); plt.savefig(FIG+'figure4_forest_plot_mortality.png'); plt.close(); print("Figure 4 saved.")
except FileNotFoundError:
    print("Figure 4 skipped — run Step 3c first.")

print(f"\n=== STEP 4 COMPLETE === Figures saved to: {FIG}")
