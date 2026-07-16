# =============================================================================
# PUBLICATION 1 — STEP 4b: ADDITIONAL FIGURES
# Run AFTER step4.
# Output: figure5_heatmap.png, figure6_consort.png, figure7_subgroup_forest.png,
#         figure8_disposition.png, figure9_temporal_trends.png
# =============================================================================

import subprocess
subprocess.run(["pip", "install", "-q", "pandas", "numpy", "matplotlib", "seaborn"], check=False)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/mimic 4/'
OUT  = os.path.join(BASE, 'outputs/')
FIG  = os.path.join(OUT, 'figures/')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family':'Arial','font.size':11,'axes.titlesize':13,'axes.labelsize':12,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
})

PALETTE = {
    'White':'#2166AC','Black':'#D6604D','Hispanic':'#4DAC26','Asian':'#7B3294',
    'Native American/Alaska Native':'#E08214','Pacific Islander':'#1A9641','Other':'#878787'
}
SHORT = {'White':'White','Black':'Black','Hispanic':'Hispanic','Asian':'Asian',
         'Native American/Alaska Native':'Native Am.','Pacific Islander':'Pacific Isl.','Other':'Other'}

try:
    cohort = pd.read_csv(OUT+'neurosurg_cohort_enriched.csv', low_memory=False)
except FileNotFoundError:
    cohort = pd.read_csv(OUT+'neurosurg_cohort_raw.csv', low_memory=False)

RACE_ORDER = [r for r in ['White','Black','Hispanic','Asian','Native American/Alaska Native','Pacific Islander','Other']
              if r in cohort['race_group'].unique()]
INS_ORDER  = [i for i in ['Private','Medicare','Medicaid','Self-Pay','Other']
              if i in cohort['insurance_group'].unique()]

# -----------------------------------------------------------------------------
# FIGURE 5: HEATMAP — outcomes by race × insurance
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1,3,figsize=(18,6))
for ax, (outcome, title, cmap) in zip(axes, [
    ('in_hospital_mortality','In-Hospital Mortality (%)','Reds'),
    ('icu_los_days',         'Median ICU LOS (Days)',   'Blues'),
    ('readmit_30day',        '30-Day Readmission (%)',  'Oranges'),
]):
    matrix = {}
    for ins in INS_ORDER:
        col = []
        for race in RACE_ORDER:
            sub = cohort[(cohort['race_group']==race)&(cohort['insurance_group']==ins)]
            if outcome=='readmit_30day': sub = sub[sub['in_hospital_mortality']==0]
            col.append(np.nan if len(sub)<10 else
                       (sub[outcome].median() if outcome=='icu_los_days' else sub[outcome].mean()*100))
        matrix[ins] = col
    mdf = pd.DataFrame(matrix, index=[SHORT.get(r,r) for r in RACE_ORDER])
    sns.heatmap(mdf, ax=ax, cmap=cmap, annot=True, fmt='.1f', linewidths=0.5,
                linecolor='white', cbar_kws={'label':title}, annot_kws={'size':9}, mask=mdf.isna())
    ax.set_title(title, fontsize=12, pad=10); ax.set_xlabel('Insurance Status')
    ax.set_ylabel('Race/Ethnicity' if ax==axes[0] else '')
    ax.tick_params(axis='x',rotation=30); ax.tick_params(axis='y',rotation=0)
plt.suptitle('Neurosurgical ICU Outcomes by Race × Insurance Status\n(MIMIC-IV; cells n<10 suppressed)', fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig(FIG+'figure5_heatmap_race_x_insurance.png'); plt.close(); print("Figure 5 saved.")

# -----------------------------------------------------------------------------
# FIGURE 6: CONSORT FLOW DIAGRAM
# -----------------------------------------------------------------------------
n_final = len(cohort)
fig, ax  = plt.subplots(figsize=(10,12))
ax.set_xlim(0,10); ax.set_ylim(0,14); ax.axis('off')

def box(ax,x,y,w,h,text,fs=10,fc='#E8F4F8',ec='#2166AC'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,linewidth=1.5))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,multialignment='center')

def arr(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#444',lw=1.5))

def excl(ax,x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.1",facecolor='#FFF3CD',edgecolor='#E08214',linewidth=1.5))
    ax.text(x,y,text,ha='center',va='center',fontsize=9,multialignment='center')

box(ax,5,13,7,0.9,"All MIMIC-IV v3.0 ICU admissions\n(hosp + icu tables merged)")
arr(ax,5,12.55,5,11.85)
box(ax,5,11.5,7,0.9,"Filtered: ICD-10 neurosurgical diagnoses/procedures\n(TBI, hemorrhagic stroke, craniotomy, spine, brain tumor)")
excl(ax,8.8,11.5,2.8,0.9,"Excluded:\nICD-9 only\nNon-neurosurgical")
arr(ax,5,11.05,5,10.35)
box(ax,5,10.0,7,0.9,"First ICU stay per admission retained")
excl(ax,8.8,10.0,2.8,0.9,"Excluded:\nSubsequent ICU stays")
arr(ax,5,9.55,5,8.85)
box(ax,5,8.5,7,0.9,"Age ≥ 18 years")
excl(ax,8.8,8.5,2.8,0.9,"Excluded:\nAge < 18")
arr(ax,5,8.05,5,7.35)
box(ax,5,7.0,7,0.9,"ICU stay ≥ 24 hours")
excl(ax,8.8,7.0,2.8,0.9,"Excluded:\nICU stay < 24h")
arr(ax,5,6.55,5,5.85)
box(ax,5,5.5,7,0.9,"Known race/ethnicity (excluded 'Unknown')")
excl(ax,8.8,5.5,2.8,0.9,"Excluded:\nRace unknown/missing")
arr(ax,5,5.05,5,4.35)
box(ax,5,4.0,7,1.1,
    f"Final Analytic Cohort: N = {n_final:,}\nPrimary Outcome: In-Hospital Mortality\nSecondary: ICU LOS, 30-Day Readmission, Time to ICU",
    fs=10,fc='#D4EDDA',ec='#28A745')
ax.set_title("Patient Inclusion/Exclusion Flow Diagram\nNeurosurgical ICU Cohort — MIMIC-IV v3.0", fontsize=13, pad=15)
plt.tight_layout(); plt.savefig(FIG+'figure6_consort_flow.png'); plt.close(); print("Figure 6 saved.")

# -----------------------------------------------------------------------------
# FIGURE 7: SUBGROUP FOREST PLOT
# -----------------------------------------------------------------------------
try:
    sg_df    = pd.read_csv(OUT+'subgroup_mortality_by_diagnosis.csv')
    race_rows = sg_df[sg_df['Variable'].str.contains('race_group')].copy()
    race_rows['Label'] = race_rows['Variable'].str.extract(r"\.'([^']+)'")
    race_rows = race_rows.dropna(subset=['Label'])

    if len(race_rows)>0:
        subgroups = race_rows['subgroup'].unique()
        cmap      = plt.cm.Set2(np.linspace(0,1,len(subgroups)))
        cmap_dict = dict(zip(subgroups,cmap))
        fig, ax   = plt.subplots(figsize=(11,max(5,len(race_rows)*0.55+2)))
        y=0; yticks=[]; ylabels=[]
        for sg in subgroups:
            sub = race_rows[race_rows['subgroup']==sg].sort_values('OR')
            ax.axhline(y=y-0.3,color='#CCC',linewidth=0.7)
            y+=0.2
            ax.text(-0.3,y,sg,fontweight='bold',fontsize=10,va='center',color=cmap_dict[sg])
            y+=0.6
            for _,row in sub.iterrows():
                if pd.isna(row['OR']): y+=0.5; continue
                ax.scatter(row['OR'],y,color=cmap_dict[sg],s=60,zorder=3)
                ax.hlines(y,row['CI_low'],row['CI_high'],color=cmap_dict[sg],linewidth=2,zorder=2)
                yticks.append(y); ylabels.append(f"  {row['Label']}"); y+=0.5
        ax.axvline(x=1.0,color='black',linestyle='--',linewidth=1,alpha=0.6)
        ax.set_yticks(yticks); ax.set_yticklabels(ylabels,fontsize=9)
        ax.set_xlabel('Adjusted Odds Ratio (ref: White)')
        ax.set_title('Adjusted OR for Mortality — Stratified by Diagnosis Category', pad=12)
        ax.spines[['top','right']].set_visible(False); ax.set_xlim(left=0)
        ax.legend(handles=[mpatches.Patch(color=cmap_dict[s],label=s) for s in subgroups],
                  loc='lower right',fontsize=9)
        plt.tight_layout(); plt.savefig(FIG+'figure7_subgroup_forest.png'); plt.close(); print("Figure 7 saved.")
except FileNotFoundError:
    print("Figure 7 skipped — run Step 3b first.")

# -----------------------------------------------------------------------------
# FIGURE 8: DISCHARGE DISPOSITION stacked bar
# -----------------------------------------------------------------------------
if 'discharge_disposition' in cohort.columns:
    surv       = cohort[cohort['in_hospital_mortality']==0].copy()
    disp_order = [d for d in ['Home','Home with Services','Rehabilitation Facility',
                               'Skilled Nursing Facility','Long-Term Acute Care','Hospice',
                               'Transfer to Another Facility','Against Medical Advice','Other']
                  if d in surv['discharge_disposition'].unique()]
    disp_pct   = pd.crosstab(surv['race_group'],surv['discharge_disposition'],normalize='index')[disp_order]*100
    d_colors   = ['#2166AC','#6BAED6','#74C476','#FD8D3C','#D94801','#9E9AC8','#FDBB84','#FC4E2A','#BCBDDC'][:len(disp_order)]
    fig, ax    = plt.subplots(figsize=(12,6))
    disp_pct.loc[[r for r in RACE_ORDER if r in disp_pct.index]].plot(
        kind='bar',stacked=True,ax=ax,color=d_colors,edgecolor='white',linewidth=0.5)
    ax.set_ylabel('Percentage of Survivors (%)'); ax.set_title('Discharge Disposition by Race/Ethnicity (Survivors Only)',pad=12)
    plt.xticks(rotation=20,ha='right')
    ax.legend(title='Disposition',bbox_to_anchor=(1.01,1),loc='upper left',fontsize=9)
    ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,105)
    plt.tight_layout(); plt.savefig(FIG+'figure8_disposition_by_race.png'); plt.close(); print("Figure 8 saved.")
else:
    print("Figure 8 skipped — run Step 1b.")

# -----------------------------------------------------------------------------
# FIGURE 9: TEMPORAL TRENDS
# -----------------------------------------------------------------------------
try:
    trend = pd.read_csv(OUT+'temporal_trend_raw.csv')
    pivot = trend.pivot(index='anchor_year_group',columns='race_group',values='mortality_pct')
    focus = [r for r in ['White','Black','Hispanic','Asian'] if r in pivot.columns]
    fig, ax = plt.subplots(figsize=(10,5))
    for race in focus:
        vals = pivot[race].dropna()
        ax.plot(vals.index,vals.values,marker='o',linewidth=2,
                color=PALETTE.get(race,'#878787'),label=race,markersize=6)
        if len(vals)>1:
            z = np.polyfit(np.arange(len(vals)),vals.values,1)
            ax.plot(vals.index,np.poly1d(z)(np.arange(len(vals))),
                    linestyle='--',color=PALETTE.get(race,'#878787'),alpha=0.5,linewidth=1)
    ax.set_xlabel('Year Group'); ax.set_ylabel('In-Hospital Mortality (%)')
    ax.set_title('In-Hospital Mortality Trends by Race/Ethnicity Over Time\n(Solid=observed, Dashed=trend line)', pad=12)
    ax.legend(title='Race/Ethnicity',bbox_to_anchor=(1.01,1),loc='upper left')
    ax.spines[['top','right']].set_visible(False); plt.xticks(rotation=30,ha='right')
    plt.tight_layout(); plt.savefig(FIG+'figure9_temporal_trends.png'); plt.close(); print("Figure 9 saved.")
except FileNotFoundError:
    print("Figure 9 skipped — run Step 3c first.")

print(f"\n=== STEP 4b COMPLETE === Figures saved to: {FIG}")
