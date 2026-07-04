# Statistical analysis of first-person pronoun usage across subcorpora
# ========================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import pingouin as pg

# === Load CSV ===
df = pd.read_csv("first_person_pronoun_frequencies_no_outliers_sd_per_subcorpus.csv")
print("=== Preview Data ===")
print(df.head())

# === ANOVA ===
model = ols('normalized_freq ~ C(subcorpus)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\n=== One-way ANOVA ===")
print(anova_table)


# === Heteroscedasticity-robust ANOVA (HC3 standard errors) ===
# note: this is NOT Welch's ANOVA, it's a standard ANOVA with robust SEs
robust_anova = sm.stats.anova_lm(model, typ=2, robust='hc3')
print("\n=== Robust ANOVA (HC3) ===")
print(robust_anova)

# === Welch's ANOVA (proper, does not assume equal variances) ===
welch_anova = pg.welch_anova(dv='normalized_freq', between='subcorpus', data=df)
print("\n=== Welch's ANOVA ===")
print(welch_anova)

# === Assumption Checks for ANOVA ===


# 1. Normality of residuals (Shapiro-Wilk) nope
residuals = model.resid
shapiro_test = stats.shapiro(residuals)
print("\n=== Shapiro-Wilk Test for Normality of Residuals ===")
print(f"Statistic = {shapiro_test.statistic:.4f}, p = {shapiro_test.pvalue:.4f}")
print("Normal? " + ("YES" if shapiro_test.pvalue > 0.05 else "NO"))

# 2. Homogeneity of variances (Levene) passt
groups = [df[df["subcorpus"] == g]["normalized_freq"] for g in df["subcorpus"].unique()]
levene_test = stats.levene(*groups)
print("\n=== Levene's Test for Homogeneity of Variance ===")
print(f"Statistic = {levene_test.statistic:.4f}, p = {levene_test.pvalue:.4f}")
print("Equal variances? " + ("YES" if levene_test.pvalue > 0.05 else "NO"))

# (Shapiro per group)

# extract group data
pt = df[df["subcorpus"] == "PT"]["normalized_freq"]
ab = df[df["subcorpus"] == "Allebrieven"]["normalized_freq"]
lb = df[df["subcorpus"] == "Letterbook"]["normalized_freq"]
dutch = df[df["subcorpus"] == "Dutch"]["normalized_freq"]

# Shapiro-Wilk test per group
shapiro_test_pt = stats.shapiro(pt)
shapiro_test_ab = stats.shapiro(ab)
shapiro_test_lb = stats.shapiro(lb)
shapiro_test_dutch = stats.shapiro(dutch)

print("\n=== Shapiro-Wilk Test: Individual Subcorpora ===")
print(f"PT:           stat = {shapiro_test_pt.statistic:.4f}, p = {shapiro_test_pt.pvalue:.4f}")
print(f"Allebrieven:  stat = {shapiro_test_ab.statistic:.4f}, p = {shapiro_test_ab.pvalue:.4f}")
print(f"Letterbook:   stat = {shapiro_test_lb.statistic:.4f}, p = {shapiro_test_lb.pvalue:.4f}")
print(f"Dutch:        stat = {shapiro_test_dutch.statistic:.4f}, p = {shapiro_test_dutch.pvalue:.4f}")


# === Boxplot Visualization ===
df['subcorpus'] = df['subcorpus'].replace({'PT': 'Philosophical Transactions'})
order = ['Dutch', 'Allebrieven', 'Philosophical Transactions', 'Letterbook']
palette = {
    'Dutch': '#FFB580',
    'Philosophical Transactions': '#9AA2FF',
    'Allebrieven': '#CBAACB',
    'Letterbook': '#9AD5FF',
}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='subcorpus', y='normalized_freq', order=order, palette=palette, hue='subcorpus', legend=False)
plt.title("Normalized Frequency of First-Person Pronouns by Subcorpus")
plt.ylabel("Normalized Frequency")
plt.xlabel("Subcorpus")
plt.tight_layout()
plt.savefig("boxplot_normalized_frequencies_pretty.png", dpi=300)
plt.show()


# === HISTOGRAMS (raw normalized frequency) ===
subcorpora = ['Philosophical Transactions', 'Allebrieven', 'Letterbook', 'Dutch']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, group in enumerate(subcorpora):
    data = df[df['subcorpus'] == group]['normalized_freq']
    sns.histplot(data, kde=True, ax=axs[i], bins=20)
    axs[i].set_title(f"{group} - Normalized Frequency")
    axs[i].set_xlabel("Normalized Frequency")
    axs[i].set_ylabel("Density")

plt.tight_layout()
plt.savefig("histograms_by_subcorpus.png", dpi=300)
plt.show()






summary_stats = df.groupby('subcorpus')['normalized_freq'].agg(['mean', 'std'])
print(summary_stats)
