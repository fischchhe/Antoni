import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, shapiro, levene
import pingouin as pg
import statsmodels.api as sm
from statsmodels.formula.api import ols

# === Load CSV ===
df = pd.read_csv("first_person_pronoun_frequencies_no_outliers_sd_per_subcorpus.csv")
print("=== Preview Data ===")
print(df.head())

# === Add log-transformed column (with small constant to avoid log(0)) ===
# caveat: for letters with 0 pronouns, log(1e-6) creates an extreme negative
# value (~ -13.8) that can distort the ANOVA. kept as-is to reproduce the
# thesis results; check below how many letters are affected.
zero_letters = (df["normalized_freq"] == 0).sum()
print(f"\nLetters with zero first-person pronouns: {zero_letters}")
df["log_freq"] = np.log(df["normalized_freq"] + 1e-6)

# === ANOVA (log-transformed) ===
model_log = ols('log_freq ~ C(subcorpus)', data=df).fit()
anova_log = sm.stats.anova_lm(model_log, typ=2)

print("\n=== One-way ANOVA (Log-Transformed) ===")
print(anova_log)

# === Heteroscedasticity-robust ANOVA (HC3, not Welch's) ===
robust_log = sm.stats.anova_lm(model_log, typ=2, robust='hc3')
print("\n=== Robust ANOVA (HC3, Log-Transformed) ===")
print(robust_log)

# === Welch's ANOVA (proper, does not assume equal variances) ===
welch_log = pg.welch_anova(dv='log_freq', between='subcorpus', data=df)
print("\n=== Welch's ANOVA (Log-Transformed) ===")
print(welch_log)

resid_log = model_log.resid
shapiro_log = shapiro(resid_log)
print("\n=== Shapiro-Wilk (Residuals, Log-Transformed) ===")
print(f"Statistic = {shapiro_log.statistic:.4f}, p = {shapiro_log.pvalue:.4f}")

# === Shapiro per group (log) ===
print("\n=== Shapiro-Wilk Test: Individual Subcorpora (Log Freq) ===")
for name in ["PT", "Allebrieven", "Letterbook", "Dutch"]:
    group = df[df["subcorpus"] == name]["log_freq"]
    result = shapiro(group)
    print(f"{name:<13} stat = {result.statistic:.4f}, p = {result.pvalue:.4f}")

# === Plot log PT ===
sns.histplot(df[df["subcorpus"] == "PT"]["log_freq"], kde=True, bins=20)
plt.title("PT Subcorpus - Log-Transformed Frequency Distribution")
plt.xlabel("Log(Normalized Frequency)")
plt.tight_layout()
plt.show()

# === Planned T-Tests (log-transformed) ===
pt_log = df[df["subcorpus"] == "PT"]["log_freq"]
ab_log = df[df["subcorpus"] == "Allebrieven"]["log_freq"]
lb_log = df[df["subcorpus"] == "Letterbook"]["log_freq"]

t_ab_log, p_ab_log = ttest_ind(pt_log, ab_log, equal_var=False)
t_lb_log, p_lb_log = ttest_ind(pt_log, lb_log, equal_var=False)

print("\n=== Planned Pairwise T-Tests (Log-Transformed, α = 0.025) ===")
print(f"PT vs Allebrieven: t = {t_ab_log:.3f}, p = {p_ab_log:.4f}, Significant? {'YES' if p_ab_log < 0.025 else 'NO'}")
print(f"PT vs Letterbook:  t = {t_lb_log:.3f}, p = {p_lb_log:.4f}, Significant? {'YES' if p_lb_log < 0.025 else 'NO'}")

# === APA Table (log-transformed) ===
def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(), g2.std()
    s_pooled = (((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))**0.5
    return (g1.mean() - g2.mean()) / s_pooled

d_ab_log = cohens_d(pt_log, ab_log)
d_lb_log = cohens_d(pt_log, lb_log)

label_pt = f"Philosophical Transactions ({pt_log.mean():.2f} ± {pt_log.std():.2f})"
label_ab = f"Allebrieven ({ab_log.mean():.2f} ± {ab_log.std():.2f})"
label_lb = f"Letterbook ({lb_log.mean():.2f} ± {lb_log.std():.2f})"

log_df = pd.DataFrame({
    "Group 1": [label_pt, label_pt],
    "Group 2": [label_ab, label_lb],
    "t": [round(t_ab_log, 3), round(t_lb_log, 3)],
    "p": [round(p_ab_log, 4), round(p_lb_log, 4)],
    "Cohen's d": [round(d_ab_log, 3), round(d_lb_log, 3)],
    "Sig.": [
        "***" if p_ab_log < 0.001 else "**" if p_ab_log < 0.01 else "*" if p_ab_log < 0.05 else "ns",
        "***" if p_lb_log < 0.001 else "**" if p_lb_log < 0.01 else "*" if p_lb_log < 0.05 else "ns"
    ]
})

print("\n=== APA-style Table: Planned Comparisons (Log) ===")
print(log_df)
log_df.to_csv("planned_comparisons_log_transformed.csv", index=False)

# === Boxplot (original, for thesis) ===
df['subcorpus'] = df['subcorpus'].replace({'PT': 'Philosophical Transactions'})
order = ['Dutch', 'Allebrieven', 'Philosophical Transactions', 'Letterbook']
palette = {
    'Dutch': '#FFB580',
    'Philosophical Transactions': '#9AA2FF',
    'Allebrieven': '#CBAACB',
    'Letterbook': '#9AD5FF',
}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='subcorpus', y='log_freq', order=order, palette=palette, hue='subcorpus', legend=False)
plt.title("Log-Transformed Frequency of First-Person Pronouns by sub corpus")
plt.ylabel("log(Normalized Frequency)")
plt.xlabel("Subcorpus")
plt.tight_layout()
plt.savefig("boxplot_normalized_frequencies_pretty_log.png", dpi=300)
plt.show()


# === HISTOGRAMS (log-transformed) ===
subcorpora = ['Philosophical Transactions', 'Allebrieven', 'Letterbook', 'Dutch']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, group in enumerate(subcorpora):
    data = df[df['subcorpus'] == group]['log_freq']
    sns.histplot(data, kde=True, ax=axs[i], bins=20)
    axs[i].set_title(f"{group} - Log Frequency")
    axs[i].set_xlabel("Log(Normalized Frequency)")
    axs[i].set_ylabel("Density")
