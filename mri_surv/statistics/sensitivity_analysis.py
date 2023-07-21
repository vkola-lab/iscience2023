import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tabulate

def load_gmv_corr_with_ad() -> pd.DataFrame:
    return pd.read_csv("./metadata/data_processed/gmv_corr_with_ad_stats.csv")

def load_risk_table(seed, risk) -> pd.DataFrame:
    df_ = pd.read_csv(f"./metadata/data_processed/censoring_probabilities_seed{seed}_riskpr{risk/100}.csv")
    df_ = df_.rename(columns={"Time": "Time[months]"})
    return df_

def summary_stat_mn_sd():
    df = load_gmv_corr_with_ad()
    df = df.groupby("Risk").agg(["mean", "std"])[["Brier", "CI"]]
    df_formatted = pd.DataFrame(index=df.index, columns=["Brier Score", "CI"])
    pm = u"\u00b1"
    df_formatted["Brier Score"] = df["Brier"].apply(
        lambda x: f"{x['mean']:.4f} {pm} {x['std']:.4f} ", axis=1
        )
    df_formatted["CI"] = df["CI"].apply(
        lambda x: f"{x['mean']:.4f} {pm} {x['std']:.4f} ", axis=1
        )
    df_formatted.index = df_formatted.index / 100
    tab = tabulate.tabulate(df_formatted, headers = df_formatted.columns, tablefmt="github", showindex=True, )
    with open("./results/sensitivity_testing.txt", "w") as fi:
        fi.write(tab)

def load_risk_table_all() -> pd.DataFrame:
    df_all = []
    for risk in np.arange(0, 20, 2):
        for seed in range(10):
            df = load_risk_table(seed, risk)
            df["Seed"] = seed
            df.loc[df["Dataset"] == "ADNICensored", "Dataset"] = f"ADNI Censored, Pr[{risk/100:.2f}]"
            df_all.append(df)
    df_all_pd = pd.concat(df_all, axis=0)
    return df_all_pd

def plot_stats_for_all_corr_vals():
    df = load_risk_table_all()
    df = df.loc[~np.isposinf(df["Time[months]"]), :]
    df['risk_censor'] = df['risk_censor']
    df["Dataset"] = df["Dataset"].astype("category")
    ax = sns.lineplot(data=df, x="Time[months]", y="risk_censor", hue ="Dataset", ci = "sd", legend = "auto")
    ax.set(xlabel="Time [months]", ylabel="Pr[Not Censored]")
    ax.legend(title='Dataset', loc='upper right')
    plt.savefig("figures/censor_vs_time.png")


if __name__ == "__main__":
    # plot_stats_for_all_corr_vals()
    summary_stat_mn_sd()