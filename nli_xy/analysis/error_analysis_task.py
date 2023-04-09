from nli_xy.analysis.utils import fetch_full_meta, reformat_meta_df
import seaborn as sns
import pandas as pd

def get_error_heatmap(meta_df):
    grouped = meta_df.groupby(by=['context'])
    heat = grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()
    heat = heat.dropna(axis=1)

    ax = sns.heatmap(heat)
    return heat
    