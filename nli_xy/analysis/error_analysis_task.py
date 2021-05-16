from nli_xy.analysis.utils import fetch_full_meta, reformat_meta_df
import seaborn as sns
import pandas as pd

def error_analysis(rep_name, encode_config):

    # fetch test meta df also?
    full_meta_df = fetch_full_meta(rep_name, encode_config)
    #full_meta_df = reformat_meta_df(full_meta_df, encode_config)

    #grammar_group = full_meta_df.loc[full_meta_df.grammar_class=='s']
    #heat = get_error_heatmap(grammar_group)

    return full_meta_df 

def get_error_heatmap(meta_df):
    grouped = meta_df.groupby(by=['context'])
    heat = grouped.correct.apply(lambda x: pd.Series(x.values)).unstack()
    heat = heat.dropna(axis=1)

    ax = sns.heatmap(heat)
    return heat
    