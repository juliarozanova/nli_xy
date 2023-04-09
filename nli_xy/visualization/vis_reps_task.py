from prefect import task
import pandas as pd
import plotly.express as px
import pdb

@task
def vis_reps(reps_reduced, meta_df, label_by):
    reps_reduced_df = pd.DataFrame(reps_reduced, columns=['x', 'y', 'z'])
    meta_df = meta_df.reset_index(drop=True)
    vis_df = pd.concat([reps_reduced_df, meta_df], axis=1)
    
    assert label_by in vis_df.columns

    fig = px.scatter_3d(vis_df, 
    x=vis_df.x, 
    y=vis_df.y, 
    z=vis_df.z, 
    color=label_by,
    hover_data=vis_df.columns)
    fig.show()