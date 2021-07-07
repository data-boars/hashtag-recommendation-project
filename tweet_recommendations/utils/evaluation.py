import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from typing import Callable


def lift(df: pd.DataFrame, split_col: str, metric_name: str, nb_groups: int = 10,
         title: str = "", backend: str = "plotly", cumulative: bool = False, pyplot_kwargs: dict = {}):

    df = df[[split_col, metric_name]].copy()
    df[split_col] = pd.qcut(df[split_col], nb_groups)

    population_mean_metric = df[metric_name].mean()

    avg_metric_name = f"Average {metric_name}"

    gb = (df.groupby(split_col)
          .agg({metric_name: "mean"})
          .reset_index()
          .rename(columns={metric_name: avg_metric_name})
          .sort_values(split_col, ascending=False))

    if cumulative:
        def cum_avg_metric(current_group):
            return df[df[split_col] >= current_group][metric_name].mean()
        gb[avg_metric_name] = gb[split_col].apply(cum_avg_metric).astype(float)

    if backend == "plotly":
        gb[split_col] = gb[split_col].astype(str)
        bar = px.bar(gb, x=split_col, y=avg_metric_name).data[0]
        line = go.Scatter(x=[bar.x[0], bar.x[-1]],
                          y=[population_mean_metric]*2, name=f"Whole population average {metric_name}")
        return go.Figure(data=[bar, line],
                         layout={"title":f"{'Cumulative - ' if cumulative else ''} {title}",
                                 "xaxis_title": split_col,
                                 "yaxis_title": avg_metric_name})
    else:
        fig, ax = plt.subplots(**pyplot_kwargs)
        bar = gb.plot.bar(split_col, avg_metric_name, title=f"{'Cumulative - ' if cumulative else ''} {title}", ax=ax)
        line = plt.plot([0, len(gb)-1], [population_mean_metric] * 2,
                        linestyle=":", c='r', linewidth=2,
                        label=f"Whole population average {metric_name}")
        return ax
