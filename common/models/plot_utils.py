# Copyright © 2019 7Learnings GmbH
"""Plotting utilities"""
import typing

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from fbprophet.plot import (
    plot_plotly,
    plot_components_plotly,
    plot,
    plot_components,
    add_changepoints_to_plot,
)

from . import sales_prophet
from .dataset_utils import Dataset, Split


pd.plotting.register_matplotlib_converters()


def violin_plot(df, target, plot_columns, group_by):
    assert target.startswith('predicted_')
    actual_target = target[len('predicted_') :]
    medians = df.groupby(by=group_by)[[target, actual_target]].agg(['median', 'size'])
    medians.index = medians.index.astype('category')
    # preserve existing category order
    if medians.index.ordered:
        if len(medians) > 10:
            ordered = medians.iloc[:5].append(medians.iloc[-5:])
        else:
            ordered = medians
    else:
        if len(medians) > 10:
            ordered = medians.nlargest(5, columns=(target, 'median')).append(
                medians.nsmallest(5, columns=(target, 'median'))[::-1]
            )
        else:
            ordered = medians.sort_values(by=[(target, 'median')], ascending=False)

    ordered['labels'] = (
        ordered.index.astype('str')
        + '\nn='
        + ordered[target]['size'].astype('str')
        + '\nmed_actual='
        + round(ordered[actual_target]['median'], 4).astype('str')
        + '\nmed_predicted='
        + round(ordered[target]['median'], 4).astype('str')
    )

    style = {
        'lines.linewidth': 2.5,
        'font.size': 12,
        'axes.titlesize': 15,
    }
    df_plot = pd.melt(
        df.rename(columns={actual_target: 'actual', target: 'predicted'}),
        id_vars=plot_columns,
        value_name=actual_target,
        value_vars=['actual', 'predicted'],
    )
    with plt.style.context(['seaborn-whitegrid', style]):
        fig = plt.figure(figsize=(24, 16))
        ax = sns.violinplot(  # variable and value from pd.melt
            x=group_by,
            y=actual_target,
            hue='variable',
            data=df_plot,
            fig=fig,
            order=ordered.index,
            split=True,
            inner="quartile",
        )
        ax.set_title(f"{actual_target} by {group_by}")
        if target == 'predicted_elasticity':
            ax.set_ylim(
                df[target].min() - df[target].std(),
                df[target].max() + df[target].std(),
            )
        if len(medians) > 10:
            ax.axvline(4.5, color='gray', linestyle='--')
        ax.set_xticklabels(ordered.labels)
    return fig


def prophet_plot(
    model: sales_prophet.Model, datasets: typing.Dict[Split, Dataset]
) -> typing.Dict[
    str, typing.Dict[str, typing.Union[go.Figure, matplotlib.figure.Figure]]
]:
    forecasts = model._prophet_predictions(
        pd.concat(model._dataset_to_dataframe(ds) for ds in datasets.values())
    )

    plots = {'plotly': {}, 'matplotlib': {}}  # type: typing.Dict[str, dict]
    for grp, grp_forecast in forecasts.groupby(['market', 'channel']):
        grp_model = model.models[grp]
        grp_suffix = '_'.join(grp).replace('/', '_').replace(' ', '_')
        plots['plotly'][f"prophet_{grp_suffix}"] = plot_plotly(
            grp_model, grp_forecast, changepoints=True, trend=True,
        )
        # re-enable yaxis zoom with xaxis rangeslider
        # (https://github.com/plotly/plotly.py/issues/932#issuecomment-692462447)
        plots['plotly'][f"prophet_{grp_suffix}"].update_yaxes(fixedrange=False)
        plots['plotly'][f"prophet_components_{grp_suffix}"] = plot_components_plotly(
            grp_model, grp_forecast
        )

        matplotlib_plot = plot(grp_model, grp_forecast)
        add_changepoints_to_plot(matplotlib_plot.gca(), grp_model, grp_forecast)
        matplotlib_plot_components = plot_components(grp_model, grp_forecast)

        for fig in [matplotlib_plot, matplotlib_plot_components]:
            for ax in fig.get_axes():
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xlabel('')

        plots['matplotlib'][f"prophet_{grp_suffix}"] = matplotlib_plot
        plots['matplotlib'][
            f"prophet_components_{grp_suffix}"
        ] = matplotlib_plot_components
    return plots
