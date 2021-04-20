import functools
import operator
import textwrap
import typing

import numpy as np
import pandas as pd
import plotly.subplots
import streamlit as st

from common.common.dataset import render_sql_from_file
# from common.dataset import render_sql_from_file
from utils import (
    select_gcp_project,
    bq_list_tables,
    bq_list_partitions,
)
from common.common.dataset.bigquery import bq_query

bq_query = st.cache(bq_query, persist=True, suppress_st_warning=True)
bq_list_partitions = st.cache(
    bq_list_partitions, persist=True, suppress_st_warning=True
)

st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
)

project_id = select_gcp_project()

id_cols = ['market', 'channel', 'product_id']
kpis = ['profit', 'sales_before_returns', 'revenue']
kpi = st.selectbox(
    'Select KPI to compare and filter outliers', options=kpis, index=0, key='kpis'
)
nr_price_changes = st.number_input(
    'Filter products with more price changes than:', min_value=0, max_value=10, value=1
)

available_forecasts = bq_list_partitions(
    project_id, 'production.forecast'
) + bq_list_tables(project_id, 'dev', prefix='forecast_')


forecast_table_1 = st.selectbox(
    'forecast1', available_forecasts, key='forecast_table_1'
)
forecast_table_2 = st.selectbox(
    'forecast2', available_forecasts, key='forecast_table_2'
)

df1 = bq_query(
    project_id,
    render_sql_from_file(
        './forecast_price_aggregation.sql', forecast_table=forecast_table_1
    ),
)
df2 = bq_query(
    project_id,
    render_sql_from_file(
        './forecast_price_aggregation.sql', forecast_table=forecast_table_2
    ),
)


is_dev_forecast = forecast_table_2.startswith('dev.forecast_')
use_dev_data = st.checkbox(
    'Use refreshed (dev) data from forecast run.', value=False, key='use_dev_data'
)

assert (
    not use_dev_data or is_dev_forecast
), f"Cannot infer run identifier from {forecast_table_2}"
run_identifier = forecast_table_2[len('dev.forecast_') :]

data_price_periods = (
    f"dev.data_price_periods_{run_identifier}" if use_dev_data else 'data.price_periods'
)
price_periods_extended = bq_query(
    project_id,
    render_sql_from_file(
        './price_periods_extended.sql',
        table=data_price_periods,
        nr_price_changes=nr_price_changes,
    ),
)

data_product_attributes = (
    f"dev.data_product_attributes_{run_identifier}"
    if use_dev_data
    else 'data.product_attributes'
)

product_attributes = bq_query(
    project_id,
    textwrap.dedent(
        f"""
        SELECT
          *
        FROM
          `{data_product_attributes}`
        """
    ),
).set_index('product_id')


def is_outlier(values: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    return (values - mean).abs() / std > 2


# filter kpi outliers from price points extended
ppe_stats = price_periods_extended.groupby(id_cols).agg({kpi: ['mean', 'std']})
ppe = price_periods_extended.set_index(id_cols + ['nth_most_recent_price_change'])
outliers = functools.reduce(
    np.bitwise_or,
    map(
        lambda kpi: is_outlier(ppe[kpi], ppe_stats[kpi]['mean'], ppe_stats[kpi]['std']),
        [kpi],
    ),
)
ppe = ppe.loc[outliers[~outliers].index].reset_index()


def _select_group(col: str, dfs: typing.List[pd.DataFrame]):
    options = list(
        functools.reduce(operator.or_, map(lambda df: set(df[col].unique()), dfs))
    )
    option = st.selectbox(f"Select {col}", options, key=f"Select {col}")
    return [df[df[col] == option].drop(columns=col) for df in dfs]


df1, df2, ppe = _select_group('market', [df1, df2, ppe])
df1, df2, ppe = _select_group('channel', [df1, df2, ppe])


@st.cache
def compute_idxmax(df_grp):
    return df_grp.idxmax()


def compute_KPImax(df, kpi):
    idxmax = compute_idxmax(df.groupby('product_id')[[kpi]])
    return df.loc[idxmax[kpi].dropna()]


maxima_fc1 = compute_KPImax(df1, kpi)
maxima_fc2 = compute_KPImax(df2, kpi)
maxima_ppe = compute_KPImax(ppe, kpi)


def wmse(a, b):
    return (a - b) ** 2


def wmae(a, b):
    return (a - b).abs()


def wsmape(a, b):
    return (a - b).abs() * 2 / (a + b)


def wmape(a, b):
    return (a / b - 1).abs()


# metrics = [wmse, wmae, wsmape]
metrics = [wmse, wmape]
suffixes = [('fc1', 'fc2'), ('fc1', 'ppe'), ('fc2', 'ppe')]

st.markdown(
    f"# Differences of price-{kpi} curve maxima (weighted by predicted revenue)"
)

if st.button('help', key='help'):
    st.write(
        "To check what to expect from your new forecast use the tool to compare it to other forecasts (fc1, fc2) and the historical price performance (ppe)."
    )
    st.write(
        f"For each of the fc1, fc2 and ppe we can find the price with the highest {kpi} (outliers are removed in the process, refer to code to learn more). We can then compare those prices between each other. By default you can choose between {[m.__name__ for m in metrics]} metrics (refer to code for other metrics). The mertics are weighted by revenue."
    )
    st.write(
        f"Check per product graphs below to better understand the price-{kpi} cruves."
    )
    st.write('This is an internal 7L tool, feel free to add changes to your needs.')

diff = (
    pd.merge(
        maxima_fc1, maxima_fc2, how='outer', on='product_id', suffixes=('_fc1', ''),
    )
    .merge(maxima_ppe, how='inner', on='product_id', suffixes=('_fc2', '_ppe'),)
    .set_index('product_id')
)
diff = diff[sorted(diff.columns)]
diff['sum_fc1_fc2_revenue'] = diff.revenue_fc1.fillna(0) + diff.revenue_fc2.fillna(0)
diff['sum_fc1_fc2_profit'] = diff.profit_fc1.fillna(0) + diff.profit_fc2.fillna(0)
for metric in metrics:
    for lsuffix, rsuffix in suffixes:
        diff[f"{metric.__name__}_{lsuffix}_{rsuffix}"] = metric(
            diff[f"gross_red_price_{lsuffix}"], diff[f"gross_red_price_{rsuffix}"]
        ).mul(diff.sum_fc1_fc2_revenue, axis=0)
metric_names = [
    f"{metric.__name__}_{lsuffix}_{rsuffix}"
    for metric in metrics
    for lsuffix, rsuffix in suffixes
]
st.markdown('### Average metrics')
st.markdown(f"fc1 = max {kpi} price from 1st selected forecast")
st.markdown(f"fc2 = max {kpi} price from 2st selected forecast")
st.markdown(f"ppe = max {kpi} price from history (price periods extended)")
st.write(
    pd.DataFrame(
        {
            'metric': metric_names,
            'price_diff': [
                diff[m].sum() / diff.sum_fc1_fc2_revenue.sum() for m in metric_names
            ],
        }
    )
)

st.markdown('### Breakdown metrics with average prices')
breakdown_column = st.selectbox(
    'breakdown_column', product_attributes.columns, index=3, key='breakdown_column'
)
selected_metric = st.multiselect(
    'metric',
    options=metric_names,
    default=['wmape_fc1_ppe', 'wmape_fc2_ppe'],
    key='metric_options',
)

diff_w_attributes = diff.join(product_attributes, how='left')
for price in ['gross_red_price_fc1', 'gross_red_price_fc2', 'gross_red_price_ppe']:
    diff_w_attributes[f"avg_{price}"] = diff_w_attributes[price].mul(
        diff_w_attributes.sum_fc1_fc2_revenue
    )
diff_grouped = diff_w_attributes.groupby(breakdown_column).aggregate('sum')
avg_prices = [
    'avg_gross_red_price_fc1',
    'avg_gross_red_price_fc2',
    'avg_gross_red_price_ppe',
]
for m in metric_names + avg_prices:
    diff_grouped[m] = diff_grouped[m] / diff_grouped.sum_fc1_fc2_revenue
st.write(diff_grouped[selected_metric + avg_prices])

st.markdown('### Top 50 price difference products')
sort_by = st.multiselect(
    'Sort by:',
    options=metric_names + ['sum_fc1_fc2_revenue', 'sum_fc1_fc2_profit'],
    default=None,
    key='sort_by',
)
for sort in sort_by:
    n = 50
    st.markdown(f"Top {n} price differences by {sort}")
    st.write(diff.nlargest(n, columns=sort).reset_index())


product_id = st.text_input('Enter product_id to plot', key='plot_product_id')
if product_id:
    product_id = np.array(product_id, df1.product_id.dtype)
    dfp_1 = df1[df1.product_id == product_id][['gross_red_price'] + kpis].sort_values(
        'gross_red_price'
    )
    dfp_2 = df2[df2.product_id == product_id][['gross_red_price'] + kpis].sort_values(
        'gross_red_price'
    )
    dfp_ppe = ppe[ppe.product_id == product_id].sort_values('gross_red_price')
    fig = plotly.subplots.make_subplots(
        specs=[[dict(secondary_y=True)]] * len(kpis), rows=len(kpis), shared_xaxes=True
    )
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS
    for i, kpi in enumerate(kpis):
        if len(dfp_1):
            fig.add_scatter(
                x=dfp_1.gross_red_price,
                y=dfp_1[kpi],
                name=forecast_table_1,
                legendgroup=forecast_table_1,
                showlegend=i == 0,
                line=dict(color=colors[0]),
                row=i + 1,
                col=1,
            )
            idx = dfp_1[kpi].idxmax()
            fig.add_annotation(
                x=dfp_1.loc[idx].gross_red_price,
                y=dfp_1.loc[idx][kpi],
                arrowcolor=colors[0],
                text='max',
                row=i + 1,
                col=1,
            )
        if len(dfp_2):
            fig.add_scatter(
                x=dfp_2.gross_red_price,
                y=dfp_2[kpi],
                name=forecast_table_2,
                legendgroup=forecast_table_2,
                showlegend=i == 0,
                line=dict(color=colors[1]),
                row=i + 1,
                col=1,
            )
            idx = dfp_2[kpi].idxmax()
            fig.add_annotation(
                x=dfp_2.loc[idx].gross_red_price,
                y=dfp_2.loc[idx][kpi],
                text='max',
                arrowcolor=colors[1],
                row=i + 1,
                col=1,
            )

        fig.add_scatter(
            x=dfp_ppe.gross_red_price,
            y=dfp_ppe[kpi],
            name='price_periods_extended',
            legendgroup='price_periods_extended',
            showlegend=i == 0,
            line=dict(color=colors[2]),
            mode='markers',
            marker=dict(size=2 * np.sqrt(dfp_ppe.num_days)),
            row=i + 1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(showgrid=False, secondary_y=True)
        fig.update_yaxes(title_text=kpi, row=i + 1)
        fig.update_yaxes(title_text=f"adj_{kpi}", row=i + 1, secondary_y=True)
    fig.update_xaxes(title_text='gross_red_price')
    fig.update_layout(height=len(kpis) * 500 + 150)
    st.plotly_chart(fig, use_container_width=True)
