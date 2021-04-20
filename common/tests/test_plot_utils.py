import numpy as np
import pandas as pd
import pytest

from .datasets import synthetic_simple_total_sales
from models.dataset_utils import Split
from models.plot_utils import violin_plot, prophet_plot


# These tests are using https://github.com/matplotlib/pytest-mpl
# Run
#   pytest common/tests/test_plot_utils.py --mpl-generate-path=common/tests/baseline
# to regenerate baseline images.


@pytest.mark.mpl_image_compare(tolerance=10)
def test_elasticity_plot():
    rnd = np.random.RandomState(42)
    predicted_elasticity = rnd.normal(loc=-2.5, size=512)
    elasticity = rnd.normal(loc=-2.4, scale=4, size=512)
    cat = rnd.choice(np.arange(10).astype(str), size=512)
    return violin_plot(
        pd.DataFrame(
            dict(
                cat=cat,
                elasticity=elasticity,
                predicted_elasticity=predicted_elasticity,
            )
        ),
        target='predicted_elasticity',
        plot_columns='cat',
        group_by='cat',
    )


@pytest.mark.mpl_image_compare(tolerance=10)
def test_elasticity_plot_qcut():
    rnd = np.random.RandomState(42)
    predicted_elasticity = rnd.normal(loc=-2.5, size=512)
    elasticity = rnd.normal(loc=-2.4, scale=4, size=512)
    kpi = pd.qcut(rnd.rand(512), 12, duplicates='drop')
    return violin_plot(
        pd.DataFrame(
            dict(
                kpi=kpi,
                elasticity=elasticity,
                predicted_elasticity=predicted_elasticity,
            )
        ),
        target='predicted_elasticity',
        plot_columns='kpi',
        group_by='kpi',
    )


@pytest.fixture(scope='module')
def prophet_plots():
    from .test_sales_prophet import TestModel

    model, _ = TestModel._test_train(synthetic_simple_total_sales, max_num_batches=1)
    datasets = {
        split: synthetic_simple_total_sales(split=split)[0]
        for split in [Split.train, Split.eval]
    }
    return prophet_plot(model, datasets)


@pytest.mark.mpl_image_compare(tolerance=20)
def test_prophet_plot_DE_online(prophet_plots):
    return prophet_plots['matplotlib']['prophet_DE_online']


@pytest.mark.mpl_image_compare(tolerance=10)
def test_prophet_plot_components_DE_online(prophet_plots):
    return prophet_plots['matplotlib']['prophet_components_DE_online']


@pytest.mark.mpl_image_compare(tolerance=20)
def test_prophet_plot_FR_offline(prophet_plots):
    return prophet_plots['matplotlib']['prophet_FR_offline']
