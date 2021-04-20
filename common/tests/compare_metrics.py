# Copyright © 2019 7Learnings GmbH
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('baseline', None, 'CSV with baseline metrics')
flags.DEFINE_string('compare', None, 'CSV with metrics to compare')
flags.DEFINE_string(
    'output', 'compare_metrics_%s.png', 'Output png file with comparison plots'
)


def main(argv_):
    baseline = pd.read_csv(FLAGS.baseline, index_col=0)
    compare = pd.read_csv(FLAGS.compare, index_col=0)
    df = compare.join(baseline, how='inner', rsuffix='_baseline')

    metrics = baseline.dtypes[baseline.dtypes == np.number].index
    for metric in metrics:
        joint_plot = sns.jointplot(
            x=metric + '_baseline', y=metric, data=df, annot_kws=dict(stat='foobar')
        )
        joint_plot.annotate(stats.pearsonr)
        joint_plot.savefig(FLAGS.output % metric)


if __name__ == '__main__':
    app.run(main)
