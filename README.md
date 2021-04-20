This folder contains analysis tools built with [Streamlit](https://streamlit.io/).

## Usage

```bash
streamlit run streamlit/compare_forecasts.py
…
```

### Stream Cache usage

Some of the streamlit apps use persisted caching for data, in order to clear those,
either run "clean cache" in streamlit or delete the cache in your local file system with `rm -r ~/.streamlit/cache/`.
Read more about streamlit and caching [here](https://streamlit.io/docs/api.html#optimize-performance).

## Difference from [notebooks](../notebooks)

Other than notebooks, those streamlit tools are subject to reviews and
maintained with the rest of our code base, allow collaborations, and could be
developed stable tools for regular usage/automation.
