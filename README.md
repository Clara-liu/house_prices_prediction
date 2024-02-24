# UK house price prediction using transformer with open source data

This is a small project inspired by [Wu et. al (2020)](https://arxiv.org/abs/2001.08317), which uses
a encoder-decoder transformer to predict house prices in the UK. The data can be obtained from the
[landregrstry's website](https://landregistry.data.gov.uk/).

To get started, install dependencies by `pip install -r requirements.txt`.

To obtain the house price data:

1. Set the desired date range in `data/get_data.py`. Note that the `total_seq_len` arg passed to the
function `split_data` in `data/process_data.py` must match the number of months within the date
range
2. Run `python data/get_data.py`


More to come on modelling and predictions.