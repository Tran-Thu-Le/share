# Cryptocurrency price data from Kaggle

The data was download at [https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv][1]
containing the price of Bitcoin since 2013-04-29 and many other coins.

To learn how to process the data please take a look at this [notebook][3] or its [original post][2].

To load the github csv, we first open it as a raw file, copy the link and then load it using pandas.
For example, to load the [raw][4] file [coin_Bitcoin.csv][5] we using the following code.
```
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Tran-Thu-Le/share/main/Time_Series/coins/coin_Bitcoin.csv')
data
```

**Remarks.**
1. To run a github notebook replace `github` by `githubtocolab` in the corresponding link.
2. To download the notebook from that webpage, I did log in to Deepnote (suggested at the top of the page) with my gmail (longtran) and then open and download it.


[1]: https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv
[2]: https://deepnote.com/@Frequentist-Statistics/Predicting-the-closing-price-of-Bitcoin-using-Multiple-Regression--L4yeXj9R-mQOzrx3M0pXA
[3]: https://github.com/Tran-Thu-Le/share/blob/main/Time_Series/bitcoin_multi_regression.ipynb
[4]: https://raw.githubusercontent.com/Tran-Thu-Le/share/main/Time_Series/coins/coin_Bitcoin.csv
[5]: https://github.com/Tran-Thu-Le/share/blob/main/Time_Series/coins/coin_Bitcoin.csv
