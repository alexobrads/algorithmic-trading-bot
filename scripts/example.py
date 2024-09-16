import models as alg


def main():
    symbol = "ETHAUD"
    bar = "1h"
    look_back = 10

    eth = alg.data.HistoricalBars(symbol, bar, look_back)


if __name__ == '__main__':
    main()
