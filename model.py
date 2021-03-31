from sklearn import linear_model
from sklearn.metrics import r2_score
import pandas as pd

def model_data(bonds_hist, ivv_hist, n, alpha):

###### For teaching & debugging ################################################
# ------------------------------------------------------------------------------
#     import pandas as pd
#     import pickle
    #
    # pickle.dump(bonds_hist, open("bonds_hist.p", "wb"))
    # pickle.dump(ivv_hist, open("ivv_hist.p", "wb"))
    # print(n)
    # print(alpha)
    #
    # bonds_hist = pd.read_json(pickle.load(open("bonds_hist.p", "rb")))
    # ivv_hist = pd.read_json(pickle.load(open("ivv_hist.p", "rb")))
    #
    # n = 5
    # alpha = 0.02
# ------------------------------------------------------------------------------
################################################################################

    bonds_hist = pd.read_json(bonds_hist)
    ivv_hist = pd.read_json(ivv_hist)

    def bonds_fun(yields_row):
        maturities = pd.DataFrame([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2])
        linreg_model = linear_model.LinearRegression()
        linreg_model.fit(maturities, yields_row[1:])
        modeled_bond_rates = linreg_model.predict(maturities)
        print(yields_row)
        print(yields_row["Date"].date())
        print(linreg_model.coef_[0])
        print(linreg_model.intercept_)
        print(r2_score(yields_row[1:], modeled_bond_rates))
        return [yields_row["Date"].date(), linreg_model.coef_[0],
                linreg_model.intercept_,
                r2_score(yields_row[1:], modeled_bond_rates)]

    features = bonds_hist[
        ["Date", "1 mo", "2 mo", "3 mo", "6 mo", "1 yr", "2 yr"]
    ].apply(bonds_fun, axis=1,result_type='expand')

    features.columns = ["Date", "a", "b", "R2"]

    ivv_response = pd.DataFrame(features["Date"][0:len(features) - n])
    ivv_response["response"] = ivv_response.apply(lambda _: '', axis=1)

    for i in range(0, len(ivv_hist) - n):
        # Start the for loop at row n.
        # If the highest price in the range i + 1 to i + n is greater than
        # the closing price on day i, then the strategy made money and is
        # assigned '1'; '0' otherwise.
        ivv_response["response"][i] = int(
            max(ivv_hist["High"][list(range(i+1, i+n, 1))]) > ivv_hist[
                "Close"][i] * (1+alpha)
        )

    return pd.DataFrame.to_json(pd.merge(features, ivv_response, on="Date"))
