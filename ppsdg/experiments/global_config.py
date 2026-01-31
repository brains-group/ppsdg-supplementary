"""Shared configuration for experiments."""

datasets = {
    "AD": "config/dataset/adult.yaml",
    "BM": "config/dataset/tabarena-bank-marketing.yaml",
    "BC": "config/dataset/tabarena-Bank_Customer_Churn.yaml",
    "CC": "config/dataset/tabarena-credit_card_clients_default.yaml",
    "CR": "config/dataset/tabarena-credit-g.yaml",
    "GM": "config/dataset/tabarena-GiveMeSomeCredit.yaml",
    "HE": "config/dataset/tabarena-heloc.yaml",
    "PB": "config/dataset/tabarena-polish_companies_bankruptcy.yaml",
    "PW": "config/dataset/phishing.yaml",
    "TB": "config/dataset/tabarena-taiwanese_bankruptcy_prediction.yaml",
    "GC": "config/dataset/tabarena-Is-this-a-good-customer.yaml",
    "AP": "config/dataset/tabarena-kddcup09_appetency.yaml",
}

pretty_dset_names = {
    "AD": "Adult",
    "BM": "Bank Marketing",
    "BC": "Bank Customer Churn",
    "CC": "Credit Card Default",
    "CR": "Credit Approval",
    "GM": "Give Me Some Credit",
    "HE": "HELOC",
    "PB": "Polish Companies Bankruptcy",
    "PW": "Phishing Websites",
    "TB": "Taiwanese Bankruptcy Prediction",
    "GC": "Is This a Good Customer",
    "AP": "KDDCup09 Appetency",
}

mle_target = "xgboost"
mle_metric = "balanced_accuracy"

privacy_metrics = [
    "dcr_baseline",
    "dcr_overfit",
]
