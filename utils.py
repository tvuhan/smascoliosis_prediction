"""Utility functions for data processing and analysis."""

__author__ = [
    "Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)",
    "Rodrigo Bermudez Schettino (Max Planck Institute for Human Development)",
]
__maintainer__ = ["Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)"]

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# Build function to process one-hot-encoded data (usually recognizable by triple-underscore "___")
def get_one_hot_undo_function(base, shift):
    columns = [base + "___" + str(ele) for ele in shift]

    def undo_encoding(row):
        for c, s in zip(columns, shift):
            if not pd.isnull(row[c]):
                if row[c] == 1:
                    return str(s)

    return undo_encoding


def get_multi_hot_undo_function(base, shift):
    columns = [base + "___" + str(ele) for ele in shift]

    def undo_encoding(row):
        ret = []
        for c, s in zip(columns, shift):
            if not pd.isnull(row[c]):
                if row[c] == 1:
                    ret.append(str(s))
        retS = ", ".join(ret)
        return retS

    return undo_encoding


def get_use_if_either_is_true(cols):
    def use_if_either_is_true(row):
        ret = 0
        for c in cols:
            if not pd.isna(row[c]) and row[c] == 1:
                ret = 1
        return ret

    return use_if_either_is_true


# Build function to process multiple-checkboxes, categorical data
def process_column_coding(df, base, shift):
    undo = get_one_hot_undo_function(base, shift)
    df[base] = df.apply(undo, axis=1)

    columns = [base + "___" + str(ele) for ele in shift]
    df.drop(columns, inplace=True, axis=1)
    return df


def process_multi_column_coding(df, base, shift):
    undo = get_multi_hot_undo_function(base, shift)
    df[base] = df.apply(undo, axis=1)

    columns = [base + "___" + str(ele) for ele in shift]
    df.drop(columns, inplace=True, axis=1)
    return df


def backfilling(
    df, fields, groups, direction="bf", sort=True, refDate="date_assessment"
):  # date_assessment
    fixed_field = [refDate]  # only for the sorting of time needed
    if sort is True:
        temp = df.sort_values(groups + fixed_field)
    else:
        temp = df

    for field in tqdm(fields):
        print("Processing field:", field)
        if direction == "bf":
            df[field] = temp.groupby(groups, sort=False, group_keys=False)[field].apply(
                lambda x: x.fillna(method="bfill").fillna(method="ffill")
            )
        elif direction == "b":
            df[field] = temp.groupby(groups, sort=False, group_keys=False)[field].apply(
                lambda x: x.fillna(method="bfill")
            )
        elif direction == "f":
            df[field] = temp.groupby(groups, sort=False, group_keys=False)[field].apply(
                lambda x: x.fillna(method="ffill")
            )
    return df


def dropCols(df, cols):
    df.drop(cols, inplace=True, axis=1)
    return df


def floatToIntsNA(df, cols):
    for col in cols:
        print("Processing: ", col)
        df = df.astype({col: str})
        df = df.astype({col: float})
        df = df.astype(
            {col: "Int64"}
        )  # that is the internal pandas int type that allows for <NA> / observe the capital I in Int64 vs int64
    return df


def getNAEmptyCols(df):
    nacols = df.columns[df.isna().all()].tolist()
    temp = df.drop(nacols, inplace=False, axis=1)
    nan_value = float("NaN")
    temp.replace("", nan_value, inplace=True)
    emptycols = temp.columns[temp.isna().all()].tolist()
    return nacols, emptycols


def convert999toNA(df):
    nan_value = float("NaN")
    coded_values = ["999", "1000", "888", 999.0, 1000.0, 888.0]
    for coded_value in coded_values:
        df.replace(coded_value, nan_value, inplace=True)
    return df


##### clean
def clean_nan_fields(df, col_name):
    df[col_name].replace(99.9, np.nan, inplace=True)
    df[col_name].replace(999, np.nan, inplace=True)
    df[col_name].replace(888, np.nan, inplace=True)
    return df


##### clean nan fields in therapy_score
def fill_nan_TherapyField(df, col_name):
    df[col_name].replace(np.nan, 0, inplace=True)
    return df


# This function converts all 'date' fields in a dataframe to dateTime
def convert_dateTime(df, is_redcap=True):
    df_converted = df.copy(deep=True)
    for col_name in df.columns:
        if "date" in col_name:
            if is_redcap is True:
                df_converted[col_name] = pd.to_datetime(df[col_name], format="%Y-%m-%d")
            else:
                df_converted[col_name] = pd.to_datetime(df[col_name], format="%d.%m.%y")
    return df_converted


# attach data from smaller dfRight to larger dfLeft via a common field and datafield
# matchRight needs to be a primary key, i.e., unique and integer
# Example: to match two record_id and IDs matchLeft = 'record_id' matchRight = 'ID'


def attachDataInnerJoin(dfLeft, dfRight, matchLeft, matchRight, dataRight):
    interDF = dfRight[[matchRight, dataRight]]
    interDict = {}
    for index, row in interDF.iterrows():
        interDict[int(row[matchRight])] = row[dataRight]

    # use same name for field
    dfLeft[dataRight] = dfLeft[matchLeft].apply(lambda x: interDict[x])

    return dfLeft


def attachDataInnerJoinDateMatch(
    dfLeft, dfRight, matchLeft, matchRight, dataRight, dateLeft, dateRight
):
    interDF = dfRight[[matchRight, dataRight, dateRight]]
    interDict = {}
    interDate = {}
    for index, row in interDF.iterrows():
        interDict[int(row[matchRight])] = row[
            dataRight
        ]  # dictionary with the label by ID
        interDate[int(row[matchRight])] = row[
            dateRight
        ]  # dictionary with the dection date by ID

    def attach_data(matchLeft, dateLeft):
        if not pd.isna(interDate[matchLeft]) and not pd.isna(dateLeft):
            onset = datetime.strptime(interDate[matchLeft], "%d.%m.%y")
            assess = datetime.strptime(dateLeft, "%Y-%m-%d")
            diff = assess - onset
            if diff.days < 0:
                return 0
            else:
                return interDict[matchLeft]

    # use same name for field
    dfLeft[dataRight] = dfLeft[[matchLeft, dateLeft]].apply(
        lambda x: attach_data(x[matchLeft], x[dateLeft]), axis=1
    )

    return dfLeft


#### FUNCTION THAT FINDS THE MISSING NUMBER ####


# This function helps find the missing numbers in a list of ids
def find_missing_number(df, col_name1, col_name2):
    # np.arange from 1 to +1 to get the entire range
    all_numbers = df[col_name1]
    record_ids = df[col_name2]
    all_numbers = set(
        np.arange(1, all_numbers.max() + 1)
    )  # set makes sure that it only looks at unique numbers
    record_ids = set(record_ids)

    missing_numbers = list(sorted(all_numbers - record_ids))

    print("Missing numbers:", missing_numbers)


# find_missing_number(all_numbers, record_ids)


###### GROWTH CURVE PERCENTILES FUNCTION #####


def get_perc_sigmoid_fit(gender, measurement, age, measurement_type):
    """
    @var gender : string 'M' or 'F'
    @var measurement : float feature to find the percentil
    @var age : float in units of Months
    @var measurement_type : string 'headc_cdc' or 'length_cdc' or 'weight_cdc'
    """

    ### Directory for the growth percentile tables data here...
    path = "/Users/tlvh/JupyterNotebooks/_data_child-growth-function/child-growth-charts-cdc/"

    if gender == "M":
        sex_label = "b"
    else:
        sex_label = "g"

    # Table with percentiles
    file_name = sex_label + "_age_" + measurement_type + ".csv"

    # Load file
    path_to_file = path + file_name
    df = pd.read_csv(path_to_file, delimiter=",")

    # Find closest month
    index = np.argmin(np.abs(df.iloc[:, 0] - age))

    # align percentiles y = f(x)
    percentiles_4_age = df.iloc[index].iloc[1:].values
    percentile_points = [3, 5, 10, 25, 50, 75, 90, 95, 97]

    # learn f from the data
    from scipy.optimize import curve_fit

    def f(x, alpha, beta):
        return 100 * np.divide(1, (1 + np.exp(-(alpha * (x - beta)))))

    popt, pcov = curve_fit(
        f, percentiles_4_age, percentile_points, p0=(1.0, np.mean(percentiles_4_age))
    )

    ## Check if fits are working.
    # plt.plot(percentiles_4_age,percentile_points,'-o')
    # plt.plot(percentiles_4_age, f(np.array(percentiles_4_age), *popt))
    # plt.show()

    return f(measurement, *popt)


# Evaluate model
def metrics(y_test, y_pred):
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_curve,
    )

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(
        roc_curve(y_test, y_pred)
    )  # , *, pos_label=None, sample_weight=None, drop_intermediate=True)


def plot_roc_curve(y_tests, y_preds):
    from sklearn.metrics import (
        roc_auc_score,
        roc_curve,
    )

    for y_test, y_pred in zip(y_tests, y_preds):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred)

        lw = 2
        plt.plot(
            fpr,
            tpr,
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    # plt.show()


######################## CLEAN UP GENDER CODES ########################
# Code M = 1, F = 0 for analysis
def convert_gender2numeric(df):
    df["gender"].replace("M", int(1), inplace=True)
    df["gender"].replace("F", int(0), inplace=True)
    return df
