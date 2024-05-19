import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ppscore as pps
import seaborn as sns
from matplotlib import cm
from matplotlib import colors as mcl
from pretty_confusion_matrix import pp_matrix
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

import ml_models

sns.set(font_scale=0.8)
DPI = 300

cm.colors.cnames.keys()

colors = [[0.0, "midnightblue"], [1.0, "yellow"]]
orange_blue = mcl.LinearSegmentedColormap.from_list("", colors)

cmap_data = plt.cm.Paired
cmap_cv = plt.cm.binary


def plot_symptoms_frequency(df):
    # DISCLAIMER: 'df' refers to the data you passed in when calling 'dtale.show'

    import pandas as pd
    import plotly.graph_objs as go

    if isinstance(df, (pd.DatetimeIndex, pd.MultiIndex)):
        df = df.to_frame(index=False)

    # remove any pre-existing indices for ease of use in the D-Tale code, but this is not required
    df = df.reset_index().drop("index", axis=1, errors="ignore")
    df.columns = [
        str(c) for c in df.columns
    ]  # update columns to strings in case they are numbers

    # df = df.sort_values(['first_symptoms_arr'], ascending=[True])

    s = df[~pd.isnull(df["first_symptoms"])]["first_symptoms"]
    chart = pd.value_counts(s).to_frame(name="data")
    chart.index.name = "labels"
    chart = chart.reset_index().sort_values(["data", "labels"], ascending=[False, True])
    chart = chart[:100]
    charts = [
        go.Bar(x=chart["labels"].values, y=chart["data"].values, name="Frequency")
    ]
    figure = go.Figure(
        data=charts,
        layout=go.Layout(
            {
                "barmode": "group",
                "legend": {"orientation": "h"},
                "title": {"text": "first_symptoms Value Counts"},
                "xaxis": {"title": {"text": "first_symptoms"}},
                "yaxis": {"title": {"text": "Frequency"}},
            }
        ),
    )

    figure.show()

    # If you're having trouble viewing your chart in your notebook try passing your 'chart' into this snippet:
    #
    # from plotly.offline import iplot, init_notebook_mode
    #
    # init_notebook_mode(connected=True)
    # for chart in charts:
    #     chart.pop('id', None) # for some reason iplot does not like 'id'
    # iplot(figure)


def plotScores(df, timeF, dataF, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    temp = df[["record_id", dataF, timeF]]

    for label, dff in temp.groupby("record_id"):
        dff = dff.dropna()
        plt.plot(dff[timeF], dff[dataF])

    plt.title(title)
    plt.grid()


# function that plots a feature (typically a score) against a measure of time (typically age_motor or age_onset)
def generateFeaturePlot(df, timeF, dataF, color="sma_type", fn="figures/featurePlot_"):
    plots = []

    title = "Data Analysis Plots: " + dataF + " vs. " + timeF

    temp = df[["record_id", dataF, timeF, color]]
    temp = temp.sort_values(["record_id", timeF], ascending=True)

    for label, dff in temp.groupby("record_id"):
        dff = dff.dropna()
        value = dff[color].values[:1].tolist()
        if value != []:
            match value[0]:
                case 1.0:
                    col = "darkblue"
                case 2.0:
                    col = "steelblue"
                case 3.0:
                    col = "lightblue"
                case _:
                    col = "Black"
        else:
            col = "Black"
        plots.append(
            go.Scatter(
                x=dff[timeF],
                y=dff[dataF],
                name="Patient " + str(label),
                marker=dict(color=col),
            )
        )

    fig = go.Figure()

    for trace in plots:
        fig.add_trace(trace)

    fig["layout"].update(height=700, width=900, title=title, xaxis=dict(tickangle=-90))

    fig.show()
    fig.write_image(fn + dataF + ".pdf")


# This generates a PPS Matrix
# read also for PPS: https://www.kaggle.com/code/frtgnn/predictive-power-score-vs-correlation
def generatePPS(df, title, fn="figures/pps.pdf"):
    navy_cmap = sns.light_palette("Navy", as_cmap=True)

    plt.figure(figsize=(16, 12))
    matrix_df = pps.matrix(df, sorted=False)[["x", "y", "ppscore"]].pivot_table(
        columns="x", index="y", values="ppscore", sort=True
    )

    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        matrix_df, vmin=0, vmax=1, cmap=navy_cmap, linewidths=0.5, annot=True, fmt=".2f"
    )
    ax.set(xlabel="Features", ylabel="Target", title=title)
    fig = ax.get_figure()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


### This generates a clustered PPS Matrix, the cluster_order has to be set to match the see correlation matrix clustering
def generatePPS_clustered(df, title, fn="figures/pps_clustered.pdf"):
    navy_cmap = sns.light_palette("Navy", as_cmap=True)

    plt.figure(figsize=(16, 12))
    matrix_df = pps.matrix(df, sorted=False)[["x", "y", "ppscore"]].pivot_table(
        columns="x", index="y", values="ppscore", sort=True
    )
    cluster_order = [
        "baseline_gastric_tube",
        "gastric_tube",
        "ventilation_score",
        "ventilation_term",
        "mwt_motor_score",
        "sma_type",
        "smn2_copies",
        "hfmse_motor_score",
        "rulm_score_calc",
        "chop_motor_score",
        "hine_motor_score",
        "bmi",
        "head_circumference",
        "scoliosis_yn",
        "weight",
        "age_assess",
        "height",
        "base_ventilation_score",
        "age_of_onset_months",
        "motormiles_score",
        "devices_score",
        "ulf_sum",
        "interventions_sum",
        "orthoses_score",
        "first_symptoms_sum",
        "smn1_mut_type",
        "contractures_score",
        "height_percentile",
    ]
    matrix_df = matrix_df[cluster_order]
    matrix_df = matrix_df.T[cluster_order]

    ax = sns.heatmap(
        matrix_df.T, vmin=0, vmax=1, cmap=navy_cmap, linewidths=0.5, fmt=".2f"
    )
    ax.set(xlabel="Features", ylabel="Target", title=title)
    fig = ax.get_figure()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


### This generates an annotated, clustered PPS Matrix, the cluster_order has to be set to match the see correlation matrix clustering
def generatePPS_clustered_annotated(
    df, title, fn="figures/pps_clustered_annotated.pdf"
):
    navy_cmap = sns.light_palette("Navy", as_cmap=True)

    plt.figure(figsize=(16, 12))
    matrix_df = pps.matrix(df, sorted=False)[["x", "y", "ppscore"]].pivot_table(
        columns="x", index="y", values="ppscore", sort=True
    )
    cluster_order = [
        "baseline_gastric_tube",
        "gastric_tube",
        "ventilation_score",
        "ventilation_term",
        "mwt_motor_score",
        "sma_type",
        "smn2_copies",
        "hfmse_motor_score",
        "rulm_score_calc",
        "chop_motor_score",
        "hine_motor_score",
        "bmi",
        "head_circumference",
        "scoliosis_yn",
        "weight",
        "age_assess",
        "height",
        "base_ventilation_score",
        "age_of_onset_months",
        "motormiles_score",
        "devices_score",
        "ulf_sum",
        "interventions_sum",
        "orthoses_score",
        "first_symptoms_sum",
        "smn1_mut_type",
        "contractures_score",
        "height_percentile",
    ]

    matrix_df = matrix_df[cluster_order]
    matrix_df = matrix_df.T[cluster_order]

    ax = sns.heatmap(
        matrix_df.T,
        vmin=0,
        vmax=1,
        cmap=navy_cmap,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
    )
    ax.set(xlabel="Features", ylabel="Target", title=title)
    fig = ax.get_figure()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


# This creates a PPS Ranking Plot
def featureScorePlot(
    df, titlePrefix, field="scoliosis_yn", fn="figures/featureScore_", fixed=False
):
    fixed_width = 0.8
    plt.figure(figsize=(16, 12))

    title = titlePrefix + " " + field
    predictors_df = pps.predictors(df, y=field)
    # sns.set(font_scale=0.7)
    if not fixed:
        ax = sns.barplot(data=predictors_df, y="x", x="ppscore", palette="Blues_d")
    else:
        ax = sns.barplot(
            data=predictors_df, y="x", x="ppscore", palette="Blues_d", width=fixed_width
        )
    ax.set(xlabel="Predictive Power Score", ylabel="Features", title=title)
    fig = ax.get_figure()
    fig.savefig(fn + field + ".pdf", dpi=DPI, bbox_inches="tight")


# This plots the Correlation Matrix
def plotCorrelation(dftemp, title, threshold=None, fn="figures/Correlation_Matrix.pdf"):
    dfclean = dftemp.copy()
    dfclean = dfclean.reindex(sorted(dfclean.columns), axis=1)
    if threshold:
        dfclean = dfclean.corr().abs()
        upper_tri = dfclean.where(np.triu(np.ones(dfclean.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]
        cols = [col for col in dftemp.columns.values.tolist() if col not in to_drop]
        dfclean = dftemp[cols]
    else:
        dfclean = dfclean.corr().abs()

    plt.figure(figsize=(16, 12))

    sns.set(font_scale=1.2)
    ax = sns.heatmap(
        dfclean, vmin=0, vmax=1, cmap="BuPu", linewidths=0.5, annot=True, fmt=".2f"
    )
    ax.set(xlabel="Features", ylabel="Features", title=title)
    fig = ax.get_figure()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


############### REMOVE BELOW LATER #################
def plotCorrelation_abs(
    dftemp, title, threshold=None, fn="figures/correlation_abs.pdf"
):
    dfclean = dftemp.copy()
    dfclean = dfclean.reindex(sorted(dfclean.columns), axis=1)
    if threshold:
        dfclean = dfclean.corr()
        upper_tri = dfclean.where(np.triu(np.ones(dfclean.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]
        cols = [col for col in dftemp.columns.values.tolist() if col not in to_drop]
        dfclean = dftemp[cols]
    else:
        dfclean = dfclean.corr()

    plt.figure(figsize=(16, 12))

    ax = sns.heatmap(
        dfclean, vmin=0, vmax=1, cmap="BuPu", linewidths=0.5, annot=True, fmt=".2f"
    )
    ax.set(xlabel="Features", ylabel="Features", title=title)
    fig = ax.get_figure()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


############### REMOVE ABOVE LATER #################


# This creates a clustered Correlation Matrix
def plotClustermap(dftemp, title, threshold=None, fn="figures/clustermap.pdf"):
    dfclean = dftemp.copy()
    dfclean = dfclean.reindex(sorted(dfclean.columns), axis=1)
    if threshold:
        dfclean = dfclean.corr().abs()
        upper_tri = dfclean.where(np.triu(np.ones(dfclean.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]
        cols = [col for col in dftemp.columns.values.tolist() if col not in to_drop]
        dfclean = dftemp[cols]
    else:
        dfclean = dfclean.corr().abs()

    plt.figure(figsize=(16, 12))

    sns.clustermap(
        dfclean.fillna(0), vmin=0, vmax=1, cmap="BuPu", linewidths=0.5, fmt=".2f"
    )
    fig = plt.gcf()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


def plotClustermap_annotated(
    dftemp, title, threshold=None, fn="figures/clustermap_annotated.pdf"
):
    dfclean = dftemp.copy()
    dfclean = dfclean.reindex(sorted(dfclean.columns), axis=1)
    if threshold:
        dfclean = dfclean.corr().abs()
        upper_tri = dfclean.where(np.triu(np.ones(dfclean.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]
        cols = [col for col in dftemp.columns.values.tolist() if col not in to_drop]
        dfclean = dftemp[cols]
    else:
        dfclean = dfclean.corr().abs()

    plt.figure(figsize=(16, 12))

    sns.set(font_scale=1.5)
    sns.clustermap(
        dfclean.fillna(0),
        vmin=0,
        vmax=1,
        cmap="BuPu",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
    )
    fig = plt.gcf()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


#### ADAPTED FOR ABSOLUTE VALUES ###
def plotClustermap_annotated_abs(
    dftemp, title, threshold=None, fn="figures/Clustermap_annotated_abs.pdf"
):
    dfclean = dftemp.copy()
    dfclean = dfclean.reindex(sorted(dfclean.columns), axis=1)
    if threshold:
        dfclean = dfclean.corr()
        upper_tri = dfclean.where(np.triu(np.ones(dfclean.shape), k=1).astype(bool))
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]
        cols = [col for col in dftemp.columns.values.tolist() if col not in to_drop]
        dfclean = dftemp[cols]
    else:
        dfclean = dfclean.corr()

    plt.figure(figsize=(16, 12))
    sns.clustermap(
        dfclean.fillna(0),
        vmin=0,
        vmax=1,
        cmap="BuPu",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
    )
    fig = plt.gcf()
    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


def draw_histograms(dfr, variables, n_rows, n_cols, fn="figures/histograms.pdf"):
    fig = plt.figure(figsize=(20, 16))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        sns.histplot(dfr[var_name], bins=50)
        ax.set_title(var_name + " Distribution")

        freq = int(2)
        xtix = ax.get_xticks()
        ax.set_xticks(xtix[::freq])
        # nicer label format for dates
        fig.autofmt_xdate()

    fig.savefig(fn, dpi=DPI, bbox_inches="tight")


def draw_hist_age_of_onset(dfr, fn="figures/age_onset_hist.pdf"):
    # define data
    aoo = dfr["age_of_onset"]
    aoo = aoo.astype(str)
    aoo = aoo.fillna("<NA>")

    # renaming data for plotting
    age_of_onset = {
        "0": "   None",
        "1": "Presymp",
        "2": "  At Birth",
        "3": "  Within 4 Wks",
        "4": " < 6M",
        "5": "6-18 M",
        "6": ">18 M",
        "999": "Unknown",
    }

    for key in age_of_onset.keys():
        aoo.replace(key, age_of_onset[key], inplace=True)

    def label_function(val):
        return f"{val / 100 * len(data_df):.0f}\n{val:.2f}%"

    data_df = pd.DataFrame(aoo)
    data_df.groupby("age_of_onset").size().plot(
        kind="bar",
        ylabel="Counts",
        title="Age of Onset Distribution",
        color=sns.color_palette("Blues_d"),
    )
    plt.savefig(fn, dpi=DPI, bbox_inches="tight")


def plot_roc_curve(y_tests, y_preds, fn="figs_featuresX/roc", field="scolios_yn"):
    def average_roc_curves(fprs, tprs):
        """
        Averages multiple ROC curves

        Parameters
        ----------
        fprs: list of numpy arrays
            False Positive Rate for each ROC curve
        tprs: list of numpy arrays
            True Positive Rate for each ROC curve

        Returns
        -------
        avg_fpr: numpy array
            Average False Positive Rate
        avg_tpr: numpy array
            Average True Positive Rate
        """
        avg_fpr = np.zeros_like(fprs[0])
        avg_tpr = np.zeros_like(tprs[0])

        for fpr, tpr in zip(fprs, tprs):
            avg_fpr += fpr
            avg_tpr += tpr

        avg_fpr /= len(fprs)
        avg_tpr /= len(tprs)

        return avg_fpr, avg_tpr

    plt.figure(figsize=(16, 12))

    tprs = []
    # fprs = []
    auc_scores = []
    base_fpr = np.linspace(0, 1, 101)

    for y_test, y_pred in zip(y_tests, y_preds):
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(roc_auc)

        lw = 2
        plt.plot(fpr, tpr, lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, alpha=0.3)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    auc_scores = np.array(auc_scores)
    avg_auc = auc_scores.mean(axis=0)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, "r", label="Avg. ROC curve (area = %0.2f)" % avg_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic: " + field)
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(fn + "_" + field + ".pdf", dpi=DPI, bbox_inches="tight")


def pretty_plot_confusion_matrix(array):
    # get pandas dataframe
    df_cm = pd.DataFrame(array)
    navy_cmap = sns.light_palette("Navy", as_cmap=True)
    pp_matrix(df_cm, cmap=navy_cmap)


def plot_rfc_roc_curve(
    y_tests, y_preds, fn="rfc_roc", field="scolios_yn", title_postfix=""
):
    """
    y_tests is a numpy array of numpy arrays
    y_preds
    """

    def average_roc_curves(fprs, tprs):
        """
        Averages multiple ROC curves

        Parameters
        ----------
        fprs: list of numpy arrays
            False Positive Rate for each ROC curve
        tprs: list of numpy arrays
            True Positive Rate for each ROC curve

        Returns
        -------
        avg_fpr: numpy array
            Average False Positive Rate
        avg_tpr: numpy array
            Average True Positive Rate
        """
        avg_fpr = np.zeros_like(fprs[0])
        avg_tpr = np.zeros_like(tprs[0])

        for fpr, tpr in zip(fprs, tprs):
            avg_fpr += fpr
            avg_tpr += tpr

        avg_fpr /= len(fprs)
        avg_tpr /= len(tprs)

        return avg_fpr, avg_tpr

    plt.figure(figsize=(16, 12))

    tprs = []
    auc_scores = []
    base_fpr = np.linspace(0, 1, 101)

    for y_test, y_pred in zip(y_tests, y_preds):
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(roc_auc)

        lw = 2
        plt.plot(fpr, tpr, lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, alpha=0.3)

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    auc_scores = np.array(auc_scores)
    avg_auc = auc_scores.mean(axis=0)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, "r", label="Avg. ROC curve (area = %0.2f)" % avg_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic: " + field + str(title_postfix))
    plt.legend(fontsize="16", loc="lower right")

    plt.savefig(fn + "_" + field + ".pdf", dpi=DPI, bbox_inches="tight")
    return avg_auc, base_fpr, mean_tprs


def plot_multiple_histograms(df, col_names, n_rows, n_cols):
    fig = plt.figure(figsize=(20, 16))
    for i, col_name in enumerate(col_names):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[col_name].hist(bins=50, ax=ax)
        ax.set_title(col_name + " Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


def n_loops_over_groupKfold_models_revisions(
    n_random_states, df_train, traincols, label, target_dir
):
    X = df_train[traincols].fillna(
        value=-1
    )  # This is the feature_names for feature importance

    # Target
    y = df_train[label].values

    # Groups
    groups = df_train["record_id"].values

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    avg_roc_list = []
    base_fpr_list = []
    mean_tprs_list = []

    # loop through n-random states for stratified groupKFold
    for random_state in n_random_states:
        rng = np.random.RandomState(seed=random_state)

        # Get Folds for this random state
        train_folds_n, test_folds_n, _, _ = ml_models.random_groupKfold_model_4_loop(
            X, y, groups, rng, n_splits=10, stratified=True
        )

        # Train on these folds
        _, _, _, _, y_test_list_n, y_pred_proba_list_n, _, _ = (
            ml_models.train_rfc_models(
                X,
                y,
                df_train,
                train_folds_n,
                test_folds_n,
                groups,
                random_state=random_state,
            )
        )

        avg_roc_n, base_fpr_n, mean_tprs_n = plot_rfc_roc_curve_revisions(
            y_test_list_n,
            y_pred_proba_list_n,
            fn="rfc_roc",
            field="scolios_yn ",
            title_postfix=random_state,
        )

        avg_roc_list.append(avg_roc_n)
        base_fpr_list.append(base_fpr_n)
        mean_tprs_list.append(mean_tprs_n)

        print("For Random State %f | avg ROC score %f" % (random_state, avg_roc_n))

    plt.figure(figsize=(16, 12))

    lw = 2
    field = "scolios_yn"

    # Loop through the traincols and train the model on each feature
    for i, traincol in enumerate(traincols):
        # Train model on single features
        _, _, _, _, y_test_list_n, y_pred_proba_list_n, _, _ = (
            ml_models.train_rfc_models(
                X[:, i].reshape(-1, 1),
                y,
                df_train,
                train_folds_n,
                test_folds_n,
                groups,
                random_state=random_state,
            )
        )

        # Save metrics for single features
        avg_roc_n, base_fpr_n, mean_tprs_n = plot_rfc_roc_curve_revisions(
            y_test_list_n,
            y_pred_proba_list_n,
            fn="rfc_roc",
            field="scolios_yn ",
            title_postfix=random_state,
        )

        plt.plot(
            base_fpr_n,
            mean_tprs_n,
            label=f"ROC curve of '{traincol}' (area = {avg_roc_n.round(2)})",
            lw=lw,
            alpha=0.3,
        )

        print(
            "Single: For Random State %f | avg ROC score %f" % (random_state, avg_roc_n)
        )

    auc_scores = np.array(avg_roc_list)
    avg_auc = auc_scores.mean(axis=0)

    print("Average AUC score: %f" % avg_auc)

    tprs = np.array(mean_tprs_list)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    base_fpr = base_fpr_list[0]

    plt.plot(base_fpr, mean_tprs, "r", label="Avg. ROC curve (area = %0.2f)" % avg_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.3)
    plt.grid(False)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic: " + field)
    plt.legend(fontsize="16", loc="lower right")
    plt.savefig(
        target_dir + "n_loops_ROC_vs_single_features.pdf", dpi=DPI, bbox_inches="tight"
    )

    plt.show()

    return np.mean(avg_roc_n), np.std(avg_roc_n)


def plot_rfc_roc_curve_revisions(
    y_tests, y_preds, fn="rfc_roc", field="scolios_yn", title_postfix=""
):
    """
    y_tests is a numpy array of numpy arrays
    y_preds
    """

    def average_roc_curves(fprs, tprs):
        """
        Averages multiple ROC curves

        Parameters
        ----------
        fprs: list of numpy arrays
            False Positive Rate for each ROC curve
        tprs: list of numpy arrays
            True Positive Rate for each ROC curve

        Returns
        -------
        avg_fpr: numpy array
            Average False Positive Rate
        avg_tpr: numpy array
            Average True Positive Rate
        """
        avg_fpr = np.zeros_like(fprs[0])
        avg_tpr = np.zeros_like(tprs[0])

        for fpr, tpr in zip(fprs, tprs):
            avg_fpr += fpr
            avg_tpr += tpr

        avg_fpr /= len(fprs)
        avg_tpr /= len(tprs)

        return avg_fpr, avg_tpr

    # plt.figure(figsize=(16,12))

    tprs = []
    # fprs = []
    auc_scores = []
    base_fpr = np.linspace(0, 1, 101)

    for y_test, y_pred in zip(y_tests, y_preds):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(roc_auc)

        # plt.plot(
        #     fpr,
        #     tpr,
        #     lw=lw,
        #     label="ROC curve (area = %0.2f)" % roc_auc,
        #     alpha=0.3
        # )

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        # fprs.append(fpr)

    # Plot average
    # does not work nicely
    # avg_fpr, avg_tpr = average_roc_curves(fprs,tprs)
    # plt.plot(avg_fpr, avg_tpr, 'r')

    auc_scores = np.array(auc_scores)
    avg_auc = auc_scores.mean(axis=0)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    np.minimum(mean_tprs + std, 1)
    mean_tprs - std
    return avg_auc, base_fpr, mean_tprs


# groupKfold visualization
def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )

    #### FIGURE THIS OUT TO GET A BETTER RESOLUTION OF GROUPS ####
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["Data\ngroup", "Data\nclass"],
        xlabel="Sample index",
    )


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["scoliosis", "patients"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="Cross-Validation iteration",
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    ax.set_facecolor("white")
    ax.grid(False)
    return ax
