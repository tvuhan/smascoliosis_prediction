"""Machine Learning"""

__author__ = [
    "Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)",
    "Rodrigo Bermudez Schettino (Max Planck Institute for Human Development)",
    "Vikram Sunkara (Zuse Institute Berlin)",
]
__maintainer__ = ["Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)"]

import copy

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz

import plots

RANDOM_STATE = 23


### define weights for an ML model ###
def weight_data(df, column_one="scoli_pacs", column_two="scoli_confidence"):
    """
    This function assigns classification confidence of a target to a Pandas dataframe based on the values provided in two columns.

    Args:
        df (pandas.DataFrame): Input dataframe with two columns containing binary values.

    Returns:
        pandas.DataFrame: Updated dataframe with a new column 'classification_confidence' containing the calculated weights.
    """
    # Column names that influence the weights.
    # The values in these columns should be 1 or 0.
    # column_one = 'scoli_pacs'
    # column_two = 'scoli_confidence'

    # Weights
    # The weights stand for the certainty of the data.
    strong_weight = 3
    medium_weight = 2
    weak_weight = 0.5

    weights_dict = {
        # (column_one, column_two): weight
        (1, 1): strong_weight,
        (1, 0): medium_weight,
        (0, 1): medium_weight,
        (0, 0): weak_weight,
    }

    weights = []
    for _, row in df.iterrows():
        # Select the weight based on the values in the two columns.
        weights.append(weights_dict[(row[column_one], row[column_two])])

    df_weighted = df.copy()
    df_weighted["classification_confidence"] = weights

    return df_weighted


#### generate decision trees of a RFC model ####
def generate_decision_tree(
    models,
    feature_names,
    fn="Decision Tree",
    model_i=0,
    estimator_idx=np.arange(3, 10),
    file_extension="pdf",
):
    for estimator_i in estimator_idx:
        tree = models[model_i].estimators_[estimator_i]
        dot_data = export_graphviz(
            tree,
            feature_names=feature_names,
            filled=True,
            max_depth=6,
            impurity=False,
            proportion=True,
        )
        graph = graphviz.Source(dot_data)

        graph.format = file_extension
        graph.render(f"{fn} {estimator_i}", view=False)


#### Test Models PredictX Project
def predictX_rf_test(df, cols, label, cross_val_runs=5, probs=True):
    traincols = copy.copy(cols)
    traincols.append(label)

    df_train = df[traincols].fillna(value=-1)

    # turn into binary classification for the moment
    df_train[[label]] = df_train[[label]].apply(lambda x: abs(x), axis=1)
    df_train[[label]] = df_train[[label]].fillna(value=0)

    y_tests = []
    y_preds = []
    y_preds_proba = []

    for i in range(cross_val_runs):
        y_test, y_pred, y_pred_proba = random_forest_results(df_train, random_state=i)
        proba_0, proba_1 = list(zip(*y_pred_proba))
        y_tests.append(y_test)
        y_preds.append(y_pred)
        y_preds_proba.append(proba_1)

        print("Metrics for fold", i)
        metrics(y_test, y_pred)
        print("\n\n")

    print("\n\nROC Curve\n")
    if probs:
        plots.plot_roc_curve(y_tests, y_preds_proba)
    else:
        plots.plot_roc_curve(y_tests, y_preds)


##### ADD A BAYESIAN LINEAR MODEL #####


#################################### Models in FeaturesX Project ###########################################################


def random_forest_results(df_train, random_state=1):
    n = len(df_train.columns)
    X = df_train.iloc[:, 0 : n - 1].values
    y = df_train.iloc[:, n - 1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    return y_test, y_pred, y_pred_proba


# Evaluate model
def metrics(y_test, y_pred):
    from sklearn.metrics import (
        confusion_matrix,
    )

    plots.pretty_plot_confusion_matrix(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def simple_rf_test(df, cols, label, cross_val_runs=5, probs=True):
    traincols = copy.copy(cols)
    traincols.append(label)

    df_train = df[traincols].fillna(value=-1)

    # turn into binary classification for the moment
    df_train[[label]] = df_train[[label]].apply(lambda x: abs(x), axis=1)
    df_train[[label]] = df_train[[label]].fillna(value=0)

    y_tests = []
    y_preds = []
    y_preds_proba = []

    for i in range(cross_val_runs):
        y_test, y_pred, y_pred_proba = random_forest_results(df_train, random_state=i)
        proba_0, proba_1 = list(zip(*y_pred_proba))
        y_tests.append(y_test)
        y_preds.append(y_pred)
        y_preds_proba.append(proba_1)

        print("Metrics for fold", i)
        metrics(y_test, y_pred)
        print("\n\n")

    print("\n\nROC Curve\n")
    if probs:
        plots.plot_roc_curve(y_tests, y_preds_proba)
    else:
        plots.plot_roc_curve(y_tests, y_preds)


def random_groupKfold_model(
    df_train, traincols, label, n_splits=10, random_state=RANDOM_STATE, stratified=True
):
    rng = np.random.RandomState(seed=random_state)

    # Load data and define groups
    # Features
    X = df_train[traincols].fillna(
        value=-1
    )  # This is the feature_names for feature importance
    feature_names = X.columns
    # Target
    y = df_train[label].values
    # Groups
    groups = df_train["record_id"].values

    gkf = None
    if stratified:
        # Remove random_state from function because setting random_state causes the following error:
        #
        # ValueError: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.

        gkf = StratifiedGroupKFold(n_splits=n_splits)  # recommended between 3 and 10
    else:
        gkf = GroupKFold(
            n_splits=n_splits, random_state=random_state
        )  # recommended between 3 and 10

    # Initialize lists
    train_folds = []
    test_folds = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        # randomly shuffle the train set
        rng.shuffle(train_idx)
        train_folds.append(train_idx)
        # randomly shuffle the test set
        rng.shuffle(test_idx)
        test_folds.append(test_idx)
    return X, y, train_folds, test_folds, groups, feature_names, n_splits


def random_groupKfold_model_4_loop(X, y, groups, rng, n_splits=10, stratified=True):
    gkf = None
    if stratified:
        gkf = StratifiedGroupKFold(n_splits=n_splits)
    else:
        gkf = GroupKFold(n_splits=n_splits)

    # Initialize lists
    train_folds = []
    test_folds = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        # randomly shuffle the train set
        rng.shuffle(train_idx)
        train_folds.append(train_idx)

        # randomly shuffle the test set
        rng.shuffle(test_idx)
        test_folds.append(test_idx)

    return train_folds, test_folds, groups, n_splits


def n_loops_over_groupKfold_models(
    n_random_states, df_train, traincols, label, target_dir, dpi
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
        train_folds_n, test_folds_n, groups_n, n_splits_n = (
            random_groupKfold_model_4_loop(
                X, y, groups, rng, n_splits=10, stratified=True
            )
        )

        # Train on these folds
        (
            scores_n,
            X_train_list_n,
            y_train_list_n,
            y_pred_list_n,
            y_test_list_n,
            y_pred_proba_list_n,
            models_n,
            X_test_list_n,
        ) = train_rfc_models(
            X,
            y,
            df_train,
            train_folds_n,
            test_folds_n,
            groups,
            random_state=random_state,
        )

        avg_roc_n, base_fpr_n, mean_tprs_n = plots.plot_rfc_roc_curve(
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

    for i in range(len(avg_roc_list)):
        plt.plot(
            base_fpr_list[i],
            mean_tprs_list[i],
            label="ROC curve (area = %0.2f)" % avg_roc_list[i],
            lw=lw,
            alpha=0.3,
        )
        plt.savefig(
            target_dir + str(i) + "th_loop_ROC_.pdf", dpi=dpi, bbox_inches="tight"
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
    plt.savefig(target_dir + "n_loops_ROC.pdf", dpi=dpi, bbox_inches="tight")

    plt.show()

    return np.mean(avg_roc_n), np.std(avg_roc_n)


def train_rfc_models_k_fold(X, y, df_train, train_folds, test_folds, groups):
    # incorporate weights using columns 'scoli_pacs' and 'scoliosis_confidence'
    df_train = weight_data(df_train)
    weights = df_train["classification_confidence"].values

    scores = []  # accuracy_score
    X_train_list = []
    y_train_list = []
    y_pred_list = []
    y_test_list = []
    y_pred_proba_list = []
    models = []
    X_test_list = []

    for i in range(len(train_folds)):
        train_idx = train_folds[i]
        test_idx = test_folds[i]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=RANDOM_STATE
        )
        model.fit(
            X_train, y_train, sample_weight=weights[train_idx]
        )  # incorporates weights in scoli_pacs and scoliosis_confidence

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        scores.append(accuracy_score(y_test, y_pred))
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)
        y_pred_proba_list.append(y_pred_proba)
        models.append(model)
        X_test_list.append(X_test)

    return (
        scores,
        X_train_list,
        y_train_list,
        y_pred_list,
        y_test_list,
        y_pred_proba_list,
        models,
        X_test_list,
    )


def multiple_model_metrics(
    models,
    y_test_list,
    y_pred_list,
    f1_scores_0,
    f1_scores_1,
    mcc_scores_0,
    mcc_scores_1,
):
    for i in range(len(models)):
        # display(Markdown(f"# Metrics for fold {i}"))
        plots.pretty_plot_confusion_matrix(
            confusion_matrix(y_test_list[i], y_pred_list[i])
        )

        f1_scores_0.append(f1_score(y_test_list[i], y_pred_list[i], pos_label=0))

        f1_scores_1.append(f1_score(y_test_list[i], y_pred_list[i], pos_label=1))

        mcc_scores_0.append(matthews_corrcoef(y_test_list[i], y_pred_list[i]))

        mcc_scores_1.append(matthews_corrcoef(y_test_list[i], y_pred_list[i]))


def train_rfc_models(
    X, y, df_train, train_folds, test_folds, groups, random_state=RANDOM_STATE
):
    # incorporate weights using columns 'scoli_pacs' and 'scoliosis_confidence'
    df_train = weight_data(df_train)
    weights = df_train["classification_confidence"].values

    scores = []  # accuracy_score
    X_train_list = []
    y_train_list = []
    y_pred_list = []
    y_test_list = []
    y_pred_proba_list = []
    models = []
    X_test_list = []

    for i in range(len(train_folds)):
        train_idx = train_folds[i]
        test_idx = test_folds[i]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=random_state
        )
        model.fit(
            X_train, y_train, sample_weight=weights[train_idx]
        )  # incorporates weights in scoli_pacs and scoliosis_confidence

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        scores.append(accuracy_score(y_test, y_pred))
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        y_pred_list.append(y_pred)
        y_test_list.append(y_test)
        y_pred_proba_list.append(y_pred_proba[:, 1])
        models.append(model)
        X_test_list.append(X_test)

    return (
        scores,
        X_train_list,
        y_train_list,
        y_pred_list,
        y_test_list,
        y_pred_proba_list,
        models,
        X_test_list,
    )


def get_gini_importance(
    model, traincols, df_train, target_dir, dpi, fn="Feature Importance"
):
    sorted_idx = model.feature_importances_.argsort()
    plt.figure()
    plt.barh(
        df_train[traincols].fillna(value=-1).columns[sorted_idx],
        model.feature_importances_[sorted_idx],
    )
    plt.xlim(0, 1)
    plt.xlabel("Random Forest Feature Importance", fontsize=12)
    plt.title(fn, fontsize=12)
    plt.figtext(x=0, y=1, s="", va="top", ha="left", fontsize=14, fontweight="bold")
    plt.savefig(target_dir + fn + ".pdf", dpi=dpi, bbox_inches="tight")
