"""
This utils file contains code for SMAScoliosis data set clean up, one-hot encoding, data processing, and feature engineering.
"""

__author__ = [
    "Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)",
    "Rodrigo Bermudez Schettino (Max Planck Institute for Human Development)",
]
__maintainer__ = ["Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)"]

import numpy as np
import pandas as pd

import utils


def feat_airwayclear(df):
    def airwayclear_arr(airway_clearance, airway_clearance_type):
        if (
            not pd.isna(airway_clearance)
            and airway_clearance == 0
            and airway_clearance_type == ""
        ):  # none
            return np.array([0, 0, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == ""
        ):
            return np.array([1, 0, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "1"
        ):
            return np.array([1, 1, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "1, 2"
        ):
            return np.array([1, 1, 1, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "1, 2, 3"
        ):
            return np.array([1, 1, 1, 1])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "3"
        ):
            return np.array([1, 0, 0, 1])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "1, 3"
        ):
            return np.array([1, 1, 0, 1])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "2, 3"
        ):
            return np.array([1, 0, 1, 1])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "2"
        ):
            return np.array([1, 0, 1, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "2, 888"
        ):
            return np.array([1, 0, 1, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "888"
        ):
            return np.array([1, 0, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "999"
        ):
            return np.array([1, 0, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "1, 888"
        ):
            return np.array([1, 1, 0, 0])
        elif (
            not pd.isna(airway_clearance)
            and airway_clearance == 1
            and airway_clearance_type == "888"
        ):
            return np.array([1, 0, 0, 0])
        else:
            return np.array([-1, -1, -1, -1])

    def airwayclear_score(airwayclear_arr):
        if np.array_equal(airwayclear_arr, np.array([0, 0, 0, 0])):
            return 0
        elif airwayclear_arr[0] == -1:
            return -1
        else:
            return sum(airwayclear_arr)
            # return sum(2**n * airwayclear_arr[n] for n in range(len(airwayclear_arr)))

    def airwayclear_term(airwayclear_arr, airway_clearance_freq):
        # reassign for weighting
        df["airway_clearance_freq"].replace(1, 4, inplace=True)
        df["airway_clearance_freq"].replace(2, 3, inplace=True)
        df["airway_clearance_freq"].replace(3, 2, inplace=True)
        df["airway_clearance_freq"].replace(np.nan, 1, inplace=True)
        if np.array_equal(airwayclear_arr, np.array([0, 0, 0, 0])):
            return 0
        elif airwayclear_arr[0] == -1:
            return -1
        else:
            return airway_clearance_freq * sum(airwayclear_arr)
        # return airway_clearance_freq * sum(2**n * airwayclear_arr[n] for n in range(len(airwayclear_arr)))

    df["airwayclear_arr"] = df.apply(
        lambda x: airwayclear_arr(x.airway_clearance, x.airway_clearance_type), axis=1
    )
    df["airwayclear_score"] = df.apply(
        lambda x: airwayclear_score(x.airwayclear_arr), axis=1
    )
    df["airwayclear_term"] = df.apply(
        lambda x: airwayclear_term(x.airwayclear_arr, x.airway_clearance_freq), axis=1
    )

    return df


def fix_growth_percentiles(df):
    def weight_percs_fixed(gender, weight, age_assess):
        measurement_type = "weight_cdc"
        if not pd.isna(weight):
            return utils.get_perc_sigmoid_fit(
                gender, weight, age_assess, measurement_type
            )
        else:
            return np.nan

    def height_percs_fixed(gender, height, age_assess):
        measurement_type = "length_cdc"
        if not pd.isna(height):
            return utils.get_perc_sigmoid_fit(
                gender, height, age_assess, measurement_type
            )
        else:
            return np.nan

    def headc_percs_fixed(gender, head_circumference, age_assess):
        measurement_type = "headc_cdc"
        if not pd.isna(head_circumference):
            return utils.get_perc_sigmoid_fit(
                gender, head_circumference, age_assess, measurement_type
            )
        else:
            return np.nan

    df_growth_percentiles = df.copy(
        deep=True
    )  # deep=True means a true copy without referencing the old data frame

    ## THIS GIVES A WARNING MESSAGE POTENTIALLY NEEDS TO BE FIXED
    df_growth_percentiles["weight_percs_fixed"] = df.apply(
        lambda x: weight_percs_fixed(x.gender, x.weight, x.age_assess), axis=1
    )
    df_growth_percentiles["height_percs_fixed"] = df.apply(
        lambda x: height_percs_fixed(x.gender, x.height, x.age_assess), axis=1
    )
    df_growth_percentiles["headc_percs_fixed"] = df.apply(
        lambda x: headc_percs_fixed(x.gender, x.head_circumference, x.age_assess),
        axis=1,
    )

    return df_growth_percentiles


# FIX ONE HOT ENCODINGS AND MULTI ENCODINGS
def fix_encodings(df):
    # Process hot encoded columns

    # process "origin"
    base = "origin"
    shift = [1, 2, 3, 4, 5, 6, 7, 888, 999]
    df = utils.process_column_coding(df, base, shift)

    # process 'orthoses_use_type' 0-3, 999
    base = "orthoses_use_type"
    shift = [0, 2, 3, 999]
    df = utils.process_column_coding(df, base, shift)

    # process 'last_visit_death' 0-5, 1000, 999
    base = "last_visit_death"
    shift = [0, 1, 2, 3, 4, 5, 1000, 999]
    df = utils.process_column_coding(df, base, shift)

    # process 'best_motor' 1-5, 0, 999
    base = "best_motor"
    shift = [0, 1, 2, 3, 4, 5, 999]
    df = utils.process_column_coding(df, base, shift)

    # now replace 888, 999, 1000 by N/A
    df = utils.convert999toNA(df)

    # MULTI HOT ENCODING

    # [Multiple Checkboxes] process "first_symptoms"
    base = "first_symptoms"
    shift = [0, 2, 3, 4, 5, 6, 7, 8, 9, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] process "airway_clearance_type"
    base = "airway_clearance_type"
    shift = [1, 2, 3, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] process 'interventions_type' 0-4, 888, 999
    base = "interventions_type"
    shift = [0, 2, 3, 4, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'orthoses_type' 0-8, 10, 888, 999
    base = "orthoses_type"
    shift = [0, 2, 5, 6, 7, 8, 10, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes]process 'device_type' 1-4, 888, 999
    base = "device_type"
    shift = [1, 2, 3, 4, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'motor_score_test' 1-8, 888, 999
    base = "motor_score_test"
    shift = [1, 2, 3, 4, 5, 6, 7, 8, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)
    # the presence of a motor_score_test may be informative of age, motor development

    # [Multiple Checkboxes] 'contractures_loco' 1-5, 888, 999
    base = "contractures_loco"
    shift = [1, 2, 3, 4, 5, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'nus_complications' 0,2,3,4, 888, 6, 999
    base = "nus_complications"
    shift = [0, 2, 3, 4, 6, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'ona_complications' 0-6, 888, 7, 999
    base = "ona_complications"
    shift = [0, 2, 3, 4, 5, 6, 7, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'ona_complications_2' 0, 2, 3, 4, 5, 6, 888, 7, 999
    base = "ona_complications_2"
    shift = [0, 2, 3, 4, 5, 6, 7, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'ris_complications' 0, 2, 3, 4, 5, 6, 7, 8, 9, 888, 999
    base = "ris_complications"
    shift = [0, 2, 3, 4, 5, 6, 7, 8, 9, 888, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [2 Checkboxes] 'xray_apcobb_body_1' 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999
    base = "xray_apcobb_body_1"
    shift = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [2 Checkboxes] 'xray_apcobb_body_2' 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999
    base = "xray_apcobb_body_2"
    shift = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'spine_complications_type' 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999
    base = "spine_complications_type"
    shift = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    # [Multiple Checkboxes] 'complication_2' 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999
    base = "complication_2"
    shift = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 999]
    df = utils.process_multi_column_coding(df, base, shift)

    return df


# FIX ULF MOTOR SCORES
def fix_ulf_scores(df):
    df["ulf_sum"] = df.loc[:, ["ulf_1", "ulf_2", "ulf_3"]].sum(axis=1)
    df = utils.floatToIntsNA(df, ["ulf_sum"])

    return df


##### orthoses score #####


def feat_orthoses(df):
    df["orthoses_type_other"].replace("Oberschenkel Orthesen", "7", inplace=True)
    df["orthoses_type_other"].replace("OS-Orthesen", "7", inplace=True)
    df["orthoses_type_other"].replace("OS Ortreten", "7", inplace=True)
    df["orthoses_type_other"].replace("thigh ortheses", "7", inplace=True)
    df["orthoses_type_other"].replace(
        "Oberschenkelorthesen (bis Becken)", "7", inplace=True
    )
    df["orthoses_type_other"].replace("OS Orteten", "7", inplace=True)
    df["orthoses_type_other"].replace("Knee", "7", inplace=True)
    df["orthoses_type_other"].replace("RGO", "7", inplace=True)
    df["orthoses_type_other"].replace(
        "Oberschenkel Orthesen, Korsett", "7, 10", inplace=True
    )
    df["orthoses_type_other"].replace("Korsett, CDS-Schiene", "7, 10", inplace=True)
    df["orthoses_type_other"].replace("RGO, Korsett", "7, 10", inplace=True)
    df["orthoses_type_other"].replace("Korsett", "10", inplace=True)
    df["orthoses_type_other"].replace("Sitzkorsett", "10", inplace=True)
    df["orthoses_type_other"].replace("corset (flexible and rigid)", "10", inplace=True)
    df["orthoses_type_other"].replace(
        "corset (flexible and rigid), Orthotic type of use", "10", inplace=True
    )
    df["orthoses_type_other"].replace(
        "Sitzkorsett, neu Swash Orthesen zur Nacht", "10", inplace=True
    )
    df["orthoses_type_other"].replace("GPS thoracic", "10", inplace=True)
    df["orthoses_type_other"].replace("CDS Schiene", "3", inplace=True)
    df["orthoses_type_other"].replace("Korsett, CDS-Schiene", "3, 10", inplace=True)
    df["orthoses_type_other"].replace("CDS", "3", inplace=True)
    df["orthoses_type_other"].replace("CDS Schiene", "3", inplace=True)
    df["orthoses_type_other"].replace("CDS Schienen", "3", inplace=True)
    df["orthoses_type_other"].replace("Kopforthese", "4", inplace=True)
    df["orthoses_type_other"].replace("Helmet", "4", inplace=True)
    df["orthoses_type_other"].replace(
        "Tübinger Hüftschiene, Helmet", "4, 9", inplace=True
    )
    df["orthoses_type_other"].replace("Helmtherapie", "4", inplace=True)
    df["orthoses_type_other"].replace("Tübinger Hüftschiene", "9", inplace=True)
    df["orthoses_type_other"].replace(
        "Tübinger Hüftschiene, Helmet", "4, 9", inplace=True
    )
    df["orthoses_type_other"].replace("GPS-Softorthese für Rumpf", "10", inplace=True)
    df["orthoses_type_other"].replace("Tutoren für Stehständer", "888", inplace=True)

    def orthoses_arr(orthoses_yn, orthoses_type, orthoses_type_other):
        # reassigned!
        if (
            not pd.isna(orthoses_yn) and orthoses_yn == 0.0
        ) or orthoses_type == "0":  # NONE or OTHER
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "6"
        ):  # Orthotic shoe inserts -> 2
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "8"
        ):  # Hand splints -> 3
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "4"
        ):  # Helmet 888 -> 4
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "9"
        ):  # Tübinger Hip -> 5
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "5"
        ):  # SMO -> 6
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "2"
        ):  # Ankle-foot Orthoses -> 7
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "3"
        ):  # CDS -> 8
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "7"
        ):  # KAFO -> 9
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "10"
        ):  # SPINAL ORTHOSIS -> 10
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "5, 6, 8"
        ):  # "2, 3, 6"
            return np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "2, 8"
        ):  # "3, 7"
            return np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "7, 8"
        ):  # "3, 9"
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "2, 10"
        ):  # "7, 10"
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "6, 7"
        ):  # "2, 9"
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "2, 7"
        ):  # "7, 9"
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "8, 10"
        ):  # "3, 10"
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "7, 10"
        ):  # "9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 7, 10"
        ):  # "7, 9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "6, 8"
        ):  # "2, 3"
            return np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "5, 7"
        ):  # "6, 9"
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn) and orthoses_yn == 1.0 and orthoses_type == "5, 6"
        ):  # "2, 6"
            return np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 5, 888"
            and orthoses_type_other == "10"
        ):  # "6, 7, 10"
            return np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 7, 888"
            and orthoses_type_other == "888"
        ):  # "7, 9, 888"
            return np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 7, 888"
            and orthoses_type_other == "3"
        ):  # "7, 8, 9"
            return np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 7, 888"
            and orthoses_type_other == "10"
        ):  # "7, 9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 888"
            and orthoses_type_other == "10"
        ):  # "7, 10"
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "2, 888"
            and orthoses_type_other == "3"
        ):  # "7, 8"
            return np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "5, 888"
            and orthoses_type_other == "10"
        ):  # "6, 10"
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "5, 888"
            and orthoses_type_other == "9"
        ):  # "5, 6"
            return np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "6, 7, 888"
            and orthoses_type_other == "10"
        ):  # "2, 9, 10"
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "7, 888"
            and orthoses_type_other == "10"
        ):  # "9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "7, 888"
            and orthoses_type_other == "3"
        ):  # "8, 9"
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "7, 888"
            and orthoses_type_other == "7, 10"
        ):  # "9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and orthoses_type == "8, 888"
            and orthoses_type_other == "10"
        ):  # "3, 10"
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "3"
        ):  # "8"
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "4"
        ):  # "4"
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "7"
        ):  # "9"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "10"
        ):  # "10"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "7, 10"
        ):  # "9, 10"
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        elif (
            not pd.isna(orthoses_yn)
            and orthoses_yn == 1.0
            and "888" in orthoses_type
            and orthoses_type_other == "4, 9"
        ):  # "4, 5"
            return np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        else:
            return np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    def orthoses_score(orthoses_arr):
        orthoses_use_type = 1
        if np.array_equal(orthoses_arr, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
            return 0
        elif orthoses_arr[0] == -1:
            return -1
        else:
            return sum(2**n * orthoses_arr[n] for n in range(len(orthoses_arr))) ** (
                orthoses_use_type / 3
            )  # currently no clear meaning of use_type not factored in

    # spinal orthoses gets the highest score! CAVE: Compare reversed weighting in contractures!!!

    df["orthoses_arr"] = df.apply(
        lambda x: orthoses_arr(x.orthoses_yn, x.orthoses_type, x.orthoses_type_other),
        axis=1,
    )
    df["orthoses_score"] = df.apply(lambda x: orthoses_score(x.orthoses_arr), axis=1)
    return df


######################### clean up medication fields #######################
def clean_medfields(df, med_col):
    # REMOVE SMA THERAPY 2390935 2170231 2170231 and code as 0 = None
    df[med_col].replace("2170231", int(0), inplace=True)
    df[med_col].replace(2170231, int(0), inplace=True)
    df[med_col].replace("2390935", int(0), inplace=True)
    df[med_col].replace(2390935, int(0), inplace=True)

    # Replace Vitamin_D3 Codes 1148597 11253 1244014 1244014 2418 with 1
    df[med_col].replace("vitamin D3", int(1), inplace=True)
    df[med_col].replace("1244014", int(1), inplace=True)
    df[med_col].replace(1244014, int(1), inplace=True)
    df[med_col].replace("373578", int(1), inplace=True)
    df[med_col].replace("2418", int(1), inplace=True)
    df[med_col].replace(2418.0, int(1), inplace=True)
    df[med_col].replace(1244014.0, int(1), inplace=True)
    df[med_col].replace(1148597, int(1), inplace=True)
    df[med_col].replace(11253, int(1), inplace=True)

    # Replace Movicol Codes 8516 8516 with 2
    df[med_col].replace("polyethylene glycols", int(2), inplace=True)
    df[med_col].replace("8516", int(2), inplace=True)
    df[med_col].replace(8516, int(2), inplace=True)

    # Replace # Prednisolone 8638 8638 373578 8640 8638 8638 373578 8640 with 3
    df[med_col].replace("prednisolone", int(3), inplace=True)
    df[med_col].replace("8638", int(3), inplace=True)
    df[med_col].replace(8638, int(3), inplace=True)
    df[med_col].replace("8640", int(3), inplace=True)
    df[med_col].replace("373578", int(3), inplace=True)

    # Replace Asthma_med 1347 41126 151390 1803004 203212 2108226 435 1347 1347 41126 151390 1803004 and code as 4
    df[med_col].replace("1347", int(4), inplace=True)
    df[med_col].replace(1347, int(4), inplace=True)
    df[med_col].replace("41126", int(4), inplace=True)
    df[med_col].replace(41126, int(4), inplace=True)
    df[med_col].replace("151390", int(4), inplace=True)
    df[med_col].replace("1803004", int(4), inplace=True)
    df[med_col].replace(203212.0, int(4), inplace=True)
    df[med_col].replace(226084.0, int(4), inplace=True)
    df[med_col].replace(2108226, int(4), inplace=True)
    df[med_col].replace(435, int(4), inplace=True)

    # Replace Iron 90176 and code as 5
    df[med_col].replace("90176", int(5), inplace=True)

    # Replace ASS 1858285 1858285 and code as 6
    df[med_col].replace("acetylsalicylsalicylic acid", int(6), inplace=True)
    df[med_col].replace("1858285", int(6), inplace=True)
    df[med_col].replace(1858285, int(6), inplace=True)

    # Replace Hydroxyurea 5552 5552 5552 5552 and code as 7
    df[med_col].replace("hydroxyurea", int(7), inplace=True)
    df[med_col].replace("5552", int(7), inplace=True)
    df[med_col].replace(5552, int(7), inplace=True)

    # Replace Diuretics: Furosemid 4603 4603 Spironolactone 9997  5487 with 8
    df[med_col].replace("4603", int(8), inplace=True)
    df[med_col].replace(9997, int(8), inplace=True)
    df[med_col].replace(5487, int(8), inplace=True)

    # Replace PPI 7646 283742 7646 with 9
    df[med_col].replace("7646", int(9), inplace=True)
    df[med_col].replace(7646, int(9), inplace=True)
    df[med_col].replace("283742", int(9), inplace=True)
    df[med_col].replace(283742, int(9), inplace=True)

    # Replace Allergy_med 275635 275635 with 10
    df[med_col].replace("275635", int(10), inplace=True)
    df[med_col].replace(275635.0, int(10), inplace=True)

    # Replace Folsäure/Supplements 4511 with 11
    df[med_col].replace(4511, int(11), inplace=True)

    # Replace Dimethicone with 12 1158750 1158750
    df[med_col].replace(324072, int(12), inplace=True)
    df[med_col].replace(1158750, int(12), inplace=True)

    # Replace Antihypertensiva Captopril with 13
    df[med_col].replace(1998.0, int(13), inplace=True)

    # Replace Antiepileptica with 14
    df[med_col].replace(114477, int(14), inplace=True)

    # Replace EPO!!! with 15 804171
    df[med_col].replace(804171, int(15), inplace=True)

    return df


def clean_growthFields(df, growth_col):
    df[growth_col].replace(99.9, np.nan, inplace=True)
    df[growth_col].replace(999, np.nan, inplace=True)

    return df


############# clean up Age of Onset ###########
def fix_age_onset(df):
    def age_groups(age_onset, age_onset_months):
        if not pd.isna(age_onset) and age_onset == 0:  # None
            return 0
        elif not pd.isna(age_onset) and age_onset == 1:  # prenatal
            return 1
        elif not pd.isna(age_onset) and age_onset == 2:  # at birth
            return 2
        elif not pd.isna(age_onset) and age_onset == 3:  # within the first 4 weeks
            return 3
        elif (
            not pd.isna(age_onset)
            and not pd.isna(age_onset_months)
            and age_onset == 4
            and age_onset_months < 6
        ):
            return 6
        elif (
            not pd.isna(age_onset)
            and not pd.isna(age_onset_months)
            and age_onset == 4
            and age_onset_months >= 6
            and age_onset_months <= 18
        ):
            return 18
        elif (
            not pd.isna(age_onset)
            and not pd.isna(age_onset_months)
            and age_onset == 4
            and age_onset_months > 18
        ):
            return 36
        else:
            return 999

    df["age_of_onset"] = df.apply(
        lambda x: age_groups(x.age_onset, x.age_onset_months), axis=1
    )

    return df


def fix2_age_onset(df):
    def age_of_onset_months(age_onset, age_onset_months):
        if not pd.isna(age_onset) and age_onset == 0:  # None
            return 0
        elif not pd.isna(age_onset) and age_onset == 1:  # prenatal
            return -1
        elif not pd.isna(age_onset) and age_onset == 2:  # at birth
            return 0.1
        elif not pd.isna(age_onset) and age_onset == 3:  # within the first 4 weeks
            return 0.5
        elif (
            not pd.isna(age_onset) and not pd.isna(age_onset_months) and age_onset == 4
        ):  # and age_onset_months == 1:
            return int(age_onset_months)
        else:
            return 999

    df["age_of_onset_months"] = df.apply(
        lambda x: age_of_onset_months(x.age_onset, x.age_onset_months), axis=1
    )

    return df


def fix_fowardfill(df):
    # Fill forward BASELINE FIELDS into every row, grouped by record_id
    baseline_cols = [
        "num_id",
        "birth_date",
        "gender",
        "birthplace",
        "residence",
        "date_genetic_test",
        "sma_test_type",
        "smn2_test_type",
        "smn1_mut_type",
        "smn1_pmut",
        "smn2_copies",
        "sma_type",
        "date_baseline_visit",
        "age_onset",
        "age_onset_months",
        "first_symptoms_other",
        "birth_history",
        "birth_weight",
        "screening_type",
        "siblings_num",
        "parents_consanguinous",
        "family_history",
        "aff_family_member",
        "family_history_2",
        "aff_family_member_2",
        "sit_wo_sup",
        "crawl_hands_knees",
        "stand_wo_sup",
        "walk_wo_sup",
        "climb_stairs",
        "sit_wo_sup_g",
        "sit_wo_sup_l",
        "crawl_hands_knees_g",
        "crawl_hands_knees_l",
        "stand_wo_sup_g",
        "stand_wo_sup_l",
        "walk_wo_sup_g",
        "walk_wo_sup_l",
        "climb_stairs_g",
        "climb_stairs_l",
        "baseline_ventilation",
        "baseline_ventilation_type",
        "baseline_ventilation_start",
        "baseline_gastric_tube",
        "baseline_date_start_gastric_t",
        "baseline_scoliosis_surgery",
        "baseline_surgery_date",
        "baseline_surgery_type",
        "baseline_illness",
        "baseline_illness_numb",
        "baseline_illness_type_test",
        "baseline_illness_type_test_2",
        "baseline_illness_type_test_3",
        "baseline_illness_type_test_4",
        "baseline_illness_type_test_5",
        "baseline_illness_type_test_6",
        "baseline_sma_meds",
        "baseline_sma_meds_other",
    ]
    groups = ["record_id"]

    # ultils.backfilling(df, fields, groups, direction="bf", sort=True, refDate="date_assessment")
    df = utils.backfilling(df, baseline_cols, groups, direction="bf")

    return df


# backfill the constant fields into every row, grouped by record_id
def fix_backfill(df):
    # Fields to fill backward AND forward into every row, grouped by record_id
    cols = [
        "num_id",
        "birth_date",
        "gender",
        "birthplace",
        "residence",
        "origin",
        "date_genetic_test",
        "sma_test_type",
        "smn2_test_type",
        "smn1_mut_type",
        "smn1_pmut",
        "smn2_copies",
        "sma_type",
        "age_onset",
        "age_onset_months",
        "first_symptoms",
        "first_symptoms_other",
        "sit_wo_sup",
        "crawl_hands_knees",
        "stand_wo_sup",
        "walk_wo_sup",
        "climb_stairs",
        "sit_wo_sup_g",
        "sit_wo_sup_l",
        "crawl_hands_knees_g",
        "crawl_hands_knees_l",
        "stand_wo_sup_g",
        "stand_wo_sup_l",
        "walk_wo_sup_g",
        "walk_wo_sup_l",
        "climb_stairs_g",
        "climb_stairs_l",
        "screening_type",
        "siblings_num",
        "parents_consanguinous",
        "family_history",
        "aff_family_member",
        "family_history_2",
        "aff_family_member_2",
        "baseline_illness",
        "baseline_illness_numb",
        "baseline_illness_type_test",
    ]

    # engineered_cols = ['age_of_onset', 'age_of_onset_months', 'first_symptoms_arr','first_symptoms_sum', 'motormiles_arr', 'motormiles_score',]

    groups = ["record_id"]

    df = utils.backfilling(df, cols, groups)

    return df


##### clean up first_symptoms fields #######


def clean_first_symptoms(df):
    df["first_symptoms_other"].replace(
        1.5634511000119108e16, int(10), inplace=True
    )  # type10_symptoms = ['Pain in bilateral legs', 1.5634511e+16]
    df["first_symptoms_other"].replace(
        2.6079004e07, int(11), inplace=True
    )  # type11_symptoms = ['Tremor', 2.6079004e+07]
    df["first_symptoms"].replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Filling forward AND backward first_symptoms (baseline variabel)
    bf_first_symptoms = ["first_symptoms", "first_symptoms_other"]
    groups = ["record_id"]
    df = utils.backfilling(df, bf_first_symptoms, groups, sort=False)

    def first_symptoms_arr(first_symptoms, first_symptoms_other):
        if not pd.isna(first_symptoms) and first_symptoms == "0":  # none
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if (
            not pd.isna(first_symptoms) and first_symptoms == "2"
        ):  # less spontaneous movements
            return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "3":  # muscle weakness
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if (
            not pd.isna(first_symptoms) and first_symptoms == "4"
        ):  # swalling difficulties
            return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if (
            not pd.isna(first_symptoms) and first_symptoms == "5"
        ):  # breathing difficulties
            return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        if (
            not pd.isna(first_symptoms) and first_symptoms == "6"
        ):  # Delayed gross motor skills
            return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if (
            not pd.isna(first_symptoms) and first_symptoms == "7"
        ):  # Spontaneous tongue movements
            return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "8":  # Limited mobility
            return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "9":  # Scoliosis
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "10":  # bilateral leg pain
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "11":  # tremor
            return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        if not pd.isna(first_symptoms) and first_symptoms == "888":  # Other
            return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "6, 8":  #
            return np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])
        if (
            not pd.isna(first_symptoms)
            and first_symptoms == "3, 888"
            and first_symptoms_other == int(10)
        ):  # update this function if needed
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        if (
            not pd.isna(first_symptoms)
            and first_symptoms == "3, 888"
            and first_symptoms_other == int(11)
        ):  # update this function if needed
            return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        if not pd.isna(first_symptoms) and first_symptoms == "3, 8":  #
            return np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "3, 6, 8":  #
            return np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "3, 6, 7":  #
            return np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "3, 6":  #
            return np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "3, 4, 6":  #
            return np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 6":  #
            return np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 4":  #
            return np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3, 8":  #
            return np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3, 7":  #
            return np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3, 6":  #
            return np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3, 5":  #
            return np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3, 4":  #
            return np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "2, 3":  #
            return np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        if not pd.isna(first_symptoms) and first_symptoms == "999":  # UNKNOWN
            return np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        else:
            return np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])  # 999 UNKNOWN

    def first_symptoms_sum(
        first_symptoms_arr,
    ):  # in python you can have a function within a function. In others you cannot.
        return np.sum((first_symptoms_arr))

    df["first_symptoms_arr"] = df.apply(
        lambda x: first_symptoms_arr(x.first_symptoms, x.first_symptoms_other), axis=1
    )
    df["first_symptoms_sum"] = df.apply(
        lambda x: first_symptoms_sum(x.first_symptoms_arr), axis=1
    )

    return df


######## clean up motor milestones fields ####################


# Then create a score that incorporates the position in the np.array as weight
def feat_motormiles(df):
    df["sit_wo_sup"].replace(1, int(0), inplace=True)
    df["sit_wo_sup"].replace(2, int(2), inplace=True)
    df["sit_wo_sup"].replace(3, int(1), inplace=True)
    df["sit_wo_sup"].replace(np.nan, int(-1), inplace=True)

    df["crawl_hands_knees"].replace(1, int(0), inplace=True)
    df["crawl_hands_knees"].replace(2, int(2), inplace=True)
    df["crawl_hands_knees"].replace(3, int(1), inplace=True)
    df["crawl_hands_knees"].replace(np.nan, int(-1), inplace=True)

    df["stand_wo_sup"].replace(1, int(0), inplace=True)
    df["stand_wo_sup"].replace(2, int(2), inplace=True)
    df["stand_wo_sup"].replace(3, int(1), inplace=True)
    df["stand_wo_sup"].replace(np.nan, int(-1), inplace=True)

    df["walk_wo_sup"].replace(1, int(0), inplace=True)
    df["walk_wo_sup"].replace(2, int(2), inplace=True)
    df["walk_wo_sup"].replace(3, int(1), inplace=True)
    df["walk_wo_sup"].replace(np.nan, int(-1), inplace=True)

    df["climb_stairs"].replace(1, int(0), inplace=True)
    df["climb_stairs"].replace(2, int(2), inplace=True)
    df["climb_stairs"].replace(3, int(1), inplace=True)
    df["climb_stairs"].replace(np.nan, int(-1), inplace=True)

    def motormiles_arr(
        sit_wo_sup, crawl_hands_knees, stand_wo_sup, walk_wo_sup, climb_stairs
    ):
        motormiles_array = np.array(
            [sit_wo_sup, crawl_hands_knees, stand_wo_sup, walk_wo_sup, climb_stairs]
        )
        return motormiles_array

    def motormiles_score(motormiles_array):
        motormiles_score = sum(
            3**n * motormiles_array[-n - 1] for n in range(len(motormiles_array))
        )
        return motormiles_score

    df["motormiles_arr"] = df.apply(
        lambda x: motormiles_arr(
            x.sit_wo_sup,
            x.crawl_hands_knees,
            x.stand_wo_sup,
            x.walk_wo_sup,
            x.climb_stairs,
        ),
        axis=1,
    )
    df["motormiles_score"] = df.apply(
        lambda x: motormiles_score(x.motormiles_arr), axis=1
    )

    return df


##### baseline ventilation score ####


def feat_baseline_ventilation(df):
    def base_ventilation_arr(baseline_ventilation, baseline_ventilation_type):
        if not pd.isna(baseline_ventilation) and baseline_ventilation == 0:  # none
            return np.array([0, 0, 0])
        elif (
            not pd.isna(baseline_ventilation)
            and baseline_ventilation == 1
            and baseline_ventilation_type == 1
        ):
            return np.array([1, 1, 0])
        elif (
            not pd.isna(baseline_ventilation)
            and baseline_ventilation == 1
            and baseline_ventilation_type == 2
        ):
            return np.array([0, 0, 1])
        elif (
            not pd.isna(baseline_ventilation)
            and baseline_ventilation == 1
            and baseline_ventilation_type == 888
        ):
            return np.array([1, 0, 0])
        elif (
            not pd.isna(baseline_ventilation)
            and baseline_ventilation == 1
            and baseline_ventilation_type == 999
        ):
            return np.array([1, 0, 0])
        else:
            return np.array([-1, -1, -1])

    def base_ventilation_score(base_ventilation_arr):
        if np.array_equal(base_ventilation_arr, np.array([0, 0, 0])):
            return 0
        elif base_ventilation_arr[0] == -1:
            return -1
        else:
            return sum(
                2**n * base_ventilation_arr[n] for n in range(len(base_ventilation_arr))
            )

    df["base_ventilation_arr"] = df.apply(
        lambda x: base_ventilation_arr(
            x.baseline_ventilation, x.baseline_ventilation_type
        ),
        axis=1,
    )
    df["base_ventilation_score"] = df.apply(
        lambda x: base_ventilation_score(x.base_ventilation_arr), axis=1
    )

    return df


######## ventilation score #######


def feat_ventilation(df):
    # replace function
    def ventilation_arr(ventilation_yn, ventilation_type):
        if not pd.isna(ventilation_yn) and ventilation_yn == 0:  # none
            return np.array([0, 0, 0])
        elif (
            not pd.isna(ventilation_yn)
            and ventilation_yn == 1
            and ventilation_type == 1
        ):
            return np.array([1, 1, 0])
        elif (
            not pd.isna(ventilation_yn)
            and ventilation_yn == 1
            and ventilation_type == 2
        ):
            return np.array([0, 0, 1])
        elif (
            not pd.isna(ventilation_yn)
            and ventilation_yn == 1
            and ventilation_type == 888
        ):
            return np.array([1, 0, 0])
        elif (
            not pd.isna(ventilation_yn)
            and ventilation_yn == 1
            and ventilation_type == 999
        ):
            return np.array([1, 0, 0])
        else:
            return np.array([-1, -1, -1])

    def ventilation_score(ventilation_arr):
        if np.array_equal(ventilation_arr, np.array([0, 0, 0])):
            return 0
        elif ventilation_arr[0] == -1:
            return -1
        else:
            return sum(2**n * ventilation_arr[n] for n in range(len(ventilation_arr)))

        # ventilation_score = (1/ventilation_freq)* np.array(ventilation_arr) **1/ventilator_time

    def ventilation_term(ventilation_arr, ventilation_freq, ventilator_time):
        if np.array_equal(ventilation_arr, np.array([0, 0, 0])):
            return 0
        elif ventilation_arr[0] == -1:
            return -1
        else:
            return (
                (1 / ventilation_freq)
                * sum(2**n * ventilation_arr[n] for n in range(len(ventilation_arr)))
                * (ventilator_time)
            )

        # ventilation_term = (1/ventilation_freq)* np.array(ventilation_arr) **1/ventilator_time

    df["ventilation_arr"] = df.apply(
        lambda x: ventilation_arr(x.ventilation_yn, x.ventilation_type), axis=1
    )
    df["ventilation_score"] = df.apply(
        lambda x: ventilation_score(x.ventilation_arr), axis=1
    )
    df["ventilation_term"] = df.apply(
        lambda x: ventilation_term(
            x.ventilation_arr, x.ventilation_freq, x.ventilator_time
        ),
        axis=1,
    )

    return df


################## contractures score ###################


def feat_contractures(df):
    def contractures_arr(contractures, contractures_loco):
        if not pd.isna(contractures) and contractures == 0:  # NONE - weight 0
            return np.array([0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "888"
        ):  # OTHER - weight 1
            return np.array([0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(contractures) and contractures == 1 and contractures_loco == "1"
        ):  # Hands - weight 2
            return np.array([0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(contractures) and contractures == 1 and contractures_loco == "2"
        ):  # Elbows - weight 3
            return np.array([0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(contractures) and contractures == 1 and contractures_loco == "5"
        ):  # FEET - Recode weighted as 4
            return np.array([0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(contractures) and contractures == 1 and contractures_loco == "3"
        ):  # HIPS - Recode weight as 5
            return np.array([0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(contractures) and contractures == 1 and contractures_loco == "4"
        ):  # KNEE - Recode weighted as 6
            return np.array([1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 2, 3, 4, 5, 888"
        ):
            return np.array([1, 1, 1, 1, 1, 1])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 2, 3, 4, 5"
        ):
            return np.array([1, 1, 1, 1, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 2, 3, 4"
        ):
            return np.array([1, 1, 0, 1, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 2, 3"
        ):
            return np.array([0, 1, 0, 1, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 2, 5"
        ):
            return np.array([0, 0, 1, 1, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 3, 4"
        ):
            return np.array([1, 1, 0, 0, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 3, 4, 5"
        ):
            return np.array([1, 1, 1, 0, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "1, 4"
        ):
            return np.array([1, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 3"
        ):
            return np.array([0, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 3, 4"
        ):
            return np.array([1, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 3, 4, 5"
        ):
            return np.array([1, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 3, 4, 888"
        ):
            return np.array([1, 1, 0, 1, 0, 1])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 4"
        ):
            return np.array([1, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 4, 5"
        ):
            return np.array([1, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 3, 4, 5"
        ):
            return np.array([1, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 4, 5, 888"
        ):
            return np.array([1, 0, 1, 1, 0, 1])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "2, 5"
        ):
            return np.array([0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "3, 4"
        ):
            return np.array([1, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "3, 4, 5"
        ):
            return np.array([1, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "3, 4, 5, 888"
        ):
            return np.array([1, 1, 1, 0, 0, 1])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "3, 5"
        ):
            return np.array([0, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "4, 999"
        ):  # same meaning as 4
            return np.array([1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(contractures)
            and contractures == 1
            and contractures_loco == "888, 999"
        ):  # same meaning as unknown other "888"
            return np.array([0, 0, 0, 0, 0, 1])
        else:
            return np.array([-1, -1, -1, -1, -1, -1])

    def contractures_score(contractures_arr, contractures_limit):
        if np.array_equal(contractures_arr, np.array([0, 0, 0, 0, 0, 0])):
            return 0
        elif pd.isna(contractures_limit) or contractures_arr[0] == -1:
            return -1
        else:
            return sum(
                2**n * contractures_arr[-n - 1] for n in range(len(contractures_arr))
            ) ** (contractures_limit / 3)

    # CAVE: the contractures_score was corrected because the ranking of the np.array is reverse ordered

    df["contractures_arr"] = df.apply(
        lambda x: contractures_arr(x.contractures, x.contractures_loco), axis=1
    )
    df["contractures_score"] = df.apply(
        lambda x: contractures_score(
            contractures_arr(x.contractures, x.contractures_loco), x.contractures_limit
        ),
        axis=1,
    )

    return df


# ################################ devices score ###############################


def feat_devices(df):
    # reassign for weighting
    df["device_type"].replace("2", "3", inplace=True)
    df["device_type"].replace("4", "5", inplace=True)
    df["device_type"].replace("1", "4", inplace=True)
    df["device_type"].replace("1, 2", "3, 4", inplace=True)
    df["device_type"].replace("2, 4", "3, 5", inplace=True)
    # df['device_type'].replace('888', '', inplace=True)
    df["device_type"].replace("1, 4", "4, 5", inplace=True)
    df["device_type"].replace("4, 888", "5, 888", inplace=True)
    df["device_type"].replace("3, 888", "6, 888", inplace=True)
    df["device_type"].replace("3", "6", inplace=True)
    df["device_type"].replace("1, 4, 888", "4, 5, 888", inplace=True)
    df["device_type"].replace("3, 4, 888", "5, 6, 888", inplace=True)
    df["device_type"].replace("1, 2, 4, 888", "3, 4, 5, 888", inplace=True)
    df["device_type"].replace("1, 2, 888", "3, 4, 888", inplace=True)
    df["device_type"].replace("1, 2, 4", "3, 4, 5", inplace=True)
    df["device_type"].replace("2, 4, 888", "3, 5, 888", inplace=True)
    df["device_type"].replace("2, 888", "3, 888", inplace=True)
    df["device_type"].replace("1, 888", "4, 888", inplace=True)
    df["device_type"].replace("1, 3, 4", "4, 5, 6", inplace=True)
    df["wheelchair_type"].replace(
        "1", "2", inplace=True
    )  # reassign active wheelchair to 2 (higher functional level)
    df["wheelchair_type"].replace(
        "2", "1", inplace=True
    )  # reassign power wheelchair to 1
    df["training_devices_other"].replace(
        "corset, complete body orthosis for standing, Therapiestuhl, Aktivrollstuhl, Autositz, rehabilitate-buggy",
        "2, 3, 4",
        inplace=True,
    )
    df["training_devices_other"].replace(
        "Rollator", "7", inplace=True
    )  # walking device
    df["training_devices_other"].replace("walker", "7", inplace=True)
    df["training_devices_other"].replace("rollator", "7", inplace=True)
    df["training_devices_other"].replace("deckenlifter", "0", inplace=True)
    df["training_devices_other"].replace("Motomed", "6", inplace=True)
    df["training_devices_other"].replace("NF-Walker", "", inplace=True)
    df["training_devices_other"].replace(
        "Rollator, Vibrationsplatte ähnlich Galileo", "5, 7", inplace=True
    )
    df["training_devices_other"].replace(
        "Rollstuhl mit Stehfunktion", "2, 4", inplace=True
    )
    df["training_devices_other"].replace(
        "Stehfunktion beim Rollstuhl", "2, 4", inplace=True
    )
    df["training_devices_other"].replace("therapeutic chair", "3", inplace=True)
    df["training_devices_other"].replace("Rehabuggy ", "3", inplace=True)
    df["training_devices_other"].replace("3 Wheelchairs", "1", inplace=True)
    df["training_devices_other"].replace("Korsett", "0", inplace=True)
    df["training_devices_other"].replace('Retrowalker "Malte"', "7", inplace=True)
    df["training_devices_other"].replace("Bewegungstrainer", "6", inplace=True)
    df["training_devices_other"].replace("Nachtlagerungskissen", "0", inplace=True)
    df["training_devices_other"].replace("Rehabuggy", "3", inplace=True)
    df["training_devices_other"].replace("Antidekubitusmatratze", "0", inplace=True)
    df["training_devices_other"].replace("Retrorollator ", "7", inplace=True)
    df["training_devices_other"].replace("retro walker", "7", inplace=True)
    df["training_devices_other"].replace("Gehwagen", "7", inplace=True)
    df["training_devices_other"].replace("Therapy seat", "3", inplace=True)
    df["training_devices_other"].replace("Neopren-Stützanzug/-Weste", "0", inplace=True)
    df["training_devices_other"].replace("Retrorollator", "7", inplace=True)
    df["training_devices_other"].replace("posterior Gehwagen", "7", inplace=True)
    df["training_devices_other"].replace(
        "retro-walker, standing frame with wheels", "4, 7", inplace=True
    )
    df["training_devices_other"].replace("motomed", "6", inplace=True)
    #    df['training_devices_other'].replace(r'^\s+$', pd.NA, inplace=True, regex=True)
    df["training_devices_other"].replace(
        "", np.NaN, inplace=True
    )  # breaks the code, cannot run a second time

    def devices_arr(
        training_devices, device_type, training_devices_other, wheelchair_type
    ):
        # reassigned!
        if (
            not pd.isna(training_devices)
            and training_devices == 0.00
            and (pd.isna(wheelchair_type) or wheelchair_type == 0.00)
        ):  # NONE
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 0.00
            and wheelchair_type == 1.00
        ):  # reassigns non-devices to the value 0 (none)
            return np.array([0, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 0.00
            and wheelchair_type == 2.00
        ):  # reassigns non-devices to the value 0 (none)
            return np.array([0, 0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "0"
            and wheelchair_type == 1.00
        ):  # reassigns non-devices to the value 0 (none)
            return np.array([0, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "0"
            and wheelchair_type == 2.00
        ):  # reassigns non-devices to the value 0 (none)
            return np.array([0, 0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "0"
        ):  # reassigns non-devices to the value 0 (none)
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "1"
            and wheelchair_type == 2.00
        ):
            return np.array([1, 0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "2, 3, 4"
        ):
            return np.array([0, 0, 1, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "2, 4"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "2, 4"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "2, 4"
        ):
            return np.array([0, 0, 1, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "3"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "3"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "3"
        ):
            return np.array([0, 0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "6"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "6"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "6"
        ):
            return np.array([0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "7"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "7"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
            and training_devices_other == "7"
        ):
            return np.array([0, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "888"
        ):
            return np.array([1, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3"
        ):  # assistive seating device
            return np.array([0, 0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4"
            and wheelchair_type == 1.00
        ):  # assistive standing device
            return np.array([0, 1, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4"
            and wheelchair_type == 2.00
        ):  # assistive standing device
            return np.array([0, 0, 1, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4"
        ):  # assistive standing device
            return np.array([0, 0, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5"
            and wheelchair_type == 1.00
        ):  # training with vibrations
            return np.array([0, 1, 0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5"
            and wheelchair_type == 2.00
        ):  # training with vibrations
            return np.array([0, 0, 1, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5"
        ):  # training with vibrations
            return np.array([0, 0, 0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6"
            and wheelchair_type == 1
        ):  # active training device
            return np.array([0, 1, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6"
            and wheelchair_type == 2
        ):  # active training device
            return np.array([0, 0, 1, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6"
        ):  # active training device
            return np.array([0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "7"
            and wheelchair_type == 1
        ):  # assistive walking device
            return np.array([0, 1, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "7"
            and wheelchair_type == 2
        ):  # assistive walking device
            return np.array([0, 0, 1, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "7"
        ):  # assistive walking device
            return np.array([0, 0, 0, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4"
        ):
            return np.array([0, 0, 0, 1, 1, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 888"
            and training_devices_other == "7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 888"
            and training_devices_other == "7"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 888"
            and training_devices_other == "7"
        ):
            return np.array([0, 0, 0, 1, 1, 0, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
        ):
            return np.array([0, 0, 0, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5"
        ):
            return np.array([0, 0, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "0"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "0"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "0"
        ):
            return np.array([0, 0, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "4, 7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "4, 7"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 4, 5, 888"
            and training_devices_other == "4, 7"
        ):
            return np.array([0, 0, 0, 1, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5"
        ):
            return np.array([0, 0, 0, 1, 0, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5, 888"
            and training_devices_other == "7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 0, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "3, 5, 888"
            and training_devices_other == "7"
        ):
            return np.array([0, 0, 0, 1, 0, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5"
        ):
            return np.array([0, 0, 0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 6"
            and wheelchair_type == 1.00
        ):
            return np.array([0, 1, 0, 0, 1, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 6"
            and wheelchair_type == 2.00
        ):
            return np.array([0, 0, 1, 0, 1, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 6"
        ):
            return np.array([0, 0, 0, 0, 1, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 6"
            and pd.isna(training_devices_other)
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 1, 1, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 6"
            and pd.isna(training_devices_other)
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 0, 1, 1, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 6"
            and pd.isna(training_devices_other)
        ):
            return np.array([0, 0, 0, 0, 1, 1, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and pd.isna(training_devices_other)
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and pd.isna(training_devices_other)
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and pd.isna(training_devices_other)
        ):
            return np.array([0, 0, 0, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "3"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "3"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "3"
        ):
            return np.array([0, 0, 0, 1, 1, 1, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "7"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 0, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "4, 5, 888"
            and training_devices_other == "7"
        ):
            return np.array([0, 0, 0, 0, 1, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 6, 888"
            and training_devices_other == "3"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 1, 0, 1, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 6, 888"
            and training_devices_other == "3"
        ):
            return np.array([0, 0, 0, 1, 0, 1, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 6, 888"
            and training_devices_other == "7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 0, 1, 1, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 6, 888"
            and training_devices_other == "7"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 0, 0, 1, 1, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 6, 888"
            and training_devices_other == "7"
        ):
            return np.array([0, 0, 0, 0, 0, 1, 1, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 888"
            and training_devices_other == "6"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 888"
            and training_devices_other == "6"
        ):
            return np.array([0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 888"
            and training_devices_other == "7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 0, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "5, 888"
            and training_devices_other == "7"
            and wheelchair_type == 2
        ):
            return np.array([0, 0, 1, 0, 0, 1, 0, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6, 888"
            and training_devices_other == "0"
        ):
            return np.array([0, 0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6, 888"
            and training_devices_other == "3"
        ):
            return np.array([0, 0, 0, 1, 0, 0, 1, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6, 888"
            and training_devices_other == "5, 7"
            and wheelchair_type == 1
        ):
            return np.array([0, 1, 0, 0, 0, 1, 1, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "6, 888"
            and training_devices_other == "5, 7"
        ):
            return np.array([0, 0, 0, 0, 0, 1, 1, 1])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "999"
            and wheelchair_type == 1.00
        ):
            return np.array([1, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "999"
            and wheelchair_type == 2.00
        ):
            return np.array([1, 0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and device_type == "999"
        ):
            return np.array([1, 0, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and (device_type == "1" or wheelchair_type == 1)
        ):  # power wheelchair
            return np.array([0, 1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(training_devices)
            and training_devices == 1.0
            and (device_type == "2" or wheelchair_type == 2)
        ):  # active wheelchair
            return np.array([0, 0, 1, 0, 0, 0, 0, 0])
        elif (
            pd.isna(training_devices)
            and pd.isna(training_devices_other)
            and wheelchair_type == 1.00
        ):  # reevaluate this vs. reassigning non-devices to the value 0 (none)
            return np.array([-1, 1, 0, -1, -1, -1, -1, -1])
        elif (
            pd.isna(training_devices)
            and pd.isna(training_devices_other)
            and wheelchair_type == 2.00
        ):  # reevaluate this vs. reassigning non-devices to the value 0 (none)
            return np.array([-1, 0, 1, -1, -1, -1, -1, -1])
        if (
            not pd.isna(training_devices)
            and training_devices == 0.00
            and pd.isna(wheelchair_type)
        ):  # NONE
            return np.array([0, 0, 0, 0, 0, 0, 0, 0])
        else:
            return np.array([-1, -1, -1, -1, -1, -1, -1, -1])

    def devices_score(devices_arr):
        if np.array_equal(devices_arr, np.array([0, 0, 0, 0, 0, 0, 0, 0])):
            return 0
        elif np.array_equal(devices_arr, np.array([-1, -1, -1, -1, -1, -1, -1, -1])):
            return -1
        else:
            return sum(2**n * devices_arr[n] for n in range(len(devices_arr)))

    df["devices_arr"] = df.apply(
        lambda x: devices_arr(
            x.training_devices,
            x.device_type,
            x.training_devices_other,
            x.wheelchair_type,
        ),
        axis=1,
    )
    df["devices_score"] = df.apply(lambda x: devices_score(x.devices_arr), axis=1)

    return df


################## interventions score ########################


def feat_interventions(df):
    df["interventions_other"].replace("swimming ", "5", inplace=True)
    df["interventions_other"].replace("water gymnastics", "5", inplace=True)
    df["interventions_other"].replace("KG im Wasser", "5", inplace=True)
    df["interventions_other"].replace("horseback riding ", "6", inplace=True)
    df["interventions_other"].replace("riding therapy ", "6", inplace=True)
    df["interventions_other"].replace(
        "riding therapy every 4 weeks ", "6", inplace=True
    )
    df["interventions_other"].replace("Reittherapie", "6", inplace=True)
    df["interventions_other"].replace("therapeutisches Reiten", "6", inplace=True)
    df["interventions_other"].replace("early intervention", "7", inplace=True)
    df["interventions_other"].replace("Frühförderung", "7", inplace=True)
    df["interventions_other"].replace("Frühförderung/ ET", "7", inplace=True)
    df["interventions_other"].replace(
        "swimming, horseback riding", "5, 6", inplace=True
    )

    def interventions_arr(interventions_type, interventions_other):
        if not pd.isna(interventions_type) and interventions_type == "0":
            return np.array([0, 0, 0, 0, 0, 0, 0])
        elif not pd.isna(interventions_type) and interventions_type == "888":
            return np.array([1, 0, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(interventions_type) and interventions_type == "2"
        ):  # Physiotherapy
            return np.array([0, 1, 0, 0, 0, 0, 0])
        elif (
            not pd.isna(interventions_type) and interventions_type == "3"
        ):  # feeding/speech
            return np.array([0, 0, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(interventions_type) and interventions_type == "4"
        ):  # occupational therapy
            return np.array([0, 0, 0, 1, 0, 0, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "888"
            and interventions_other == "5"
        ):  # swimming
            return np.array([0, 0, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "888"
            and interventions_other == "6"
        ):  # riding therapy
            return np.array([0, 0, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "888"
            and interventions_other == "7"
        ):  # other early intervention
            return np.array([0, 0, 0, 0, 0, 0, 1])
        elif not pd.isna(interventions_type) and interventions_type == "2, 4":
            return np.array([0, 1, 0, 1, 0, 0, 0])
        elif not pd.isna(interventions_type) and interventions_type == "2, 3, 4":
            return np.array([0, 1, 1, 1, 0, 0, 0])
        elif not pd.isna(interventions_type) and interventions_type == "2, 3":
            return np.array([0, 1, 1, 0, 0, 0, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 888"
            and interventions_other == "5"
        ):
            return np.array([0, 1, 0, 0, 1, 0, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 888"
            and interventions_other == "6"
        ):
            return np.array([0, 1, 0, 0, 0, 1, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 888"
            and interventions_other == "7"
        ):
            return np.array([0, 1, 0, 0, 0, 0, 1])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 4, 888"
            and interventions_other == "5"
        ):
            return np.array([0, 1, 0, 1, 1, 0, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 4, 888"
            and interventions_other == "6"
        ):
            return np.array([0, 1, 0, 1, 0, 1, 0])
        elif (
            not pd.isna(interventions_type)
            and interventions_type == "2, 3, 888"
            and interventions_other == "7"
        ):
            return np.array([0, 1, 1, 0, 0, 0, 1])
        else:
            return np.array([-1, -1, -1, -1, -1, -1, -1])

    def interventions_sum(
        interventions_arr,
    ):  # in python you can have a function within a function. In others you cannot.
        return np.sum((interventions_arr))

    df["interventions_arr"] = df.apply(
        lambda x: interventions_arr(x.interventions_type, x.interventions_other), axis=1
    )
    df["interventions_sum"] = df.apply(
        lambda x: interventions_sum(x.interventions_arr), axis=1
    )

    return df
