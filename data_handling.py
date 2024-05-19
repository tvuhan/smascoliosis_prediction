"""Data handling"""

__author__ = [
    "Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)",
    "Rodrigo Bermudez Schettino (Max Planck Institute for Human Development)",
]
__maintainer__ = ["Tu-Lan Vu-Han (Charité - Universitätsmedizin Berlin)"]

from pathlib import Path

import numpy as np
import pandas as pd

import utils


# loads main dataframe
def load_dataframe(path_to_file):
    clean_path = Path(path_to_file)
    df = pd.read_csv(clean_path, delimiter=";")
    return df


# loads labels
def load_dataframe_labels(df, path_to_file):
    clean_path = Path(path_to_file)
    spine = pd.read_csv(clean_path)

    df = utils.attachDataInnerJoinDateMatch(
        df,
        spine,
        "record_id",
        "ID",
        "scoliosis_yn",
        "date_assessment",
        "date_scoli_detect",
    )
    return df


# returns the visits in the data frame
def df_slice_visits(df):
    dfr = df.loc[
        (df["redcap_event_name"] == "first_visit_arm_1")
        | (df["redcap_event_name"] == "visits_arm_1")
    ]
    return dfr


# returns the FIRST line of every record_id in a data frame
def df_slice_firstline(df):
    df_first = df.groupby("record_id").first()
    return df_first


# returns the LAST line of every record_id in a data frame
def df_slice_lastline(df):
    df_last = df.groupby("record_id").last()
    return df_last


# returns and fixes the therapy slice in the data frame
def df_slice_therapy(df):
    # "Fixing" Therapies. (NOTE: highly questionable. 2/1/2023 17:59) # NOTE: fixed 22.04.2023 see code below

    # fixing nus
    dfr = df.loc[df["redcap_event_name"] == "nus_arm_1"]
    cols = [
        "date_start_nus",
        "age_start_therapy",
        "nus_presymp",
        "date_nus",
        "datediff_nus",
        "nus_dosetype",
        "nus_regular",
        "nus_appl",
        "nus_complications",
        "therapy_alt",
    ]
    groups = ["record_id"]
    df_nus = dfr.copy()
    utils.backfilling(
        df_nus, cols, groups, direction="f", sort=True, refDate="redcap_repeat_instance"
    )

    # fixing ona
    dfr = df.loc[df["redcap_event_name"] == "ona_arm_1"]
    cols = [
        "date_start_ona",
        "date_ona",
        "age_ona",
        "ona_repeat_yn",
        "ona_presymp",
        "ona_aav9ab",
        "ona_weight",
        "ona_dose",
        "ona_route",
        "prednisolon_prophyl",
        "prednisolon_prolong",
        "adverse_ona",
        "ona_complications",
        "ona_complications_2",
    ]
    groups = ["record_id"]
    df_ona = dfr.copy()
    utils.backfilling(
        df_ona, cols, groups, direction="f", sort=True, refDate="redcap_repeat_instance"
    )

    # fixing ris
    dfr = df.loc[df["redcap_event_name"] == "ris_arm_1"]
    cols = ["date_start_ris", "ris_time", "ris_regular", "ris_complications", "age_ris"]
    groups = ["record_id"]
    df_ris = dfr.copy()
    utils.backfilling(
        df_ris, cols, groups, direction="f", sort=True, refDate="redcap_repeat_instance"
    )

    # Note: here NUS, ONA and RIS are concatenated in that order, which is why
    df_therapy = pd.concat([df_nus, df_ona, df_ris])

    # generate therapy vectors
    # this is just a util function for the apply
    def integrateTherapyOverTime(df, row):
        eventToDateMap = {
            "nus_arm_1": "date_nus",
            "ona_arm_1": "date_start_ona",
            "ris_arm_1": "date_start_ris",
        }
        eventToCompare = {
            "nus_arm_1": "date_start_nus",
            "ona_arm_1": "date_start_ona",
            "ris_arm_1": "date_start_ris",
        }
        eventToIndicator = {"nus_arm_1": 0, "ona_arm_1": 2, "ris_arm_1": 1}

        record_id = row.record_id
        therapy = [-1, -1, -1]

        # we check always against the event names to make sure we can run over the whole dataframe
        if row.redcap_event_name in eventToDateMap.keys():
            therapy = [0, 0, 0]
            matchDate = row[eventToDateMap[row.redcap_event_name]]

        if not pd.isna(matchDate) and row.redcap_event_name in eventToDateMap.keys():
            therapy[eventToIndicator[row.redcap_event_name]] = 1

            for index, search_row in df.iterrows():
                if (
                    search_row.record_id == record_id
                    and search_row.redcap_event_name in eventToCompare.keys()
                ):  # first match the record id
                    #                if record_id == 5 and row.redcap_event_name == 'nus_arm_1' and search_row.redcap_event_name == 'ris_arm_1':
                    #                    print(row.redcap_event_name, matchDate, search_row.redcap_event_name, search_row[eventToCompare[search_row.redcap_event_name]], eventToIndicator[search_row.redcap_event_name], matchDate >= search_row[eventToCompare[search_row.redcap_event_name]])
                    if (
                        not pd.isna(
                            search_row[eventToCompare[search_row.redcap_event_name]]
                        )
                        and matchDate
                        >= search_row[eventToCompare[search_row.redcap_event_name]]
                    ):
                        therapy[eventToIndicator[search_row.redcap_event_name]] = 1

        else:
            therapy = [-1, -1, -1]
        return therapy

    def therapy_score(therapy_arr):
        if np.array_equal(therapy_arr, np.array([0, 0, 0])):
            return 0
        elif min(therapy_arr) == -1:
            return -1
        else:
            return sum(2**n * therapy_arr[n] for n in range(len(therapy_arr)))

    df_therapy["therapy_arr"] = df_therapy.apply(
        lambda x: integrateTherapyOverTime(df_therapy, x), axis=1
    )
    df_therapy["therapy_score"] = df_therapy.apply(
        lambda x: therapy_score(x.therapy_arr), axis=1
    )

    # add assessment dates if missing
    def date_assessment_simulation(row):
        eventToDateMap = {
            "nus_arm_1": "date_nus",
            "ona_arm_1": "date_start_ona",
            "ris_arm_1": "date_start_ris",
        }
        return row[eventToDateMap[row.redcap_event_name]]

    df_therapy["date_assessment"] = df_therapy.apply(
        lambda x: date_assessment_simulation(x), axis=1
    )

    return df_therapy


# (1) create new date_column that contains all therapy dates (nus, ona, ris) in one column:
def fix_TherapyDates(df_therapy):
    def therapy_dates(date_nus, date_ona, date_start_ris):
        if not pd.isna(date_nus):
            return date_nus
        elif not pd.isna(date_ona):
            return date_ona
        elif not pd.isna(date_start_ris):
            return date_start_ris
        else:
            return np.nan

    df_therapy["therapy_dates"] = df_therapy.apply(
        lambda x: therapy_dates(x.date_nus, x.date_ona, x.date_start_ris), axis=1
    )
    #
    return df_therapy


# (2) create a new data df_therapy data frame that sorts each therapy by dateField 'therapy_dates' of each patient
def get_df_therapy_by_patient(df_therapy):
    df_therapy["therapy_dates"] = pd.to_datetime(
        df_therapy["therapy_dates"], format="%Y-%m-%d"
    )

    patients = np.sort(df_therapy["record_id"].unique())
    df_therapy_by_patient = pd.DataFrame(columns=df_therapy.columns)
    df_therapy_by_patient

    for record_id in patients:
        # group df_therapy by patient
        df_therapy_by_record_id = df_therapy.groupby("record_id").get_group(record_id)
        # sort therapy rows of a patient by date
        df_therapy_by_record_id = df_therapy_by_record_id.sort_values(
            by="therapy_dates", ascending=True, na_position="first"
        )
        # concatenate patient to the main data frame and call the nes data frame df_therapy_by_patient
        df_therapy_by_patient = pd.concat(
            [df_therapy_by_patient, df_therapy_by_record_id]
        )

    return df_therapy_by_patient


# (3) create a new age_column that contains age_therapy (nus, ona, ris) in one column:
def fix_TherapyAge(df):
    def therapy_age(age_start_therapy, age_ona, age_ris):
        if not pd.isna(age_start_therapy):
            return age_start_therapy
        elif not pd.isna(age_ona):
            return age_ona
        elif not pd.isna(age_ris):
            return age_ris
        else:
            return np.nan

    df["therapy_age"] = df.apply(
        lambda x: therapy_age(x.age_start_therapy, x.age_ona, x.age_ris), axis=1
    )

    return df


# THIS FUNCTION GROUPS A DATA FRAME BY RECORD ID and creates a new data frame with the most recent entry for each patient
def get_df_therapy_last(df_therapy, groupby_column="record_id"):
    row_indices = []
    patients = np.sort(df_therapy["record_id"].unique())

    for record_id in patients:
        df_therapy_by_record_id = df_therapy.groupby(groupby_column).get_group(
            record_id
        )
        row = df_therapy_by_record_id.tail(1)

        row_indices.append(row.index.to_list()[0])

    if groupby_column == "record_id":
        assert len(df_therapy[groupby_column].unique()) == len(row_indices)

    df_therapy_last_by_record_id = df_therapy.copy()
    df_therapy_last = df_therapy_last_by_record_id.loc[row_indices]

    return df_therapy_last


def get_first_valid_value_from_column(df, col_name):
    # Get first valid index from column with col_name in dataframe df and store in variable ix
    ix = df[col_name].first_valid_index()

    # Get value at index ix
    first_valid_value = df[col_name].at[ix]

    return first_valid_value


### BREAK OUT PATIENT SPECIFIC SUBFRAMES FOR PLOTTING PATIENT SPECIFIC TIME FRAME DATA ###
def get_subframe_spine_pat(df_spine, patient):
    subframe_spine_pat = df_spine.loc[(df_spine["record_id"] == patient)]
    return subframe_spine_pat


def get_subframe_scolisurg_pat(df_scolisurg, patient):
    subframe_scolisurg_pat = df_scolisurg.loc[(df_scolisurg["record_id"] == patient)]
    return subframe_scolisurg_pat


def get_subframe_therapy_pat(df_therapy, patient):
    subframe_therapy_pat = df_therapy.loc[(df_therapy["record_id"] == patient)]
    return subframe_therapy_pat


##### merge two data frames on a common field and sort by date #####
def get_df_merged_by_patient(
    df_1_before, df_2_before, dateField_1, dateField_2, commonField
):
    # convert date fields in the data frames to dateTime
    df_1 = utils.convert_dateTime(df_1_before)
    df_2 = utils.convert_dateTime(df_2_before)

    df_1 = df_1[df_1[dateField_1].notnull()]
    df_2 = df_2[df_2[dateField_2].notnull()]

    patients = np.sort(df_1[commonField].unique())

    # loop through every patient and create a mini data frame
    for record_id in patients:
        df_1_small = df_1[df_1[commonField] == record_id]
        df_2_small = df_2[df_2[commonField] == record_id]

        df_1_small = df_1_small.sort_values(dateField_1)
        df_2_small = df_2_small.sort_values(dateField_2)

        temp = pd.merge_asof(
            df_1_small, df_2_small, on=dateField_1, suffixes=("", "df2")
        )

        # sort therapy rows of a patient by date before concatenating
        df_merged_by_record_id = temp.sort_values(by=dateField_1, ascending=True)

        # concatenate patient to the main data frame and call the new data frame df_merged_by_patient
        if record_id == 1:
            df_merged_by_patient = df_merged_by_record_id.copy(deep=True)
        else:
            df_merged_by_patient = pd.concat(
                [df_merged_by_patient, df_merged_by_record_id]
            )

    return df_merged_by_patient.reset_index()


def get_df_merged_by_patient_2(
    df_1_before, df_2_before, dateField_1, dateField_2, commonField
):
    # convert date fields in the data frames to dateTime
    df_1 = utils.convert_dateTime(df_1_before)
    df_2 = utils.convert_dateTime(df_2_before)

    df_1 = df_1[df_1[dateField_1].notnull()]
    df_2 = df_2[df_2[dateField_2].notnull()]

    patients = np.sort(df_1[commonField].unique())

    # loop through every patient and create a mini data frame
    for record_id in patients:
        df_1_small = df_1[df_1[commonField] == record_id]
        df_2_small = df_2[df_2[commonField] == record_id]

        df_1_small = df_1_small.sort_values(dateField_1)
        df_2_small = df_2_small.sort_values(dateField_2)

        temp = pd.merge_asof(
            df_1_small,
            df_2_small,
            left_on=dateField_1,
            right_on=dateField_2,
            suffixes=("", "df2"),
        )

        # sort therapy rows of a patient by date before concatenating
        df_merged_by_record_id = temp.sort_values(by=dateField_1, ascending=True)

        # concatenate patient to the main data frame and call the new data frame df_merged_by_patient
        if record_id == 1:
            df_merged_by_patient = df_merged_by_record_id.copy(deep=True)
        else:
            df_merged_by_patient = pd.concat(
                [df_merged_by_patient, df_merged_by_record_id]
            )

    return df_merged_by_patient.reset_index()


to_drop_def = [
    "smartcare_id",  # REDCap ID sufficient for analysis 'num_id',
    "form1_demographic_complete",
    "form3_genetic_complete",
    "baseline_clinicaltrial",
    "baseline_clinicaltrial_date",
    "form4_baseline_history_complete",
    "form5_smavisit_complete",
    "form6_physical_exam_complete",
    "form7_nusinersen_complete",
    "form7_onasemnogen_complete",
    "form7_risdiplam_complete",
    "form8_spine_xray_complete",
    "neurophysiology_yn",
    "date_neurophys",
    "motornerve_type___0",
    "motornerve_type___2",
    "motornerve_type___3",
    "motornerve_type___4",
    "motornerve_type___999",
    "motornerve_velo",
    "fvc_percentage",
    "fvc_litres",
    "peakcough_flow",
    "vaccination_yn",
    "vaccination_type___1",
    "vaccination_type___2",
    "vaccination_type___3",
    "vaccination_type___888",
    "vaccination_type___999",
    "fwave_latency_type___0",
    "fwave_latency_type___2",
    "fwave_latency_type___3",
    "fwave_latency_type___4",
    "fwave_latency_type___999",
    "fwave_latency_ms",
    "cmap_type___0",
    "cmap_type___2",
    "cmap_type___3",
    "cmap_type___4",
    "cmap_type___999",
    "cmap_mv",
    "confirm_therapy_alt",
    "date_xray",
    "xray_ap_quality",
    "xray_position",
    "scoliosis_surgery_yn",
    "scoliosis_surgery_yn",
    "scoliosis_yn",
    "ap_cobb_maj",
    "xray_apcobb_body_1",
    "scoliosis_majcurve",
    "scoliosis_curvetype",
    "scoliosis_lumbarmod",
    "scoliosis_maj_convex",
    "scoliosis_rva",
    "scoliosis_rva",
    "ap_cobb_min",
    "xray_apcobb_body_2",
    "scoliosis_secondcurve",
    "scoliosis_second_convex",
    "pelvic_tilt",
    "pelvic_tilt_angle",
    "xray_sag_yn",
    "xray_sag_quality",
    "sag_t_cobb",
    "hyperkyphosis_yn",
    "sag_l_cobb",
    "hyperlordosis_yn",
    "ssl",
    "spine_complications",
    "spine_complications_type",
    "complication_1",
    "complication_2",
    "complication_3",
    "pjk_quantification",
    "xray_apcobb_body_1___1",
    "scoliosis_mincurve_yn",
    "xray_apcobb_body_2___9",
]
