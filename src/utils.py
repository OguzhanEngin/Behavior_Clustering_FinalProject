import os
import numpy as np
import pandas as pd
import streamlit as st


def file_selector(
    folder_path: str,
    select_label: str,
    not_contains: str | None = None,
    contains: str | None = None,
):
    filenames = os.listdir(folder_path)
    if not_contains is not None:
        selected_file_arr = []
        for file in filenames:
            if not_contains in file:
                continue
            else:
                selected_file_arr.append(file)
    elif contains is not None:
        selected_file_arr = []
        for file in filenames:
            if not contains in file:
                continue
            else:
                selected_file_arr.append(file)

    else:
        selected_file_arr = filenames

    selected_filename = st.selectbox(label=select_label, options=selected_file_arr)
    return os.path.join(folder_path, selected_filename)


def bhv_imp_slider(behavior_label: str):
    behavior_importance = st.slider(
        label=behavior_label, min_value=0, max_value=10, value=1
    )
    return behavior_importance


def input_check(input_df: pd.DataFrame):
    assert (
        len(input_df) > 1
    ), "Need more than one simulation output to start clustering!"

    input_col_arr = input_df.columns
    if "Label" in input_col_arr:
        input_col_arr = input_col_arr.drop("Label")

    try:
        input_col_arr = input_col_arr.astype(float)
    except:
        raise ValueError(
            "Data contains a column which is not a number!, other than Label column"
        )

    assert (
        input_col_arr != np.sort(input_col_arr)
    ).sum() == 0, "Input data frequency columns are not sorted!"
