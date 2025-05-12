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
    if selected_filename is None:
        st.warning("Select the input data to continue!")
        st.stop()
    return os.path.join(folder_path, selected_filename)


def image_question(caption: str, question: str, img_path: str):
    st.image(image=img_path)
    img_answer = importance_checkbox(label=question, caption=caption)
    return img_answer


def importance_checkbox(label: str, caption: str = ""):
    bhv_mode_res = st.checkbox(label=label, value=False, key=label + caption)
    return bhv_mode_res


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
