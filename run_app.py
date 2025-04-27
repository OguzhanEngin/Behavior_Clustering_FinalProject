import pandas as pd
import streamlit as st

from src.feature_extractor import FeatureExtractor
from src.clusterer import Clusterer
from src.reporter import Reporter
from src.utils import file_selector, bhv_imp_slider, input_check


def main():
    st.set_page_config(page_title="Behavior Clustering Server", layout="centered")

    st.title("Behavior Clustering")
    st.header("Input Data Parameters")

    input_data_dir = file_selector(
        folder_path=".",
        select_label=r"$\textsf{\Large Select the Input Directory}$",
        not_contains=".",
    )
    if input_data_dir is None:
        st.warning("Enter the problem directory to continue!")
        st.stop()

    input_data_path = file_selector(
        folder_path=input_data_dir,
        select_label=r"$\textsf{\Large Select the Input File}$",
        contains=".csv",
    )

    st.header("Behavior Importance Parameters")
    lnr_importance = bhv_imp_slider(behavior_label=r"$\textsf{\Large Linear}$")
    osc_importance = bhv_imp_slider(behavior_label=r"$\textsf{\Large Oscillation}$")
    gad_importance = bhv_imp_slider(
        behavior_label=r"$\textsf{\Large Growth and Decline}$"
    )

    exp_importance = bhv_imp_slider(
        behavior_label=r"$\textsf{\Large Positive Exponential Growth}$"
    )
    nex_importance = bhv_imp_slider(
        behavior_label=r"$\textsf{\Large Negative Exponential Growth}$"
    )
    ssh_importance = bhv_imp_slider(behavior_label=r"$\textsf{\Large S-shape Growth}$")
    bhv_imp_dict = {
        "lnr": lnr_importance,
        "osc": osc_importance,
        "gad": gad_importance,
        "log": nex_importance,
        "exp": exp_importance,
        "nexp": nex_importance,
        "ssh": ssh_importance,
    }

    st.header("Output Parameters")
    output_dir = st.text_input(r"$\textsf{\Large Enter the output directory name}$")
    if output_dir == "":
        st.warning("Enter an output directory name to continue!")
        st.stop()

    run_clustering = st.button(label="Start Clustering!", type="primary")
    if run_clustering:
        input_df = pd.read_csv(input_data_path)
        input_check(input_df=input_df)

        feature_extractor = FeatureExtractor()
        trans_input_df = feature_extractor.extract_features(
            input_df=input_df, bhv_imp_dict=bhv_imp_dict
        )

        clusterer = Clusterer()
        clusterer.find_n_clusters(bhv_imp_dict)
        cluster_result_df = clusterer.run_clustering(trans_input_df=trans_input_df)

        reporting_df = input_df.copy()
        reporting_df["Cluster"] = cluster_result_df["Cluster"]
        reporter = Reporter()
        reporter.report_results(reporting_df=reporting_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
