import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st


class Reporter:
    def __init__(self) -> None:
        self.output_dir = "Outputs"
        self.plot_out_dir = "Plots"

    def report_results(self, reporting_df: pd.DataFrame, output_dir: str):
        main_out_dir = os.path.join(output_dir, self.output_dir)
        plot_out_dir = os.path.join(main_out_dir, self.plot_out_dir)
        os.makedirs(plot_out_dir, exist_ok=True)

        self._calculate_performance(reporting_df=reporting_df)

        self._visuzalize_clusters(reporting_df=reporting_df, plot_out_dir=plot_out_dir)

    def _visuzalize_clusters(self, reporting_df: pd.DataFrame, plot_out_dir: str):
        for cluster in np.sort(reporting_df["Cluster"].unique()):
            img_out_path = os.path.join(plot_out_dir, f"cluster_{cluster}.png")
            cluster_mask = reporting_df["Cluster"] == cluster
            cluster_df = reporting_df[cluster_mask].drop(["Label", "Cluster"], axis=1)

            _, ax = plt.subplots(figsize=(11, 8))
            ax.set_title(f"Cluster {cluster}")
            ax.plot(cluster_df.T)
            tick_spacing = 20
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Value")
            plt.savefig(img_out_path, dpi=300)
            st.image(img_out_path)

    def _calculate_performance(self, reporting_df: pd.DataFrame):
        if "Label" not in reporting_df.columns:
            return None
        else:
            adj_rand_score = adjusted_rand_score(reporting_df["Label"], reporting_df["Cluster"])
            adj_rand_score = round(adj_rand_score, 3)
            st.subheader(f"Adjusted Rand Score: {adj_rand_score}")

            adj_mtl_inf_score = adjusted_mutual_info_score(reporting_df["Label"], reporting_df["Cluster"])
            adj_mtl_inf_score = round(adj_mtl_inf_score, 3)
            st.subheader(f"Adjusted Mutual Information Score: {adj_mtl_inf_score}")
