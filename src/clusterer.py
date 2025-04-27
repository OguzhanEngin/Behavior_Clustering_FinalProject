import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


class Clusterer:
    def __init__(self) -> None:
        self.n_clusters = None
        self.linkage = None

    def find_n_clusters(self, bhv_imp_dict: dict):
        n_clusters = -1
        for value in bhv_imp_dict.values():
            if value > 0:
                n_clusters += 1
        self.n_clusters = n_clusters

    def run_clustering(self, trans_input_df: pd.DataFrame):
        result_df = trans_input_df.copy()
        result_df["Cluster"] = None

        feature_col_arr = trans_input_df.columns.values
        result_df = self._clustering_step_1(
            result_df=result_df, feature_col_arr=feature_col_arr
        )

        result_df = self._clustering_step_2(result_df=result_df)

        return result_df

    def _clustering_step_1(self, result_df: pd.DataFrame, feature_col_arr: np.array):
        temp_result_df = result_df.copy()

        if "Linear" in feature_col_arr:
            lnr_cluster_idx = temp_result_df[temp_result_df["Linear"] >= 0.999].index
            temp_result_df.loc[lnr_cluster_idx, "Cluster"] = -2

        if "Oscillation" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            osc_mask = (temp_result_df["Oscillation"] >= 0.6) & next_cluster_mask
            osc_cluster_idx = temp_result_df[osc_mask].index
            temp_result_df.loc[osc_cluster_idx, "Cluster"] = -1

        if "Growth_Decline" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            gad_mask = (temp_result_df["Growth_Decline"] == 1) & next_cluster_mask
            gad_cluster_idx = temp_result_df[gad_mask].index
            temp_result_df.loc[gad_cluster_idx, "Cluster"] = 0

        return temp_result_df

    def _clustering_step_2(self, result_df: pd.DataFrame):
        temp_result_df = result_df.copy()
        next_cluster_df = result_df.copy()
        next_cluster_mask = next_cluster_df["Cluster"].isna()
        next_cluster_df = next_cluster_df[next_cluster_mask]
        next_cluster_df = next_cluster_df.drop(
            ["Linear", "Oscillation", "Growth_Decline", "Cluster"], axis=1
        )

        next_cluster_df.loc[:, :] = (
            next_cluster_df - next_cluster_df.mean(axis=0)
        ) / next_cluster_df.std(axis=0)
        self.linkage = linkage(next_cluster_df, method="median")
        cluster_labels = fcluster(
            self.linkage, self.n_clusters - 3, criterion="maxclust"
        )
        temp_result_df.loc[next_cluster_df.index, "Cluster"] = cluster_labels

        return temp_result_df
