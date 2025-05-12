import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


class Clusterer:
    def __init__(self, n_sub_clusters: int) -> None:
        self.n_clusters = None
        self.n_sub_clusters = n_sub_clusters
        self.n_step1_cluster = 0
        self.linkage = None

    def find_n_clusters(self, bhv_imp_dict: dict):
        n_clusters = 0
        for value in bhv_imp_dict.values():
            if value > 0:
                n_clusters += 1
        self.n_clusters = n_clusters

    def run_clustering(self, trans_input_df: pd.DataFrame, input_df: pd.DataFrame, consider_val: bool):
        result_df = trans_input_df.copy()
        result_df["Cluster"] = None

        feature_col_arr = trans_input_df.columns.values
        result_df = self._clustering_step_1(
            result_df=result_df, feature_col_arr=feature_col_arr
        )

        result_df = self._clustering_step_2(result_df=result_df)

        sub_cluster_df = None
        if consider_val:
            temp_inp_df = input_df.copy()
            temp_inp_df["Cluster"] = result_df["Cluster"]
            sub_cluster_df = self._value_clustering(input_df=temp_inp_df)

        return result_df, sub_cluster_df

    def _clustering_step_1(self, result_df: pd.DataFrame, feature_col_arr: np.array):
        temp_result_df = result_df.copy()

        if "Linear" in feature_col_arr:
            lnr_cluster_idx = temp_result_df[temp_result_df["Linear"] >= 0.999].index
            temp_result_df.loc[lnr_cluster_idx, "Cluster"] = -2
            self.n_step1_cluster += 1

        if "Oscillation" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            osc_mask = (temp_result_df["Oscillation"] >= 0.6) & next_cluster_mask
            osc_cluster_idx = temp_result_df[osc_mask].index
            temp_result_df.loc[osc_cluster_idx, "Cluster"] = -1
            self.n_step1_cluster += 1

        if "Growth_Decline" in feature_col_arr:
            next_cluster_mask = temp_result_df["Cluster"].isna()
            gad_mask = (temp_result_df["Growth_Decline"] == 1) & next_cluster_mask
            gad_cluster_idx = temp_result_df[gad_mask].index
            temp_result_df.loc[gad_cluster_idx, "Cluster"] = 0
            self.n_step1_cluster += 1

        return temp_result_df

    def _clustering_step_2(self, result_df: pd.DataFrame):
        temp_result_df = result_df.copy()
        next_cluster_df = result_df.copy()
        next_cluster_mask = next_cluster_df["Cluster"].isna()
        next_cluster_df = next_cluster_df[next_cluster_mask]
        drop_col_arr = next_cluster_df.columns.isin(["Linear", "Oscillation", "Growth_Decline", "Cluster"])
        drop_col_arr = next_cluster_df.columns[drop_col_arr]
        next_cluster_df = next_cluster_df.drop(
            drop_col_arr, axis=1
        )

        next_cluster_df.loc[:, :] = (
            next_cluster_df - next_cluster_df.mean(axis=0)
        ) / next_cluster_df.std(axis=0)
        self.linkage = linkage(next_cluster_df, method="median")
        cluster_labels = fcluster(
            self.linkage, self.n_clusters - self.n_step1_cluster, criterion="maxclust"
        )
        temp_result_df.loc[next_cluster_df.index, "Cluster"] = cluster_labels

        return temp_result_df

    def _value_clustering(self, input_df: pd.DataFrame):
        temp_result_df = input_df.copy()
        drop_col_mask = temp_result_df.columns.isin(["Label", "Index", "Cluster"])
        value_agg_df = temp_result_df.loc[:, ~drop_col_mask]

        input_min_ser = value_agg_df.min(axis=1).values
        input_max_ser = value_agg_df.max(axis=1).values
        #input_mean_ser = value_agg_df.mean(axis=1).values
        input_value_df = pd.DataFrame(data={"min": input_min_ser, "max": input_max_ser}, index=temp_result_df.index)

        temp_result_df["Sub-cluster"] = temp_result_df["Cluster"].astype(str) + "."
        for cluster in temp_result_df["Cluster"].unique():
            cluster_mask = temp_result_df["Cluster"] == cluster
            cluster_value_df = input_value_df[cluster_mask]
            cluster_value_df = (cluster_value_df - cluster_value_df.mean(axis=0)) / cluster_value_df.std(axis=0)

            cluster_kmeans = KMeans(n_clusters=self.n_sub_clusters, random_state=0)
            cluster_kmeans.fit(cluster_value_df)
            sub_cluster_arr = cluster_kmeans.labels_
            temp_result_df.loc[cluster_mask, "Sub-cluster"] = temp_result_df.loc[cluster_mask, "Sub-cluster"] + np.astype(sub_cluster_arr, str)

        return temp_result_df[["Label", "Cluster", "Sub-cluster"]]
