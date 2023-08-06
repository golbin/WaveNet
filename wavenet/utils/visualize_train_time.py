from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def main():
    path_to_data = Path("")
    network_res_paths = sorted(list(path_to_data.glob("*.csv")))

    for net_res_path in network_res_paths:
        results = pd.read_csv(net_res_path)

        x = results["batch_size"].unique()
        y = results["samples_per_batch"].unique()

        x.sort()
        y.sort()

        X, Y = np.meshgrid(x, y)
        Z = np.full(X.shape, np.nan)

        for spb_idx, spb_val in enumerate(y):
            for bs_idx, bs_val in enumerate(x):
                try:
                    Z[spb_idx, bs_idx] = results[
                        (results["samples_per_batch"] == spb_val) & (results["batch_size"] == bs_val)
                    ]["without_data_mean"].values[0] * 1000  # * 1000 for ms
                except IndexError:
                    continue

        print(net_res_path.stem)
        fig = go.Figure(go.Surface(x=x, y=y, z=Z))
        fig.update_layout(scene=dict(
            xaxis_title="Batch size",
            yaxis_title="No of samples per batch",
            zaxis_title="Execution time [ms]",
            xaxis=dict(
                ticktext=x,
                tickvals=x
            ),
        ))
        fig.show()


if __name__ == '__main__':
    main()
