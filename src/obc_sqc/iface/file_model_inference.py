from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
import json

import pandas as pd

from obc_sqc.model.obc_sqc_driver import ObcSqcCheck
from obc_sqc.schema.schema import SchemaDefinitions

logger = logging.getLogger("obc_sqc")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

import warnings

warnings.filterwarnings("ignore")


def main():
    """The algo requires an input a timeseries in csv with raw data of parameters of 'temperature',
    'humidity', 'wind_speed', 'wind_direction', 'pressure' and 'illuminance'. The following are conducted:
     - create a new timeframe with fixed time interval
     - check for constant data
     - check for jumps and availability in raw level (a)
     - check for jumps and availability in minute level (b)
     - check for availability in hourly level (c)
     - export results from a, b and c into csvs
    """  # noqa: D202, D205, D400, D415

    time.time()

    parser = argparse.ArgumentParser(description="OBC SQC Direct Inference")

    # parser.add_argument("--device_id", help="Device ID", required=True)
    parser.add_argument("--date", help="", required=True)
    parser.add_argument("--day1", help="", required=True)
    parser.add_argument("--day2", help="", required=True)
    parser.add_argument("--output_file_path", help="", default="output.parquet")

    (k_args, unknown_args) = parser.parse_known_args(sys.argv[1:])

    args = vars(k_args)

    # Convert start and end dates to datetime
    input_date: datetime = datetime.datetime.strptime(args["date"], "%Y-%m-%d")
    starting_date = input_date - pd.Timedelta(hours=6)
    end_date = input_date + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # QoD object/model/classifier
    qod_model = ObcSqcCheck()

    df1: pd.DataFrame = pd.read_parquet(args["day1"])
    df2: pd.DataFrame = pd.read_parquet(args["day2"])

    unique_device_ids_day1 = set(df1["device_id"].unique())
    unique_device_ids_day2 = set(df2["device_id"].unique())
    all_unique_device_ids = unique_device_ids_day1.union(unique_device_ids_day2)

    df1 = df1.astype(SchemaDefinitions.qod_input_schema()).drop_duplicates()
    df2 = df2.astype(SchemaDefinitions.qod_input_schema()).drop_duplicates()

    combined_df: pd.DataFrame = pd.concat([df1, df2])
    result_qod_devices = []
    for device_id in list(all_unique_device_ids):
        device_df = combined_df[combined_df["device_id"] == device_id].drop(columns=["device_id"])

        # In-memory filtering
        df_with_schema: pd.DataFrame = device_df[
            (device_df["utc_datetime"] >= str(starting_date)) & (device_df["utc_datetime"] <= str(end_date))
        ].reset_index(drop=True)

        result_df, score = qod_model.run(df_with_schema)
        result_qod_devices.append({"device_id": device_id, "qod_score": score})
        # result_df.to_parquet(f"{args['output_file_path']}.parquet", index=False)
    
    with open(f"{args['output_file_path']}.json", "w") as f:
        json.dump(result_qod_devices, f)


if __name__ == "__main__":
    main()
