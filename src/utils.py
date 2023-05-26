import json
from typing import Dict

import numpy as np
import pandas as pd
import logging


def read_weather_csv(path: str) -> pd.DataFrame:
    "Read a csv file and returns a DataFrame object"
    logging.info(f"Reading {path} . . .")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
    df = df.set_index("Date")
    return df


def read_his_data_csv(path: str, source_tag: bool) -> pd.DataFrame:
    "Read a csv file and returns a DataFrame object"
    logging.info(f"Reading {path} . . .")
    df = pd.read_csv(path, index_col=0)
    # df = df[df['label'].isin([22])]
    # 打乱df 顺序
    if source_tag:
        logging.info('suiji')
        df = df[df['label'].isin([22, 18, 19, 23, 1, 9])]
        # df = df.sample(frac=1).reset_index(drop=True)
    return df


def clip_num(value: float, minimum: float = -np.inf, maximum: float = np.inf) -> float:
    "Clip the value between minimum and maximum parameters"
    return min(max(value, minimum), maximum)


def save_dict(dic: Dict, path: str, verbose: bool = True) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(dic))
    if verbose:
        logging.info(f"Dictionary saved to {path}")


def load_dict(path: str) -> Dict:
    with open(path) as f:
        dic = json.loads(f.read())
    logging.info(f"Dictionary readed from {path}")
    return dic
