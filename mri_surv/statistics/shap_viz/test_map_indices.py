import pandas as pd
from importlib import reload
from typing import Dict

def test__map_old_to_new_index():
    index_to_lobe = (
        pd.read_csv("metadata/data_processed/index_to_lobe_map.csv")
        .set_index("ID")
        .to_dict()["Lobe"]
    )
    index_to_lobe = _map_old_to_new_index(index_to_lobe)

def _assign_id_to_side(l: list, side: str) -> int:
    if len(l) == 1:
        return int(l[0])    
    assert len(l) == 2
    if side == "l":
        return int(l[0])
    elif side == "r":
        return int(l[1])
    raise ValueError

def _map_old_to_new_index(new_index_to_lobe: pd.DataFrame) -> Dict[int, int]:
    df = pd.read_csv("./metadata/data_raw/neuromorphometrics/neuromorphometrics.csv", sep=";")
    df = df.set_index("ROIid")
    df = df.assign(
            ROIid_new=df["ROIbaseid"].apply(lambda x: x.strip("[]").strip().split(" ")),
            Side=df["ROIabbr"].apply(lambda x: x[0] if x != "BG" else x)
        )
    df = df.assign(
            ROIid_new=df[["ROIid_new", "Side"]].agg(lambda x: _assign_id_to_side(x[0],x[1]), axis=1)
        )
    roi_dict = df["ROIid_new"].to_dict()
    new_dict = {roi_dict[x]: y for x,y in new_index_to_lobe.items()}
    return new_dict
