import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import load_data, basic_summary
from config import DATA_PATH, TARGET

def test_dataset_loads():
    df = load_data(DATA_PATH)
    assert len(df) == 10000
    assert TARGET in df.columns

def test_target_is_binary():
    df = load_data(DATA_PATH)
    assert set(df[TARGET].unique()).issubset({0, 1})

def test_no_missing_values():
    df = load_data(DATA_PATH)
    assert int(df.isna().sum().sum()) == 0
