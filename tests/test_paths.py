import pytest

from solete.paths import data_filename, normalize_resolution


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [
        ("1sec", "SOLETE_Pombo_1sec_v4.h5"),
        ("1min", "SOLETE_Pombo_1min_v4.h5"),
        ("5min", "SOLETE_Pombo_5min_v4.h5"),
        ("60min", "SOLETE_Pombo_60min_v4.h5"),
        ("1h", "SOLETE_Pombo_60min_v4.h5"),
    ],
)
def test_v4_release_filename(resolution, expected):
    assert data_filename(resolution, version="v4") == expected


def test_v4_original_release_filename():
    assert data_filename(
        "1sec", version="v4", fmt="parquet", original=True
    ) == "SOLETE_Pombo_1sec_original_v4.parquet"


def test_original_release_is_only_valid_at_one_second():
    with pytest.raises(ValueError, match="only"):
        data_filename("1min", version="v4", original=True)


def test_one_hour_is_input_alias_only():
    assert normalize_resolution("1h") == "60min"
    assert "1h" not in data_filename("1h", version="v4")