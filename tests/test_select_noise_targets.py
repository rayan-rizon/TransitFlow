from scripts.select_noise_targets import (
    archive_target_id,
    is_selected_standard,
    parse_catalog_tsv,
    select_targets,
)


def test_archive_target_id_normalization():
    assert archive_target_id("HIP085425") == "HIP 85425"
    assert archive_target_id("TYC0123-0045-1") == "TYC 123-45-1"
    assert archive_target_id("2MASSJ00154919+1333218") \
        == "2MASS J00154919+1333218"


def _row(index: int, **overrides):
    row = {
        "ID": f"HIP {index}",
        "Flag": "CAL1",
        "Vmag": "9.0",
        "s_RV": "0.02",
        "Tbase": "1200",
        "N": "8",
        "SpType": "G2V",
        "otype": "Star",
        "RA_ICRS": str(index),
    }
    row.update(overrides)
    return row


def test_parse_catalog_tsv_skips_units_and_separator_rows():
    text = "\n".join([
        "# catalog",
        "ID\ts_RV\tFlag\tVmag\tTbase\tN\tSpType\totype\tRA_ICRS",
        "\tkm/s\t\tmag\td\t\t\t\tdeg",
        "------\t------\t----\t----\t----\t---\t----\t----\t----",
        "HIP 1\t0.02\tCAL1\t9.0\t1000\t8\tG2V\tStar\t10.0",
    ])
    rows = parse_catalog_tsv(text)
    assert rows == [{
        "ID": "HIP 1", "s_RV": "0.02", "Flag": "CAL1", "Vmag": "9.0",
        "Tbase": "1000", "N": "8", "SpType": "G2V", "otype": "Star",
        "RA_ICRS": "10.0"}]


def test_standard_filters_fail_closed():
    assert is_selected_standard(_row(1))
    assert not is_selected_standard(_row(1, Flag="VAL"))
    assert not is_selected_standard(_row(1, SpType="M3V"))
    assert not is_selected_standard(_row(1, otype="Flare*"))
    assert not is_selected_standard(_row(1, s_RV="0.2"))


def test_target_selection_is_seeded_and_requires_enough_rows():
    rows = [_row(index) for index in range(40)]
    first = select_targets(rows, 30, seed=4)
    second = select_targets(rows, 30, seed=4)
    assert [row["ID"] for row in first] == [row["ID"] for row in second]
    assert len({row["ID"] for row in first}) == 30
    try:
        select_targets(rows, 41, seed=4)
    except ValueError as exc:
        assert "requested 41" in str(exc)
    else:
        raise AssertionError("underfilled target selection did not fail")
