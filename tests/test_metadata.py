from rppg10.metadata import _parse_row_xml


def test_sparse_row_gap_does_not_shift_columns():
    """A missing spreadsheet cell must not shift a later value."""
    row_xml = (
        '<c r="A1" t="inlineStr"><is><t>alpha</t></is></c>'
        '<c r="C1" t="inlineStr"><is><t>gamma</t></is></c>'
    )
    row = _parse_row_xml(row_xml, ss=[])
    assert row == ["alpha", "", "gamma"]
