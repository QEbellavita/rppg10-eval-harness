"""Parse Subject Data.xlsx with the stdlib and join strictly by 'Subject N' header label."""
import re
import html
import zipfile
from dataclasses import dataclass


@dataclass
class SubjectMeta:
    subject_id: int
    age: "int | None"
    sex: "str | None"
    fitzpatrick: "str | None"
    arrhythmia: bool
    skin_obstruction: "str | None"


def _col_letters_to_index(letters: str) -> int:
    """Convert OOXML column letters to a 0-based column index (A→0, B→1, Z→25, AA→26)."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def _parse_row_xml(row_xml: str, ss: list) -> list:
    """Parse a single OOXML <row> inner XML into a list of cell values.

    Cells are placed at their true column index (from the ``r`` attribute) so that
    sparse rows with missing cells do NOT shift later columns ;  the gap is filled
    with an empty string instead.
    """
    row: list = []
    sequential_pos = 0  # fallback for <c> elements without an r attribute (rare)

    for c_xml in re.findall(r"<c\b[^>]*>.*?</c>|<c\b[^>]*/>", row_xml, re.S):
        # Determine true column index from the r attribute (e.g. r="E3" → col 4)
        ref_match = re.search(r'\br="([A-Z]+)\d+"', c_xml)
        if ref_match:
            col_idx = _col_letters_to_index(ref_match.group(1))
            sequential_pos = col_idx + 1
        else:
            col_idx = sequential_pos
            sequential_pos += 1

        # Extend the row list with "" to reach this column if necessary
        while len(row) <= col_idx:
            row.append("")

        # Extract the cell value
        t = re.search(r'\bt="(\w+)"', c_xml)
        v = re.search(r"<v>(.*?)</v>", c_xml, re.S)
        it = re.search(r"<is>.*?<t[^>]*>(.*?)</t>", c_xml, re.S)
        val = it.group(1) if it else (v.group(1) if v else "")
        if t and t.group(1) == "s" and val.isdigit():
            val = ss[int(val)]
        row[col_idx] = html.unescape(val)

    return row


def _read_sheet_rows(xlsx_path: str) -> list:
    with zipfile.ZipFile(xlsx_path) as z:
        names = z.namelist()
        ss: list = []
        if "xl/sharedStrings.xml" in names:
            raw = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
            ss = [html.unescape(re.sub("<[^>]+>", "", m))
                  for m in re.findall(r"<si>(.*?)</si>", raw, re.S)]
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8", "ignore")

    rows = []
    for r in re.findall(r"<row[^>]*>(.*?)</row>", sheet, re.S):
        rows.append(_parse_row_xml(r, ss))
    return rows


def load_metadata(xlsx_path: str) -> "dict[int, SubjectMeta]":
    rows = _read_sheet_rows(xlsx_path)
    header = rows[0]
    # column index -> subject number, parsed from the 'Subject N' label (NEVER positional)
    col_subject: dict = {}
    for idx, label in enumerate(header):
        mobj = re.match(r"\s*Subject\s+(\d+)\s*$", str(label))
        if mobj:
            col_subject[idx] = int(mobj.group(1))

    def row_for(name: str) -> list:
        for r in rows:
            if r and r[0].strip() == name:
                return r
        return []

    skin = row_for("Skin Tone (FitzPatrick)")
    health = row_for("Health Condition")
    age = row_for("Age (years)")
    sex = row_for("Sex (M/F)")
    obstruction = row_for("Skin obstruction")

    def cell(row: list, idx: int) -> "str | None":
        return row[idx].strip() if (row and idx < len(row) and row[idx] != "") else None

    out: dict = {}
    for idx, subj in col_subject.items():
        age_val = cell(age, idx)
        health_val = cell(health, idx)
        out[subj] = SubjectMeta(
            subject_id=subj,
            age=int(age_val) if (age_val and age_val.isdigit()) else None,
            sex=cell(sex, idx),
            fitzpatrick=cell(skin, idx),
            arrhythmia=bool(health_val) and health_val.lower().startswith("arr"),
            skin_obstruction=cell(obstruction, idx),
        )
    return out
