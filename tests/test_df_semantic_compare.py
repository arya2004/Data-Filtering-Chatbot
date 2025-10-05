import pandas as pd
from utils.df_semantic_compare import assert_semantic_frame_equal, CompareOptions, compare_dataframes

def test_ignore_column_and_row_order():
    a = pd.DataFrame({"b":[1,2], "a":[3,4]})
    b = pd.DataFrame({"a":[4,3], "b":[2,1]})
    assert compare_dataframes(a,b).get("equal") is True

def test_numeric_tolerance_and_rounding():
    a = pd.DataFrame({"x":[1.0001, 2.0004]})
    b = pd.DataFrame({"x":[1.0002, 2.0005]})
    assert_semantic_frame_equal(a,b, atol=0.0003)

def test_string_normalization():
    a = pd.DataFrame({"name":[" Alice  ", "BOB"], "x":[1,2]})
    b = pd.DataFrame({"name":["alice","bob"], "x":[1,2]})
    assert_semantic_frame_equal(a,b, case_insensitive=True, strip_strings=True)

def test_keyed_alignment_cells():
    a = pd.DataFrame({"id":[1,2], "val":[10.0, 20.0]})
    b = pd.DataFrame({"id":[1,2], "val":[10.0, 20.001]})
    rep = compare_dataframes(a,b, CompareOptions(keys=["id"], atol=0.01))
    assert rep["equal"] is True
