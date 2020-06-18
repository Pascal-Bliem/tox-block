from tox_block.data_processing.data_handling import validate_input_data
from tox_block.config import config

import pytest

def test_validate_input_data():

    no_string_or_list = 123
    list_with_empty_string = ["a","","b"]
    list_with_no_string = ["a",1,"b"]
    empty_list = []
    empty_string = ""
    
    valid_string = "Hi, I'm Pascal"
    valid_list_of_strings = ["Hi","I'm","Pascal"]

    with pytest.raises(ValueError, match=r"The list item at position \d+ is an empty string."):
        validate_input_data(list_with_empty_string)

    with pytest.raises(ValueError, match=r"Passed an empty string."):
        validate_input_data(empty_string)
    
    with pytest.raises(TypeError, match=r"The list item at position \d+ is not a string."):
        validate_input_data(list_with_no_string)
    
    with pytest.raises(ValueError, match=r"Passed an empty list."):
        validate_input_data(empty_list)

    with pytest.raises(TypeError, match=r"The passed object is neither a string nor a list of strings."):
        validate_input_data(no_string_or_list)

    assert valid_string == validate_input_data(valid_string)
    assert valid_list_of_strings == validate_input_data(valid_list_of_strings)