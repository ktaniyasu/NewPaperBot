from src.utils.json_utils import extractJsonBlocks, parseJsonFromText


def test_extract_json_blocks_fenced():
    text = """
    Here is output:
    ```json
    {"a": 1, "b": 2}
    ```
    and done.
    """
    blocks = extractJsonBlocks(text)
    assert len(blocks) == 1
    assert parseJsonFromText(text) == {"a": 1, "b": 2}


def test_extract_json_blocks_curly():
    text = "prefix {\n \t \n \n \n  \n \n} suffix"
    # minimal curly braces shouldn't crash, but parse likely returns empty
    parse = parseJsonFromText(text)
    assert isinstance(parse, dict)


def test_parse_json_invalid_returns_empty():
    text = "```json\nnot a json\n```"
    assert parseJsonFromText(text) == {}
