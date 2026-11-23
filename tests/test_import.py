def test_import():
    import calliopy
    import calliopy.io

    assert isinstance(calliopy.__version__, str)
