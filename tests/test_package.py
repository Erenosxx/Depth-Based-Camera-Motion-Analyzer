def test_package_imports_and_has_version():
    import dcma
    assert isinstance(dcma.__version__, str)
    assert dcma.__version__
