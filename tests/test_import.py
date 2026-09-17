"""Smoke test: the src/ layout resolves and the package is importable."""


def test_package_imports() -> None:
    import perceptual_media  # noqa: F401
