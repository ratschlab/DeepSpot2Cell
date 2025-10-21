from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
REQUIREMENTS = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if (ROOT / "requirements.txt").exists() else []

setup(
    name="deepspot2cell",
    version="0.1.0",
    description="DeepSpot2Cell: virtual single-cell spatial transcriptomics from H&E images",
    long_description=README,
    long_description_content_type="text/markdown",
    author="DeepSpot2Cell Contributors",
    packages=find_packages(include=("deepspot2cell", "deepspot2cell.*")),
    install_requires=REQUIREMENTS,
    python_requires=">=3.9",
    include_package_data=True,
)
