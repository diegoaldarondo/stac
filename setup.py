"""Setup file for stac."""
from setuptools import setup, find_packages

setup(
    name="stac",
    version="0.0.1",
    packages=find_packages(),
    scripts=["cluster/submit_stac_single_batch.sh", "cluster/dannce2npmp.sh"],
    entry_points={
        "console_scripts": [
            "stac-submit = stac.interface:submit",
            "stac-merge = merge_snippets:stac_merge",
            "stac-submit-unfinished = stac.interface:submit_unfinished",
            "stac-compute-single-batch = stac.interface:compute_single_batch",
        ]
    },
    install_requires=[
        "six >= 1.12.0",
        "clize >= 4.0.3",
        "absl-py >= 0.7.1",
        "enum34",
        "future",
        # 'futures',
        "glfw",
        "lxml",
        "numpy",
        "pyopengl",
        "pyparsing",
        "h5py >= 2.9.0",
        "scipy >= 1.2.1",
        "pyyaml",
        "opencv-python",
    ],
)
