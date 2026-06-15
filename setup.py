from setuptools import setup, find_packages

setup(
    name="lv_harness",
    version="0.1.0",
    description="LV-Harness: a Harness architecture framework for streaming long-video reasoning Agents",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.20",
        "tqdm>=4.60",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "lv-harness=lv_harness.cli:main",
        ],
    },
)
