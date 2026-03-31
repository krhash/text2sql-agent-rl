from setuptools import setup, find_packages

setup(
    name="text2sql",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pyarrow",
        "tqdm",
    ],
    extras_require={
        "gpu": [
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "trl",
        ],
        "dev": ["pytest", "black", "ruff"],
    },
    python_requires=">=3.10",
)
