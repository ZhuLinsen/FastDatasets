from setuptools import setup, find_packages

setup(
    name="fastdatasets",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "gradio",
        "pydantic",
        "python-dotenv>=1.0.0",
        "loguru",
        "openai>=1.0.0",
        "rich>=13.0.0",
        "textract-py3",
        "tqdm>=4.65.0",
    ],
) 