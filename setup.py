import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pop2piano",
    version="0.1.0.0",
    author="sweetcocoa",
    author_email="sweetcocoa@snu.ac.kr",
    description="pop2piano: generate piano cover from pop songs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "ConfigArgParse>=1.5.3",
        "pretty-midi==0.2.9",
        "omegaconf==2.1.1",
        "youtube-dl==2021.12.17",
        "transformers==4.16.1",
        "pytorch-lightning",
        "torchaudio",
        "essentia==2.1b6.dev609",
        "note-seq==0.0.3",
        "pyFluidSynth==1.3.0",
    ],
    url="https://github.com/sweetcocoa/pop2piano",
    license="Apache2?",
    project_urls={"Bug Tracker": "https://github.com/sweetcocoa/pop2piano/issues"},
    classifiers=[],
    include_package_data=True,
    packages=setuptools.find_packages(include=["pop2piano", "pop2piano.*"]),
    python_requires=">=3",
    entry_points={
        'console_scripts': ['pop2piano = pop2piano.main:main'],
        'gui_scripts': []
    },
)
