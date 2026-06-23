import setuptools
from pathlib import Path

this_directory = Path(__file__).resolve().parent
long_description = (this_directory / "./README.md").read_text()
setuptools.setup(
	name="openspliceai",
	version="0.0.7",
	author="Kuan-Hao Chao",
	author_email="kh.chao@cs.jhu.edu",
	description="Deep learning framework that decodes splicing across species",
	url="https://github.com/Kuanhao-Chao/OpenSpliceAI",
	license="GPL-3.0-only",
	classifiers=[
	    "Development Status :: 4 - Beta",
	    "Intended Audience :: Science/Research",
	    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
	    "Operating System :: POSIX :: Linux",
	    "Operating System :: MacOS",
	    "Programming Language :: Python :: 3",
	    "Programming Language :: Python :: 3.9",
	    "Programming Language :: Python :: 3.10",
	    "Programming Language :: Python :: 3.11",
	    "Programming Language :: Python :: 3.12",
	    "Topic :: Scientific/Engineering :: Bio-Informatics",
	],
	# install_requires=
    install_requires=[
        'h5py>=3.9.0',
        'numpy>=1.24.4',
        'gffutils>=0.12',
        'pysam>=0.22.0',
        'pandas>=1.5.3',
        'pyfaidx>=0.8.1.1',
        'tqdm>=4.65.2',
        'torch>=2.2.1',
        'scikit-learn>=1.4.1.post1',
        'biopython>=1.83',
        'matplotlib>=3.8.3',
        'psutil>=5.9.2',
        'mappy>=2.28'
    ],
    extras_require={
        'test': ['pytest>=7', 'pytest-cov>=4'],
        'dev': ['pytest>=7', 'pytest-cov>=4', 'ruff>=0.4', 'pre-commit>=3'],
    },
    include_package_data=True,
    package_data={'openspliceai.variant': ['annotations/*.txt']},
	python_requires='>=3.9',
	packages=setuptools.find_packages(include=['openspliceai', 'openspliceai.*']),
	entry_points={'console_scripts': ['openspliceai = openspliceai.openspliceai:main'], },
        long_description=long_description,
        long_description_content_type='text/markdown'
)
