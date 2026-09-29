import setuptools
from pathlib import Path

this_directory = Path(__file__).resolve().parent
long_description = (this_directory / "./README.md").read_text()
setuptools.setup(
	name="openspliceai",
	version="0.0.8.dev0",
	author="Kuan-Hao Chao",
	author_email="kh.chao@cs.jhu.edu",
	description="Deep learning framework that decodes splicing across species",
	url="https://github.com/Kuanhao-Chao/OpenSpliceAI",
	project_urls={
	    "Documentation": "https://khchao.com/OpenSpliceAI/",
	    "Source": "https://github.com/Kuanhao-Chao/OpenSpliceAI",
	    "Bug Tracker": "https://github.com/Kuanhao-Chao/OpenSpliceAI/issues",
	},
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
    # Binary-library floors must support the NumPy 2 ABI.
    install_requires=[
        'h5py>=3.11.0',
        # numpy>=2.0: numpy removed `np.long` in 1.24 and re-added it in 2.0, so
        # 1.24-1.26 is a gap where a numpy-2-era dependency reading `np.long`
        # crashes on import (GitHub issue #19). OpenSpliceAI's own code is
        # numpy-2.0 clean, so we floor at 2.0 to keep the whole stack on the same
        # side of that gap. Linux CPU wheels for torch 2.3.0 and 2.4.1 are checked
        # against numpy 2 in the repository audit; other platforms need their
        # own wheel/ABI checks.
        'numpy>=2.0.0',
        'gffutils>=0.12',
        'pysam>=0.22.0',
        'pandas>=2.2.2',
        'pyfaidx>=0.8.1.1',
        'tqdm>=4.65.2',
        'torch>=2.3.0',
        'scikit-learn>=1.4.2',
        'scipy>=1.13.0',
        'biopython>=1.83',
        'matplotlib>=3.8.4',
        'psutil>=5.9.2',
        'mappy>=2.28'
    ],
    extras_require={
        'test': ['pytest>=7', 'pytest-cov>=4', 'Markdown>=3.4'],
        'dev': ['pytest>=7', 'pytest-cov>=4', 'Markdown>=3.4', 'ruff>=0.4', 'pre-commit>=3'],
        'analysis': ['Markdown>=3.4'],
    },
    include_package_data=True,
    package_data={'openspliceai.variant': ['annotations/*.txt']},
	python_requires='>=3.9',
	packages=setuptools.find_packages(include=['openspliceai', 'openspliceai.*']),
	entry_points={'console_scripts': ['openspliceai = openspliceai.openspliceai:main'], },
        long_description=long_description,
        long_description_content_type='text/markdown'
)
