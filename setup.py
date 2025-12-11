import setuptools
from pathlib import Path

this_directory = Path(__file__).resolve().parent
long_description = (this_directory / "./README.md").read_text()
setuptools.setup(
	name="openspliceai",
	version="0.0.5",
	author="Kuan-Hao Chao",
	author_email="kh.chao@cs.jhu.edu",
	description="Deep learning framework that decodes splicing across species",
	url="https://github.com/Kuanhao-Chao/OpenSpliceAI",
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
        'torchaudio>=2.2.1',
        'torchvision>=0.17.1',
        'scikit-learn>=1.4.1.post1',
        'biopython>=1.83',
        'matplotlib>=3.8.3',
        'matplotlib-inline>=0.1.7',
        'psutil>=5.9.2',
        'mappy>=2.28'
    ],
    include_package_data=True,
	python_requires='>=3.9',
	packages=setuptools.find_packages(),
	entry_points={'console_scripts': ['openspliceai = openspliceai.openspliceai:main'], },
        long_description=long_description,
        long_description_content_type='text/markdown'
)
