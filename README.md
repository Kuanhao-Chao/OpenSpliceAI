<img alt="My Logo" class="logo header-image only-dark align-center" src="https://khchao.com/images/openspliceai_logo_black.png" style="width:90%">


<a class="reference external image-reference" href="https://img.shields.io/badge/License-GPLv3-yellow.svg"><img alt="https://img.shields.io/badge/License-GPLv3-yellow.svg" src="https://img.shields.io/badge/License-GPLv3-yellow.svg"></a>
<a class="reference external image-reference" href="https://img.shields.io/badge/version-v.0.0.4-blue"><img alt="https://img.shields.io/badge/version-v.0.0.4-blue" src="https://img.shields.io/badge/version-v.0.0.4-blue"></a>
<a class="reference external image-reference" href="https://pepy.tech/project/openspliceai"><img alt="https://static.pepy.tech/personalized-badge/openspliceai?period=total&amp;units=abbreviation&amp;left_color=grey&amp;right_color=blue&amp;left_text=PyPi%20downloads" src="https://static.pepy.tech/personalized-badge/openspliceai?period=total&amp;units=abbreviation&amp;left_color=grey&amp;right_color=blue&amp;left_text=PyPi%20downloads"></a>
<a class="reference external image-reference" href="https://github.com/Kuanhao-Chao/OpenSpliceAI/releases"><img alt="https://img.shields.io/github/downloads/Kuanhao-Chao/OpenSpliceAI/total.svg?style=social&amp;logo=github&amp;label=Download" src="https://img.shields.io/github/downloads/Kuanhao-Chao/OpenSpliceAI/total.svg?style=social&amp;logo=github&amp;label=Download"></a>
<a class="reference external image-reference" href="https://github.com/Kuanhao-Chao/OpenSpliceAI/releases"><img alt="https://img.shields.io/badge/platform-macOS_/Linux-green.svg" src="https://img.shields.io/badge/platform-macOS_/Linux-green.svg"></a>
<div class="line-block">
<div class="line"><br></div>
</div>
<p>OpenSpliceAI is an open‐source, efficient, and modular framework for splice site prediction. It is a reimplementation and extension of SpliceAI (Jaganathan et al., 2019) built on the modern PyTorch framework. OpenSpliceAI provides researchers with a user‐friendly suite of tools for studying transcript splicing - from creating training datasets and training models to predicting splice sites and assessing the impact of genetic variants.</p>
<div class="line-block">
<div class="line"><br></div>
</div>
<section id="key-features">
<h1>Key Features<a class="headerlink" href="#key-features" title="Permalink to this heading">#</a></h1>
<ul class="simple">
<li><p><strong>Modern, Retrainable Framework:</strong> Built on Python 3 and PyTorch, OpenSpliceAI improves the limitations of older TensorFlow/Keras implementations. Its modular design enables fast and efficient prediction, as well as easy retraining on species-specific data with just a few commands.</p></li>
<li><p><strong>Updated and Cross-Species Models:</strong> OpenSpliceAI includes a pre-trained human model, <b>OSAI<sub>MANE</sub>-10000nt</b>, updated from GRCh37 to GRCh38 using the latest MANE annotations, along with models for mouse, thale cress (<em>Arabidopsis</em>), honey bee, and zebrafish. This versatility empowers researchers to study splicing across diverse species.</p></li>
<li><p><strong>Variant Impact Prediction:</strong> OpenSpliceAI not only predicts splice sites but also assesses the impact of genetic variants (SNPs and INDELs) on splicing. Its <code class="docutils literal notranslate"><span class="pre">variant</span></code> subcommand calculates “delta” scores that quantify changes in splice site strength and predicts cryptic splice sites.</p></li>
<li><p><strong>Efficiency and Scalability:</strong> Optimized for improved processing speeds, lower memory usage, and efficient GPU utilization, OpenSpliceAI can handle large genomic regions and whole-genome predictions on a single GPU.</p></li>
</ul>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="who-should-use-openspliceai">
<h1>Who Should Use OpenSpliceAI?<a class="headerlink" href="#who-should-use-openspliceai" title="Permalink to this heading">#</a></h1>
<ul class="simple">
<li><p><strong>Human Genomics Researchers:</strong>
Use the newly retrained OpenSpliceAI model, <b>OSAI<sub>MANE</sub>-10000nt</b>,  for highly accurate splice site predictions based on the latest human annotations.</p></li>
<li><p><strong>Comparative and Non-Human Genomics:</strong>
Whether you’re studying mouse, zebrafish, honey bee, or thale cress, OpenSpliceAI offers models pre-trained on multiple species — and the ability to train your own models — ensuring broad applicability.</p></li>
<li><p><strong>Variant Analysts:</strong>
If you need to predict how genetic variants affect splicing, OpenSpliceAI’s variant subcommand provides detailed delta scores and positional information to assess functional impacts.</p></li>
</ul>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="what-openspliceai-does">
<h1>What OpenSpliceAI Does<a class="headerlink" href="#what-openspliceai-does" title="Permalink to this heading">#</a></h1>
<ul class="simple">
<li><p><strong>Data Preprocessing</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#create-data-subcommand"><span class="std std-ref">create-data</span></a>):
Converts genome FASTA and annotation (GFF/GTF) files into one-hot encoded datasets (HDF5 format) for training and testing.</p></li>
<li><p><strong>Model Training</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#train-subcommand"><span class="std std-ref">train</span></a>):
Trains deep residual convolutional neural networks on the preprocessed datasets. OpenSpliceAI supports training from scratch and employs adaptive learning rate schedulers and early stopping.</p></li>
<li><p><strong>Transfer Learning</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#transfer-subcommand"><span class="std std-ref">transfer</span></a>):
Fine-tunes a pre-trained human model for other species, reducing training time and improving performance on species with limited data.</p></li>
<li><p><strong>Model Calibration</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#calibrate-subcommand"><span class="std std-ref">calibrate</span></a>):
Adjusts model output probabilities to better reflect true splice site likelihoods, enhancing prediction accuracy.</p></li>
<li><p><strong>Prediction</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#predict-subcommand"><span class="std std-ref">predict</span></a>):
Uses trained models to generate splice site predictions from FASTA sequences, outputting BED files with donor and acceptor site coordinates.</p></li>
<li><p><strong>Variant Analysis</strong> (<a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#variant-subcommand"><span class="std std-ref">variant</span></a>):
Annotates VCF files with delta scores and positions to evaluate the impact of genetic variants on splicing.</p></li>
</ul>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="cite-us">
<h1>Cite Us<a class="headerlink" href="#cite-us" title="Permalink to this heading">#</a></h1>
<p>If you use OpenSpliceAI in your research, please cite our work as well as the original SpliceAI paper:</p>
<p>Kuan-Hao Chao, Alan Mao, Anqi Liu, Mihaela Pertea, and Steven L. Salzberg. <i>"OpenSpliceAI provides an efficient modular implementation of SpliceAI enabling easy retraining across nonhuman species"</i> <a href="https://doi.org/10.7554/eLife.107454.3" target="_blank"><b><i>eLife</i> 14:RP107454</b></a>.

<p>Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae, Siavash Fazel Darbandi, David Knowles, Yang I. Li, Jack A. Kosmicki, Juan Arbelaez, Wenwu Cui, Grace B. Schwartz, Eric D. Chow, Efstathios Kanterakis, Hong Gao, Amirali Kia, Serafim Batzoglou, Stephan J. Sanders, and Kyle Kai-How Farh. <i>"Predicting splicing from primary sequence with deep learning"</i> <b>Cell</b>.

<div class="line"><br></div>
</div>
</section>
<section id="user-support-contributors">
<h1>User Support &amp; Contributors<a class="headerlink" href="#user-support-contributors" title="Permalink to this heading">#</a></h1>
<p>If you have questions, encounter issues, or would like to request a new feature, please use our GitHub issue tracker at:
<a class="reference external" href="https://github.com/Kuanhao-Chao/OpenSpliceAI/issues">https://github.com/Kuanhao-Chao/OpenSpliceAI/issues</a></p>
<p>OpenSpliceAI was developed by Kuan-Hao Chao, Alan Mao, and collaborators at Johns Hopkins University. For further details on usage, methods, and performance, please refer to the full documentation and online methods sections.</p>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="next-steps">
<h1>Next Steps<a class="headerlink" href="#next-steps" title="Permalink to this heading">#</a></h1>
<p>Check out the <a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#installation"><span class="std std-ref">Installation Guide</span></a> to get started with OpenSpliceAI. For a quick overview of the main commands and subcommands, see the <a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#quick-start-home"><span class="std std-ref">Quick Start Guide</span></a>.</p>
<div class="line-block">
<div class="line"><br></div>
</div>
</section>
<section id="table-of-contents">
<h1>Table of Contents<a class="headerlink" href="#table-of-contents" title="Permalink to this heading">#</a></h1>
<div class="toctree-wrapper compound">
<ul>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html">Installation</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#overview">Overview</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#prerequisites">Prerequisites</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#installation-methods">Installation Methods</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#detailed-installation-for-pytorch-and-mappy">Detailed Installation for PyTorch and mappy</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#check-openspliceai-installation">Check OpenSpliceAI Installation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#terminal-output-example">Terminal Output Example</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/installation.html#next-steps">Next Steps</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html">Quick Start Guide</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-1-predict">Usage 1 – Predict</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-2-train-from-scratch">Usage 2 – Train from Scratch</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/quick_start_guide/index.html#usage-3-transfer-learning">Usage 3 – Transfer Learning</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html">create-data</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#processing-pipeline">Processing Pipeline</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_create-data.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html">train</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#subcommand-description">Subcommand Description</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#processing-steps">Processing Steps</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_train.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html">transfer</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#subcommand-description">Subcommand Description</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#processing-pipeline">Processing Pipeline</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_transfer.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html">calibrate</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#subcommand-description">Subcommand Description</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#processing-steps">Processing Steps</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_calibrate.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html">predict</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#overview">Overview</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#processing-pipeline">Processing Pipeline</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_predict.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html">variant</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#overview">Overview</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#input-files">Input Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#output-files">Output Files</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#delta-score-computation">Delta Score Computation</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#usage">Usage</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#examples">Examples</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#processing-pipeline">Processing Pipeline</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_variant.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/train_your_own_model/index.html">Steps &amp; Commands to Train OpenSpliceAI Models</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/pretrained_models/index.html">Released OpenSpliceAI models</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html">OpenSpliceAI vs. SpliceAI</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#overview">Overview</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#architectural-and-implementation-differences">Architectural and Implementation Differences</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#performance-and-efficiency">Performance and Efficiency</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#training-flexibility-and-transfer-learning">Training Flexibility and Transfer Learning</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#model-calibration-and-variant-analysis">Model Calibration and Variant Analysis</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/openspliceai_vs_spliceai.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html">Behind the Scenes</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#architecture-and-framework">Architecture and Framework</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#data-preprocessing-and-one-hot-encoding">Data Preprocessing and One-Hot Encoding</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#training-and-optimization">Training and Optimization</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#transfer-learning">Transfer Learning</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#model-calibration">Model Calibration</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#variant-effect-analysis">Variant Effect Analysis</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#performance-and-benchmarking">Performance and Benchmarking</a></li>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/behind_scenes.html#conclusion">Conclusion</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/how_to_page.html">Q &amp; A</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/changelog.html">Changelog</a><ul>
<li class="toctree-l2"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/changelog.html#v1-0-0">v1.0.0</a></li>
</ul>
</li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/license.html">License</a></li>
<li class="toctree-l1"><a class="reference internal" href="https://ccb.jhu.edu/openspliceai/content/contact.html">Contact</a></li>
</ul>
</div>
<div class="line-block">
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
<div class="line"><br></div>
</div>
<img alt="My Logo" class="logo header-image only-light align-center" src="https://khchao.com/images/jhu-logo-dark.png">
</section>
