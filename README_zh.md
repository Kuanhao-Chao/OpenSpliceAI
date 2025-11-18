# OpenSpliceAI 中文工作流指南

OpenSpliceAI 是在 PyTorch 中重构的 SpliceAI，实现了从原始基因组数据到剪接位点预测/变异注释的全流程。本指南以“从零到组织特异模型”的视角，覆盖数据准备、基础模型训练、FiLM 条件化微调以及变异注释等全部步骤，帮助你在自己的物种/组织上复现完整实验。

---

## 1. 环境与资源

- **依赖**：Python ≥ 3.10、PyTorch（GPU 训练建议 CUDA≥11.7）、NumPy/Pandas/HDF5/pyfaidx 等；执行 `pip install -e .` 会自动安装需要的 Python 包。
- **硬件**：基础模型/FiLM 微调建议使用至少 16GB GPU；Variant 注释可在 CPU 上运行但会较慢。
- **基础数据**：
  - 参考基因组 FASTA（例：`data/genome.fa` + `.fai`）。
  - 组织/物种对应的 GTF/GFF 注释文件。
  - SpliceAI 官方 annotation（示例：`data/grch38.txt`）或自定义注释。
  - 组织表达矩阵：RBP 表达 `data/tissue_rbp_matrix.csv` 与高变基因 `data/developmental_system_hvg.csv`。

安装完成后可用下列命令快速检查：

```bash
pip install -e .
openspliceai --help
```

---

## 2. 完整流程概览

1. **create-data**：读取 GTF/GFF + FASTA，生成 HDF5 数据集（训练/验证/测试）。
2. **train**：在生成的 HDF5 上训练基础模型（无 FiLM，捕捉通用剪接规则）。
3. **prepare_rbp_expression**：为目标组织生成条件向量（RBP + HVG）。
4. **transfer**：加载基础模型 checkpoint，开启 FiLM，在组织特异数据上微调。
5. **variant**：使用组织特异模型 + 条件向量对 VCF 进行剪接影响注释。

以下章节将详细说明每一步。

---

## 3. Step 1：生成 HDF5 数据 (`create-data`)

`create-data` 会执行两件事：1) `create_datafile` 将注释切成序列窗口；2) `create_dataset` 输出 HDF5 分片。典型命令：

```bash
openspliceai create-data \
  --annotation-gff data/limb_filtered.gff3 \
  --genome-fasta data/genome.fa \
  --output-dir /home1/xyf/data/openspliceai_data/dataset_limb \
  --parse-type canonical \
  --biotype protein-coding \
  --chr-split train-test \
  --split-method human \
  --split-ratio 0.8 \
  --val_split_ratio 0.1 \
  --flanking-size 10000 \
  --verify-h5 \
  --remove-paralogs \
  --min-identity 0.8 \
  --min-coverage 0.5
```

**要点**：

- `--flanking-size` 控制模型输入序列长度（越长越能捕捉远端上下文，在 10000 时需更多显存）。
- `--chr-split` + `--split-method` 决定染色体划分方式；`human` 预设以 SpliceAI 论文规则划分。
- `--remove-paralogs` 会调用 minimap2 检查训练/测试集之间的同源序列，避免信息泄漏。
- `--verify-h5` 可在生成后自动运行一致性检查。

输出目录一般包含 `dataset_train.h5`、`dataset_validation.h5`、`dataset_test.h5` 及相关日志/统计文件，后续 `train`、`transfer` 都直接引用（命名中需保留 `train`/`validation`/`test` 关键字）。

---

## 4. Step 2：训练基础模型 (`train`)

基础模型学习“组织无关”的剪接规则，稍后 FiLM 微调会在此基础上注入条件信息。示例命令：

```bash
openspliceai train \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --flanking-size 10000 \
  --epochs 10 \
  --scheduler CosineAnnealingWarmRestarts \
  --loss cross_entropy_loss \
  --output-dir runs/base_model \
  --project-name limb_base \
  --random-seed 42
```

**提示**：

- `train-dataset` 名称需包含 `train`，程序会自动将同目录下的文件名替换为 `validation` 以加载验证集（`create-data` 默认输出 `dataset_train/validation/test.h5`）。
- `train` 模式默认不启用 FiLM，也不需要 RBP/HVG 输入。
- 输出目录包含每个 epoch 的 `model_{epoch}.pt`、best checkpoint、训练/验证日志（AUPRC、loss 曲线等）。
- 若训练多个物种，可在 `create-data` 与 `train` 中切换不同 HDF5，即可得到多条基础模型。

---

## 5. Step 3：生成组织条件向量 (`prepare_rbp_expression`)

FiLM 需要一个固定的条件向量。我们提供 RBP 表达矩阵和高变基因矩阵，它们具有相同行索引（组织名）。运行脚本即可将两者拼接、标准化，并保存 JSON/NPY：

```bash
python -m openspliceai.scripts.prepare_rbp_expression \
  --matrix data/tissue_rbp_matrix.csv \
  --hvg-matrix data/developmental_system_hvg.csv \
  --tissue limb \
  --output data/limb_features.json \
  --format json \
  --standardize zscore \
  --hvg-standardize zscore
```

输出文件格式：

```json
{
  "values": [...],          # RBP + HVG 拼接后的向量
  "rbp_names": ["ADAR", ..., "HVGene_001", ...]
}
```

**务必在训练和推理时复用同一文件**，checkpoint 会记录 `rbp_dim` 与 `rbp_names` 用于校验。如果你有额外的组织特征，也可以将其加入 CSV，再通过该脚本导出。

---

## 6. Step 4：FiLM 微调 (`transfer`)

`transfer` 在基础模型之上加载 FiLM 层，将组织向量注入后半段残差块。典型命令：

```bash
openspliceai transfer \
  --train-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_train.h5 \
  --test-dataset /home1/xyf/data/openspliceai_data/dataset_limb/dataset_test.h5 \
  --pretrained-model runs/base_model/model_best.pt \
  --flanking-size 10000 \
  --epochs 5 \
  --rbp-expression data/limb_features.json \
  --film-start-layer 7 \
  --unfreeze 4 \
  --output-dir runs/limb_film \
  --project-name limb_film
```

关键参数说明：

- 与 `train` 相同，`train-dataset` 名称含 `train` 即可，程序会自动定位同目录 `dataset_validation.h5` 作为验证集。
- `--rbp-expression`：指向前一步生成的 JSON/NPY，内部包含 RBP + HVG 特征。
- `--film-start-layer`：FiLM 生效的 Residual Unit 起点（1-based），一般取网络后半段（如 12 层网络设为 7）。
- `--unfreeze` / `--unfreeze-all`：控制微调时解冻多少层；默认只训练被 FiLM 改动的层，可根据数据量调整。
- 训练日志结构与 `train` 类似，`model_best.pt` 中同时记录了 `rbp_dim`、`rbp_names` 等元信息。未提供 `--rbp-expression` 时，FiLM 会退化为 γ=1/β=0，表现等同基础模型。

要比较多个组织，可重复执行 `prepare_rbp_expression` + `transfer`，每个组织单独生成一个 checkpoint；或者在数据加载器中混合多个组织并自行扩展训练逻辑。

---

## 7. Step 5：Variant 注释 (`variant`)

最后，将 VCF 输入组织特异模型即可得到 delta 分数和剪接位点位移：

```bash
openspliceai variant \
  --input data/decipher_variants_all.vcf \
  --output results/annotated_limb.vcf \
  --model runs/limb_film/model_best.pt \
  --ref-genome data/genome.fa \
  --annotation data/grch38.txt \
  --flanking-size 10000 \
  --rbp-expression data/limb_features.json
```

**注意**：

- 若 checkpoint 包含 FiLM/条件元数据，`variant` 会检查输入向量维度与名称；缺失或顺序错误会直接报错，避免预测偏差。
- 可以多次运行 `variant`，在不同组织模型之间比较 delta 分数，评估组织特异性的剪接影响。

---

## 8. 结果与目录结构示例

```
OpenSpliceAI/
├── data/
│   ├── genome.fa / genome.fa.fai
│   ├── grch38.txt
│   ├── tissue_rbp_matrix.csv
│   └── developmental_system_hvg.csv
├── runs/
│   ├── base_model/
│   │   ├── model_best.pt
│   │   └── metrics/*.txt
│   └── limb_film/
│       ├── model_best.pt
│       └── metrics/*.txt
├── results/
│   └── annotated_limb.vcf
└── scripts/prepare_rbp_expression.py
```

建议将训练日志（TensorBoard/自定义可视化）与 checkpoint 同步归档，方便比较不同组织或参数设置。

---

## 9. 常见问题

1. **FiLM 模型可以在没有条件向量时运行吗？**  
   - 可以；若不传 `--rbp-expression`，FiLM γ=1、β=0，相当于标准 SpliceAI。仅当 checkpoint 中声明 `rbp_dim>0` 且你忘记提供向量时，程序才会报错。

2. **如何自定义组织特征？**  
   - 将任意组织级别特征（如 RBP TPM、高变基因表达、UMAP 坐标等）拼成 CSV，行名与 `tissue_rbp_matrix.csv` 保持一致，再传给 `--hvg-matrix` 或直接替换原矩阵。`prepare_rbp_expression` 会自动拼接、标准化。

3. **一次训练能覆盖多个组织吗？**  
   - 当前 CLI 默认一个组织一个 checkpoint。若想在同一模型中混合多个组织，需要自定义 dataloader，使每个 batch 附带对应的向量，再修改 `train_model` 传入不同的 `rbp_batch`。这是更复杂的改动，建议验证单组织流程后再尝试。

4. **Variant 结果如何解释？**  
   - `variant` 输出与官方 SpliceAI 相同的 delta scores（ΔAG、ΔAL、ΔDG、ΔDL）及位置偏移，可直接用于筛选可能影响剪接的突变。若对多个组织运行，可比较不同组织的分数差异。

---

## 10. 更多资源

- 文档/教程（英文）：`README.md` 与 `docs/` 目录。
- 相关脚本：
  - `openspliceai/create_data/*`：HDF5 生成与验证。
  - `openspliceai/train_base/*`：SpliceAI 主体、FiLM 层、训练/验证循环。
  - `openspliceai/rbp/expression.py`：条件向量读写及标准化工具。
  - `openspliceai/variant/variant.py`：VCF 注释入口。
- 若遇到问题或希望贡献功能，欢迎在 GitHub Issues 中提问。

祝你在 OpenSpliceAI 的研究中取得好结果！ 😊
