# RNA GCP-VQVAE

RNA adaptation of GCP-VQVAE for backbone structure tokenization and reconstruction.

This repository is an RNA-focused adaptation of the original GCP-VQVAE project. We explicitly acknowledge and thank the original GCP-VQVAE authors, Mahdi Pourmirzaei, Alex Morehead, Farzaneh Esmaili, Jarett Ren, Mohammadreza Pourmirzaei, and Dong Xu, for releasing the original method and codebase that this work builds upon.

本仓库是在原始 GCP-VQVAE 项目基础上发展出的 RNA 版本。这里首先明确致谢 GCP-VQVAE 原作者 Mahdi Pourmirzaei、Alex Morehead、Farzaneh Esmaili、Jarett Ren、Mohammadreza Pourmirzaei 和 Dong Xu，本项目建立在他们公开的方法与代码基础之上。

This repository keeps the original GCP-VQVAE architecture skeleton, but the current training path is RNA-specific:

- raw data are split at the PDB level with homology control
- split PDBs are converted to RNA HDF5 files
- training reads RNA HDF5 files and uses RNA-specific graph features, decoder templates, and loss branches

## Overview

项目概览。
The active RNA training stack is built around these files:

- data split: [`data/rna_homology_split.py`](data/rna_homology_split.py)
- PDB to H5 conversion: [`data/rna_pdb_to_h5.py`](data/rna_pdb_to_h5.py)
- RNA dataset loader: [`data/dataset.py`](data/dataset.py)
- RNA encoder config: [`configs/config_gcpnet_encoder_rna.yaml`](configs/config_gcpnet_encoder_rna.yaml)
- AF2-style RNA decoder config: [`configs/config_rna_af2_decoder.yaml`](configs/config_rna_af2_decoder.yaml)
- RNA training configs:
  - scratch: [`configs/config_vqvae_dihedral.yaml`](configs/config_vqvae_dihedral.yaml)
  - continue: [`configs/config_vqvae_dihedral_continue.yaml`](configs/config_vqvae_dihedral_continue.yaml)
  - continue + validate every epoch: [`configs/config_vqvae_dihedral_continue_val.yaml`](configs/config_vqvae_dihedral_continue_val.yaml)
- AF2-style RNA decoder implementation: [`models/rna_af2_decoder.py`](models/rna_af2_decoder.py)
- training entry: [`train.py`](train.py)

The config directory has been trimmed around this active RNA path:

- the `config_vqvae*` family now keeps only the dihedral series above
- generic or older experimental `config_vqvae*.yaml` variants were removed
- inference and evaluation configs were updated to point to the dihedral + RNA AF2 decoder stack

## Installation

安装说明。
Core dependencies:

核心依赖：
- Python 3.10
- PyTorch
- torch-geometric
- torch-scatter
- torch-cluster
- accelerate
- transformers
- vector-quantize-pytorch
- x-transformers
- graphein
- h5py
- biopython
- tmtools

Recommended one-line installation:

推荐的一行安装命令：

```bash
bash install.sh
```

For an exact frozen environment, use:

如果需要严格复现冻结环境，可使用：

```bash
pip install -r requirements_freeze.txt
```

Optional, Hopper only:

可选，Hopper GPU 专用：
```bash
bash install_flash_attention_3_hopper.sh
```

Install scripts:

相关安装文件：
- base environment: [`install.sh`](install.sh)
- FlashAttention-3 for Hopper: [`install_flash_attention_3_hopper.sh`](install_flash_attention_3_hopper.sh)
- frozen package list: [`requirements_freeze.txt`](requirements_freeze.txt)

## Data Pipeline

数据处理流程。
The intended RNA pipeline is:

推荐的数据流程如下：

1. split raw PDB files by homology
2. convert split PDBs to RNA HDF5
3. point the training config to the generated HDF5 split directories
4. launch training with `accelerate`

This is intentionally decoupled:

这几个步骤是刻意解耦的：
- homology split outputs PDB files, not H5
- training consumes H5 files, not raw PDBs

The trainer enforces this at load time in [`data/dataset.py`](data/dataset.py).

训练器会在加载阶段检查这一约束，相关逻辑见 [`data/dataset.py`](data/dataset.py)。
## Dataset

数据集说明。
The dataset is built from the RNAsolo database, which collects experimentally determined RNA 3D structures from the Protein Data Bank (PDB), removes non-RNA chains, and organizes structures into equivalence classes to provide non-redundant representatives.

本项目的数据集来自 RNAsolo。RNAsolo 从 PDB 收集实验解析得到的 RNA 三维结构，去除非 RNA 链，并按等价类组织为非冗余代表集。
Dataset construction workflow:

数据构建流程：
1. RNA structures are downloaded from RNAsolo.
2. The following filters are applied: repository `BGSU`, redundancy `representatives`, experimental methods `X-ray crystallography` and `NMR spectroscopy`, X-ray resolution `<= 3.0 A`, and molecule type `all`.
3. Because RNAsolo files already contain RNA chains only, no additional protein-chain removal is required.
4. If a PDB file contains multiple RNA chains, each chain is treated as an independent sample during preprocessing.
5. Only the RNA backbone atoms required by the model are retained when converting structures into training-ready HDF5 files.

## Step 1: Homology Split Raw PDBs

第 1 步：对原始 PDB 做同源划分。
Use [`data/rna_homology_split.py`](data/rna_homology_split.py).

This script currently:

- parses BGSU-style filenames such as `PDB_00001A9N_1_Q.pdb`
- queries RCSB for RNA sequences
- clusters sequences with `cd-hit-est`
- writes `train/val/test` split directories containing symlinks or copies of the original PDBs

Important:

注意：
- current `cd-hit-est` usage in this repo requires `--identity >= 0.80`
- the script supports shared cache reuse through `--rcsb_cache_path`

Example:

```bash
python -u data/rna_homology_split.py \
  --input_root /path/to/raw_bgsu_pdb_root \
  --extensions .pdb \
  --output_dir /path/to/bgsu_pdb_homology_split_c80 \
  --identity 0.80 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --threads 16 \
  --materialize symlink \
  --api_sleep_s 0.15 \
  --cache_flush_every 20 \
  --rcsb_cache_path /path/to/rcsb_cache_bgsu.json \
  --overwrite
```

Relevant code:

对应代码：
- BGSU filename parsing: [`data/rna_homology_split.py`](data/rna_homology_split.py)
- chain-group resolution such as `A-B`: [`data/rna_homology_split.py`](data/rna_homology_split.py)
- `cd-hit-est` identity guard: [`data/rna_homology_split.py`](data/rna_homology_split.py)

## Step 2: Convert Split PDBs to RNA HDF5

第 2 步：将划分后的 PDB 转为 RNA HDF5。
Use [`data/rna_pdb_to_h5.py`](data/rna_pdb_to_h5.py).

This converter writes RNA-specific datasets:

转换后会写入以下 RNA 专用字段：
- `seq`
- `C4p_C1p_N_coord`
- `plddt_scores`
- `P_O5p_C5p_C4p_C3p_O3p_coord`

Example:

```bash
for split in train val test; do
  python -u data/rna_pdb_to_h5.py \
    --data /path/to/bgsu_pdb_homology_split_c80/splits/${split} \
    --save_path /path/to/bgsu_h5_homology_split_c80/splits/${split} \
    --max_len 511 \
    --min_len 11 \
    --max_missing_ratio 0.30 \
    --gap_threshold 5 \
    --max_workers 16
done
```

Relevant code:

对应代码：
- RNA residue and atom mapping: [`data/rna_pdb_to_h5.py`](data/rna_pdb_to_h5.py)
- RNA H5 keys: [`data/rna_pdb_to_h5.py`](data/rna_pdb_to_h5.py)

## Step 3: Modify Training Paths

第 3 步：修改训练配置中的路径。
You only need to edit a few keys in the config files.

通常只需要修改少数几个配置项。
### Scratch Training

Edit [`configs/config_vqvae_dihedral.yaml`](configs/config_vqvae_dihedral.yaml):

- `train_settings.data_path`
- `valid_settings.data_path`
- optionally `result_path`
- decoder hyperparameters such as `num_layer` and `share_weights` live in [`configs/config_rna_af2_decoder.yaml`](configs/config_rna_af2_decoder.yaml)
- validation now runs every epoch by default through `valid_settings.do_every: 1`
- validation PDB export is enabled by default through `valid_settings.save_pdb_every: 1`

Current repository default:

当前仓库默认值：
- `train_settings.data_path: /data/ymxue/p2_rnavqvae/alex/data/bgsu_h5_homology_split_c80/splits/train`
- `valid_settings.data_path: /data/ymxue/p2_rnavqvae/alex/data/bgsu_h5_homology_split_c80/splits/val`

### Continue Training

Edit [`configs/config_vqvae_dihedral_continue.yaml`](configs/config_vqvae_dihedral_continue.yaml):

- `train_settings.data_path`
- `valid_settings.data_path`
- `resume.resume_path`
- optionally `result_path`

This config now matches the same active RNA decoder/loss stack as scratch training:

- decoder: `rna_af2_decoder`
- reconstruction losses: `final_fape + aux_fape + vq`

### Continue Training With Validation Every Epoch

Edit [`configs/config_vqvae_dihedral_continue_val.yaml`](configs/config_vqvae_dihedral_continue_val.yaml):

- `train_settings.data_path`
- `valid_settings.data_path`
- `resume.resume_path`
- optionally `result_path`

This file already sets:

这个配置已经默认设置：
- `valid_settings.do_every: 1`

It also uses the same AF2-style RNA decoder and FAPE-based loss stack as the other dihedral configs.

So it is the config to use when you want to continue training and run validation every epoch.

因此，如果你希望继续训练并且每个 epoch 都做验证，直接使用这个配置文件。
## Which Config To Use

不同场景下推荐使用的配置文件。
Recommended mapping:

| Scenario | Config |
| --- | --- |
| Train from scratch on RNA | [`configs/config_vqvae_dihedral.yaml`](configs/config_vqvae_dihedral.yaml) |
| Continue training from an existing checkpoint | [`configs/config_vqvae_dihedral_continue.yaml`](configs/config_vqvae_dihedral_continue.yaml) |
| Continue training and validate every epoch | [`configs/config_vqvae_dihedral_continue_val.yaml`](configs/config_vqvae_dihedral_continue_val.yaml) |

## Training Commands

训练命令。
First configure Accelerate once:

首先执行一次 Accelerate 初始化配置：

```bash
accelerate config
```

### Train From Scratch

```bash
accelerate launch --mixed_precision=bf16 train.py \
  --config_path configs/config_vqvae_dihedral.yaml
```

### Continue Training

```bash
accelerate launch --mixed_precision=bf16 train.py \
  --config_path configs/config_vqvae_dihedral_continue.yaml
```

### Continue Training and Validate Every Epoch

```bash
accelerate launch --mixed_precision=bf16 train.py \
  --config_path configs/config_vqvae_dihedral_continue_val.yaml
```

Example single-GPU command:

单卡示例命令：
```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train.py --config_path configs/config_vqvae_dihedral.yaml
```

Example with log capture:

带日志保存的示例命令：
```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train.py --config_path configs/config_vqvae_dihedral.yaml 2>&1 | tee logs/dihedral_$(date +%F_%H-%M-%S).log
```

Memory note:

显存说明：
- the RNA AF2-style decoder plus RNA FAPE supervision can be tight on a 24 GB RTX 3090
- if you hit CUDA OOM, reduce `train_settings.batch_size` first
- if needed, reduce `num_layer` in [`configs/config_rna_af2_decoder.yaml`](configs/config_rna_af2_decoder.yaml)

## What Changed For RNA

相对蛋白版本，RNA 版本的主要改动如下。
Compared with the original protein-oriented code path, the RNA version changes the following parts.

### 1. RNA modality switch in the trainer

1. 在训练入口中增加 RNA 模态分支。
Training now dispatches by `data_modality: rna` and can optionally estimate an RNA rigid template from training H5 files.

Relevant code:

- modality resolution and RNA template injection: [`train.py`](train.py)
- training/validation entry loops: [`train.py`](train.py)

### 2. RNA-specific dataset and H5 format

2. 新增 RNA 专用数据集与 HDF5 格式。
The RNA loader reads:

- `C4p_C1p_N_coord` as the 3-atom rigid-body representation
- `P_O5p_C5p_C4p_C3p_O3p_coord` as auxiliary backbone coordinates for dihedral features

Relevant code:

- RNA H5 loading: [`data/dataset.py`](data/dataset.py)
- RNA dataset class: [`data/dataset.py`](data/dataset.py)
- trainer-side H5 requirement for train/val splits: [`data/dataset.py`](data/dataset.py)

### 3. RNA encoder feature set

3. 使用 RNA 专用编码器特征。
The protein encoder config is replaced by [`configs/config_gcpnet_encoder_rna.yaml`](configs/config_gcpnet_encoder_rna.yaml), which uses:

- `representation: C1P`
- `rna_base_one_hot`
- `sequence_positional_encoding`
- `rna_purine_pyrimidine`
- `rna_backbone_dihedrals_sincos`
- `rna_orientation`

Relevant code:

- RNA encoder config: [`configs/config_gcpnet_encoder_rna.yaml`](configs/config_gcpnet_encoder_rna.yaml)
- RNA feature dispatch in model preparation: [`models/super_model.py`](models/super_model.py)
- RNA scalar/vector node feature implementation: [`models/gcpnet/features/node_features.py`](models/gcpnet/features/node_features.py)

### 4. RNA backbone dihedral features

4. 新增 RNA backbone 二面角特征。
RNA dihedrals are computed from `x.rna_backbone_coords` with atom order:

- `[P, O5', C5', C4', C3', O3']`

Per residue, the encoder receives sin/cos features for:

- `alpha`
- `beta`
- `gamma`
- `delta`
- `epsilon`
- `zeta`

Relevant code:

- dihedral implementation: [`models/gcpnet/features/node_features.py`](models/gcpnet/features/node_features.py)

### 5. AF2-style RNA structure decoder

5. 使用 AlphaFold2 风格的 RNA 结构解码器。
The active RNA scratch config now uses an AF2-inspired structure decoder rather than the original geometric decoder head.

This decoder:

- takes VQ output tokens `(B, L, D)` as the only sequence input
- initializes one backbone rigid frame per residue
- applies invariant point attention without pair representation
- predicts relative rigid updates for each structure block
- reconstructs RNA local 3-atom rigid coordinates `[C4', C1', N1/N9]`
- runs `num_layer` structure blocks, with optional shared weights controlled by `share_weights` in [`configs/config_rna_af2_decoder.yaml`](configs/config_rna_af2_decoder.yaml)

Important clarification:

- `share_weights=True` means weight tying across structure blocks
- this is not AlphaFold2 recycling

Relevant code:

- AF2-style decoder config: [`configs/config_rna_af2_decoder.yaml`](configs/config_rna_af2_decoder.yaml)
- AF2-style decoder implementation: [`models/rna_af2_decoder.py`](models/rna_af2_decoder.py)
- RNA template constants: [`models/gcpnet/layers/structure_proj.py`](models/gcpnet/layers/structure_proj.py)
- runtime template replacement with estimated RNA template: [`train.py`](train.py)

### 6. RNA FAPE loss and AF2-style auxiliary supervision

6. 损失函数切换为 RNA FAPE 与 AF2 风格辅助监督。
The active dihedral config path no longer trains with the old `MSE + backbone distance + backbone direction` reconstruction objective.

Current reconstruction loss for the AF2-style RNA decoder includes:

- final backbone FAPE loss
- auxiliary backbone FAPE loss over all intermediate structure-block trajectories
- optional TikTok padding loss
- VQ loss is still added separately at the training step level

Training/validation monitoring in the active RNA path is now loss-first:

- best checkpoints are selected by validation loss
- epoch logs emphasize `loss`, `rec_loss`, `final_fape`, `aux_fape`, and `vq_loss`
- the old MAE / RMSD / TM-score / GDT-TS training-time metrics are no longer the primary monitoring path

Memory note:

- the current RNA FAPE implementation is chunked for memory safety
- this is intentional for single-GPU 24 GB class cards such as RTX 3090 / 3080 setups

Here, the AF2-style auxiliary loss means:

- every intermediate frame trajectory produced by the RNA structure decoder is supervised with backbone FAPE
- this is analogous to AlphaFold2 trajectory supervision
- it is not the protein-specific torsion / chi / violation loss stack

Relevant code:

- reconstruction loss assembly: [`utils/custom_losses.py`](utils/custom_losses.py)
- AF2-style decoder outputs consumed by the loss: [`models/rna_af2_decoder.py`](models/rna_af2_decoder.py)
- dihedral configs enabling `final_fape` and `aux_fape`:
  - [`configs/config_vqvae_dihedral.yaml`](configs/config_vqvae_dihedral.yaml)
  - [`configs/config_vqvae_dihedral_continue.yaml`](configs/config_vqvae_dihedral_continue.yaml)
  - [`configs/config_vqvae_dihedral_continue_val.yaml`](configs/config_vqvae_dihedral_continue_val.yaml)

### 7. Validation exports and PyMOL playback

7. 验证导出与 PyMOL 回放。
For server-side training, the repository now assumes offline visualization rather than a live GUI window.

The intended workflow is:

- training periodically writes validation prediction / label PDBs
- the exported structures stay in the original coordinate frame
- the label PDB is no longer rewritten after Kabsch alignment
- you can sync those files back to a local machine and inspect the same validation sample across epochs in PyMOL

## Minimal End-to-End Workflow

最小可运行全流程示例。
```bash
# 1) split raw RNA PDBs
python -u data/rna_homology_split.py \
  --input_root /path/to/raw_bgsu_pdb_root \
  --extensions .pdb \
  --output_dir /path/to/bgsu_pdb_homology_split_c80 \
  --identity 0.80 \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --threads 16 \
  --materialize symlink \
  --api_sleep_s 0.15 \
  --cache_flush_every 20 \
  --rcsb_cache_path /path/to/rcsb_cache_bgsu.json \
  --overwrite

# 2) convert split PDBs to H5
for split in train val test; do
  python -u data/rna_pdb_to_h5.py \
    --data /path/to/bgsu_pdb_homology_split_c80/splits/${split} \
    --save_path /path/to/bgsu_h5_homology_split_c80/splits/${split} \
    --max_len 511 \
    --min_len 11 \
    --max_missing_ratio 0.30 \
    --gap_threshold 5 \
    --max_workers 16
done

# 3) train from scratch
accelerate launch --mixed_precision=bf16 train.py \
  --config_path configs/config_vqvae_dihedral.yaml
```

Or, if you are continuing and want validation every epoch:

如果你是继续训练，并希望每个 epoch 都进行验证：

```bash
accelerate launch --mixed_precision=bf16 train.py \
  --config_path configs/config_vqvae_dihedral_continue_val.yaml
```

## Citation

引用信息。
If you use this repository, please cite the original GCP-VQVAE paper:

```bibtex
@article{Pourmirzaei2025gcpvqvae,
  author  = {Pourmirzaei, Mahdi and Morehead, Alex and Esmaili, Farzaneh and Ren, Jarett and Pourmirzaei, Mohammadreza and Xu, Dong},
  title   = {GCP-VQVAE: A Geometry-Complete Language for Protein 3D Structure},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.10.01.679833},
  url     = {https://www.biorxiv.org/content/10.1101/2025.10.01.679833v1}
}
```

## Acknowledgments

致谢。
This codebase builds on the original GCP-VQVAE project and its supporting libraries, especially:

- `vector-quantize-pytorch`
- `x-transformers`
- `ProteinWorkshop` and GCPNet-related components
