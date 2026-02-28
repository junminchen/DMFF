# bond_energy_examples (DMFF JAX sGNN)

这个目录提供 DMFF 原生 `dmff.sgnn` 的逐中心键贡献分析示例，默认针对 `PEG4` 的 `C-H` 键断裂趋势。

## 默认输入

- 结构：`examples/sgnn/peg4.pdb`
- 参数：`examples/sgnn/model1.pickle`
- 原子类型映射（固定，用于兼容 `model1.pickle`）：`ATYPE_INDEX = {'H': 0, 'C': 1, 'O': 2}`

## 文件

- `analyze_bond.py`
  - 单点分析某根键的局部贡献。
  - 默认自动选择第一根 C-H 键。
  - 支持 `--bond I J` 指定任意单根键（0-based）。
  - 会从参数文件自动推断网络层配置（`n_layers/sizes`）。

- `scan_bond_stretch.py`
  - 对目标键做键长扫描（近似断键趋势分析）。
  - 输出 `csv`：`Delta E_total`、`Delta E_bond_contrib`
  - 有 `matplotlib` 时输出 `png` 曲线。
  - 同样自动推断网络层配置。

## 运行

在 DMFF 仓库根目录：

```bash
cd /home/am3-peichenzhong-group/Documents/project/package/DMFF
python examples/sgnn/bond_energy_examples/analyze_bond.py
python examples/sgnn/bond_energy_examples/analyze_bond.py --bond 4 5
python examples/sgnn/bond_energy_examples/scan_bond_stretch.py --dr-min 0.0 --dr-max 1.5 --n-points 16
```

## 说明

这些结果是模型局部贡献（bond-centered contribution），可用于趋势分析，但不等同严格化学定义的 BDE 或真实反应势垒。
