## 目標

Smoothed Particle Hydrodynamics (SPH) のGPU実装を作る。

構成としてはmassively-labs/bph-gpuに似せたい。

Massivelyを利用したライブラリ実装と、それを使った実験をする。

## 構成

- sph-gpu: ライブラリ本体
- experiments: 実験用クレート置き場
- workspace: 実験用ワークスペース
  - Rakefile