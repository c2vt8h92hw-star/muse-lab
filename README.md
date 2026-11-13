# muse-lab

Muse が **GitHub → GPUノートブック（Colab / Kaggle）** で実験を回すための実験場。
将棋AIの実践例（@hirotakaster 氏の投稿）で実働が確認された型を、自分の環境向けに整えたもの。

## 型（どう回るか）

1. **GitHub が正本** — 実験コードはすべてこのリポジトリに置く。Muse はコネクタ経由でコードを読み書きできる。
2. **ノートブックは使い捨ての実行環境** — `notebooks/` のランナーが、実行のたびに指定ブランチを clone して回す。ノートブック側にコードを貯めない。
3. **結果は `METRICS_JSON:` 行で返す** — 実験スクリプトは最後に1行のJSONで結果を要約する。Muse はノートブックの出力からこれを読んで次の手を決める。
4. **条件変更は2通り** — 軽い変更はランナーの CONFIG（引数・ブランチ）、コード変更は GitHub 側を更新して再実行。

## 構成

- `experiments/train.py` — スモーク実験（PyTorch MLP × two-moons分類）。引数: `--epochs --lr --seed --hidden --n`
- `notebooks/colab_runner.ipynb` — Google Colab 用ランナー（無料枠は T4 GPU / Pro は A100 等）
- `notebooks/kaggle_runner.ipynb` — Kaggle 用ランナー（無料GPU枠あり。要: ノートブック設定で Internet ON + GPU アクセラレータ）

## 手動で回す場合

- Colab: このリポジトリの `notebooks/colab_runner.ipynb` を Colab で開き、「ランタイム → すべてのセルを実行」
- Kaggle: ノートブックを import し、Internet と GPU を有効化して Run All

## 運用メモ

- 実験ごとにブランチを切ると並行実験が混ざらない（例: `exp/lr-sweep`）
- このリポジトリは Public。**秘密情報・認証情報・個人データは絶対に置かない**
- Colab / Kaggle ともセッションに時間制限がある。長時間学習はチェックポイント保存を前提に設計する
