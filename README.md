# 信号データ拡張・論文追試プロジェクト

公開時系列データ(UCR Time Series Classification Archive)を用いてデータ拡張手法の効果を系統的に追試し、その知見を被験者ID付きデータへ展開して「所定の分類性能に必要な実被験者数をデータ拡張でどこまで削減できるか」を定量評価する研究プロジェクトです。

- 仕様: [`docs/spec.md`](docs/spec.md)
- 実行計画: [`docs/plan.md`](docs/plan.md)
- 進捗レポート: `report/dist/index.html`(オフライン閲覧可)

## 対象論文・追試条件

追試対象は [`references/reproduction_targets.yaml`](references/reproduction_targets.yaml) で構造化管理し、論文横断比較は `references/paper_matrix.csv` に記録します。主要な参照文献は時系列データ拡張のサーベイ(Iwana & Uchida, PLOS ONE 2021)、MiniRocket(Dempster et al., KDD 2021)などです(レポートの参考文献セクション参照)。

## セットアップ

```bash
# Python 3.11+ / uv / Node.js が必要
make setup        # uv sync + report/ の npm install
```

## Smoke test(Phase 0 品質ゲート)

ネットワーク不要・1分未満で完走します。

```bash
make test         # 全テスト(unit / integration / regression)
make smoke        # 合成データでの最小実験(manifest 付き)
```

## Phase 1 実行

```bash
make download     # UCR 3データセットの取得と checksum 記録
make phase1       # 3データセット x 7拡張 x 2モデル x 3seeds = 126 runs
```

## Resume

runner は manifest の status を確認し、完了済み run を自動でスキップします。中断後は同じコマンドを再実行するだけで残りが実行されます。強制再実行は `--no-resume`。

```bash
uv run python scripts/run_experiment.py --config config/experiments/phase1.yaml
```

## Audit(結果監査)

```bash
make audit        # manifest スキーマ・metrics 範囲・予測ファイル整合性の機械検査
```

## Report 生成

```bash
make all-report   # 監査 → 集計 → report/dist/index.html 生成
```

結果は `runs/` と `artifacts/` から自動生成され、HTML への手入力は行いません。

## ディレクトリ構成

| ディレクトリ | 役割 |
|---|---|
| `config/` | データセット・拡張・モデル・実験条件(YAML、コード埋め込み禁止) |
| `src/signal_aug/` | 研究コード本体(data / augmentations / models / experiments / evaluation / reporting) |
| `data/` | raw(原データ・変更禁止)/ interim / processed / metadata(checksum 等) |
| `runs/` | manifests / metrics / predictions / logs / checkpoints(run_id 単位) |
| `artifacts/` | state.md・task_queue.yaml(唯一の順序管理簿)・decision_log.md 等 |
| `report/` | Tailwind CSS 静的 HTML レポート |
| `references/` | 論文・ノート・追試対象定義 |
| `tests/` | unit / integration / regression / fixtures |

## 制約

- test データを fit・拡張・モデル選択等に使用しない(リーケージ防止、`docs/spec.md` §8)
- 被験者ID付きデータでは同一被験者を train/test に跨がせない
- UCR のサンプル数を被験者数として解釈しない
- サブエージェントの同時実行は最大 3
