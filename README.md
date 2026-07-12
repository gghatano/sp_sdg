# 信号データ拡張・論文追試プロジェクト

公開時系列データ(UCR Time Series Classification Archive)を用いてデータ拡張手法の効果を系統的に追試し、その知見を被験者ID付きデータへ展開して「所定の分類性能に必要な実被験者数をデータ拡張でどこまで削減できるか」を定量評価する研究プロジェクトです。

- 仕様: [`docs/spec.md`](docs/spec.md)
- 実行計画: [`docs/plan.md`](docs/plan.md)
- 進捗レポート: `report/dist/index.html`(オフライン閲覧可)。公開版: https://gghatano.github.io/sp_sdg/

## 対象論文・追試条件

追試対象は [`references/reproduction_targets.yaml`](references/reproduction_targets.yaml) で構造化管理し、論文横断比較は `references/paper_matrix.csv` に記録します。主要な参照文献は時系列データ拡張のサーベイ(Iwana & Uchida, PLOS ONE 2021)、MiniRocket(Dempster et al., KDD 2021)などです(レポートの参考文献セクション参照)。

## セットアップ

前提条件: Python 3.11+、[uv](https://docs.astral.sh/uv/) 0.4+、Node.js 18+。**GPU は不要**です(torch は CPU 版 wheel に固定してあり、全実験が CPU で完結します)。コマンドはすべて**リポジトリのルートディレクトリで**実行してください(config や runs を相対パスで参照するため)。

```bash
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
make download     # UCR 3データセットの取得と checksum 記録(数分・数十MB)
make phase1       # 3データセット x 7拡張 x 2モデル x 3seeds = 126 runs
```

`make phase1` の所要時間の目安は **4 コア CPU で約 2 時間**です(FordA の 1D-CNN 学習が支配的。1 run 最長で 6〜7 分)。進捗は `ls runs/manifests | wc -l` で確認できます。実験開始前にコード変更をコミットしておくと、manifest の `git_commit` から実行時のコードを一意に特定できます。

## Resume(中断・再実行)

runner は manifest の status を確認し、**completed の run をスキップ**します。中断後は同じコマンドを再実行するだけで残り(failed や実行途中で止まった running 含む)が実行されます。

```bash
uv run python scripts/run_experiment.py --config config/experiments/phase1.yaml
```

注意: resume の判定は run_id と status のみで、**config のパラメータ変更は検知しません**。実験条件を変えて再実行する場合は `--no-resume` を付けるか、該当する `runs/manifests/*.json` を削除してください。

## Audit(結果監査)

```bash
make audit        # manifest スキーマ・metrics 範囲・予測ファイル整合性の機械検査
```

## Report 生成

```bash
make all-report   # 監査 → 集計 → report/dist/index.html 生成
```

結果は `runs/` と `artifacts/` から自動生成され、HTML への手入力は行いません。`all-report` は監査が失敗すると意図的にそこで停止します(不正な結果をレポートに載せないためのゲート)。

### GitHub Pages への自動デプロイ

`.github/workflows/deploy-pages.yml` が committed データからレポートを再生成し、GitHub Pages(https://gghatano.github.io/sp_sdg/ )へ公開します。

- **ビルド検証**はどのブランチの push でも実行され、レポートが生成できることを確認します(PR チェック)。
- **公開(deploy)は既定ブランチ `main` からのみ**行われます。GitHub Pages の `github-pages` 環境が既定で非既定ブランチのデプロイを制限するためで、この挙動に合わせています。したがってレポートは PR が main にマージされた時点で公開されます。手動実行(workflow_dispatch)を main 上で行うこともできます。
- 前提: リポジトリの Settings → Pages で「Source: GitHub Actions」を設定しておく必要があります(設定済み)。フィーチャーブランチから先に公開したい場合は、Settings → Environments → github-pages の "Deployment branches" に当該ブランチを許可してください。

## 実験条件の追加方法

- **データセット追加**: `config/datasets.yaml` に登録 → 実験 config(`config/experiments/*.yaml`)の `datasets` に追加 → `make download`
- **拡張手法追加**: `src/signal_aug/augmentations/methods.py` に関数を実装して `REGISTRY` に登録 → `config/augmentations.yaml` にパラメータを定義(YAML の追記だけでは動きません)
- **モデル追加**: `src/signal_aug/models/minirocket.py` の `build_model()` に分岐を追加 → `config/models.yaml` に登録

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

- test データを fit・拡張・モデル選択等に使用しない(リーケージ = テストデータの情報が学習側へ漏れ込むことの防止、`docs/spec.md` §8)
- 被験者ID付きデータでは同一被験者を train/test に跨がせない
- UCR のサンプル数を被験者数として解釈しない
- サブエージェントの同時実行は最大 3
