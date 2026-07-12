# RESEARCH_PLAN.md

研究としての問い・仮説・評価設計を定める。作業手順は [`docs/plan.md`](docs/plan.md)、規約は [`docs/spec.md`](docs/spec.md) を参照。

## 研究の問い

**RQ1**: 時系列分類向けデータ拡張(jittering、scaling、mixup、DTW ベース合成、raw-SMOTE 等)は、学習データが少ない条件でどの程度性能を改善するか。効果はデータセット・モデルにどう依存するか。

**RQ2**: 被験者ID付き信号データにおいて、データ拡張は「実被験者の追加」をどの程度代替できるか。すなわち、所定性能の達成に必要な実被験者数を何%削減できるか。

## 仮説

- H1: 拡張の効果は学習サンプル比率が小さいほど大きい(Iwana & Uchida 2021 のサーベイ結果と整合)
- H2: 効果はデータセット依存であり、単一の拡張手法が常勝することはない
- H3: 被験者単位の学習曲線において、有効な拡張は曲線を左方シフトさせる(= 少ない被験者数で同じ性能)。ただし削減率には上限がある

## 評価設計

### Phase 1–2(UCR)

- 指標: accuracy(主)、macro-F1、balanced accuracy(seed 平均 ± 標準偏差)
- 比較: 拡張なし(none)をベースラインとした対応差。Phase 2 では Wilcoxon 符号順位検定等で有意性を確認
- 条件統制: モデル・ハイパーパラメータは拡張条件間で固定。seed はデータ分割・拡張・初期化のすべてを支配

### Phase 4–5(被験者データ)

- 被験者単位の group split(同一被験者が train/test を跨がない)
- 目標性能は Phase 4 実行前に事前登録し、事後変更しない
- 必要被験者数 N*(aug) を学習曲線から推定し、削減率 = 1 − N*(aug)/N*(none)、bootstrap で信頼区間
- negative control(ラベルシャッフル拡張等)で「見かけの改善」を検出

## 脅威と対策(validity)

| 脅威 | 対策 |
|---|---|
| データリーケージ | test 利用禁止ルール + 自動テスト(tests/unit, integration) |
| seed 依存の偶然 | 複数 seed + 分散報告 + 統計検定 |
| 実装差による追試乖離 | 原論文条件と差分を references/notes と deviations.md に明示 |
| 目標性能の事後調整 | Phase 4 前の事前登録(subject_count.yaml の target_metric) |
| UCR の「サンプル≠被験者」誤解 | レポート・文書で明示的に区別(spec §8) |

## 成果物

各 Phase の完了ごとに `report/dist/index.html` を更新。最終成果は Phase 7 で統合レポートおよび研究発表可能な形式(論文草稿相当)にまとめる。
