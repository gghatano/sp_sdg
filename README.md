# 時系列データ拡張は所定性能に必要な実被験者数を削減できるか

時系列データ拡張(data augmentation)が分類精度に与える効果を公開ベンチマーク(UCR Time Series Classification Archive)で系統的に評価し、その知見を被験者ID付きデータ(UCI HAR: スマートフォン慣性センサによる行動認識)へ展開して「**所定の分類性能に必要な実被験者数を、データ拡張でどれだけ削減できるか**」を、誤検出に頑健な評価枠組みで定量した研究の公開実装です。

**主結論(先取り)**: 拡張の効果はモデルに強く依存し、被験者数削減は本設定では統計的に確定できませんでした。見かけの削減の多くは、ラベルを壊した対照(negative control)でも同程度に再現される正則化アーティファクトです。

- 📄 **レポート(論文形式の全文)**: https://gghatano.github.io/sp_sdg/ (オフライン版 `report/dist/index.html`)
- 仕様・設計: [`docs/spec.md`](docs/spec.md)

## 研究課題

被験者ごとの計測が必要な生体・行動信号の収集は高コストです。データ拡張は少データ性能を補う代表的手法ですが、時系列では効果が手法・データ・モデルに依存することが知られています(Iwana & Uchida 2021)。本研究は次の2点を問います。

- **RQ1(精度効果)**: 時系列拡張はどの条件で分類精度を改善するか。
- **RQ2(被験者数削減)**: 所定の目標性能 τ の達成に必要な実被験者数を、拡張はどれだけ削減できるか。

## 主な結果

| # | 知見 | 確度 | 種別 |
|---|------|:---:|:---:|
| F-5 | **1D-CNN** では複数の拡張が精度を有意に改善(mixup +6.2pt, dtw +5.7pt, smote +5.1pt ほか。多重比較補正[Holm–Bonferroni]後も α=0.05 で有意) | 高 | 独自 |
| F-6 | **MiniRocket** ではどの拡張も精度を有意に改善しない(効果はモデル依存) | 高 | 独自 |
| F-7 | 拡張効果(1D-CNN)は学習データが中〜低量(25–50%)で相対的に大きい | 中 | 追試 |
| F-8 | UCI HAR で目標精度 0.90 に必要な実被験者数は拡張なしで約 8.85 名。点推定では DTW が最大削減(10.8%, 約1名)だが CI が広く確定しない | 低 | 独自 |
| F-10 | **対照 label_shuffle(元データ保持+ラベルノイズ)ですら 4.3% の見かけの削減**を示し、mixup(4.1%)・scaling(3.9%)と同等。約4%以下の削減はアーティファクトと区別できない | 高 | 独自 |
| F-11 | 「拡張が効くか」は拡張なしモデルの汎化ギャップ(過剰適合の度合い)では説明できない。効き目はモデルが拡張された変動を表現・利用できるかで決まる | 中 | 独自 |
| F-12 | 第2の被験者データ **WISDM v1.1(36名)でも削減は確定できない**(帰無を再現)。ただし最良手法の順位は再現せず(UCI HAR は DTW 最良、WISDM は SMOTE 最良)、順位は評価指標にも依存する | 低 | 独自 |

**本研究の中心的貢献は、被験者数削減の評価に対照による「対照越え」判定を課したこと**です。少データでは、ラベルを壊す・量を単純に増やすといった操作でも見かけ上 N* が下がりえます(対照 label_shuffle は「元データ保持+ラベルノイズ」の悲観的対照、oversample は純粋な量の水増し)。これらの対照を明確に超える削減は本設定のどの手法にも見られず、「データ拡張が実被験者を代替する」という主張は本設定では支持されませんでした。

詳細な統計・図表・考察は[レポート全文](https://gghatano.github.io/sp_sdg/)を参照してください。

## 評価対象の手法(用語)

**データ拡張**(学習データにのみ適用):

| 名前 | 概要 |
|---|---|
| `oversample` | 既存学習サンプルの単純複製 |
| `jitter` / `scaling` | ガウスノイズ付加 / 振幅ランダムスケーリング(ウェアラブル信号の標準的摂動) |
| `mixup` | 2サンプルの線形補間(本実装は同一クラス限定・ハードラベル) |
| `smote` | 近傍内挿による合成(本実装は生信号ベクトル空間・全クラス) |
| `dtw` | DTW 整列に基づく系列平均合成(本実装はペアワイズ簡略版) |
| `label_shuffle` | **対照(negative control)**。正しいラベルの元データを残し、複製分にだけ乱ラベルを与える(元データ保持+ラベルノイズの悲観的対照) |

**モデル**: `1D-CNN`(標準的な深層時系列分類器)/ `MiniRocket`(ROCKET 系の高速な特徴変換 + 線形分類器)。

**評価指標の記号**: **N\***(必要被験者数)= 目標精度 τ を初めて超えるのに要する学習被験者数。**削減率** = 1 − N\*(拡張) / N\*(なし)。

## セットアップ

前提: Python 3.11+、[uv](https://docs.astral.sh/uv/) 0.4+、Node.js 18+。**GPU 不要**(torch は CPU 版に固定、全実験が CPU で完結)。コマンドはリポジトリのルートで実行してください。

```bash
make setup     # uv sync + report/ の npm install
make test      # 全テスト(リーク防止ガードを含む)
```

## 結果の再現

各実験は `config/*.yaml` で条件を定義し、全 run に一意 run_id + manifest(git_commit・seed・依存チェックサム)が付きます。同一 commit・同一 config なら結果は決定的に再現されます。runner は completed の run をスキップするため、中断後は同じコマンドの再実行で再開します。

```bash
# RQ1: UCR での精度効果(F-5〜F-7)
make download                       # UCR データ取得 + checksum 記録
make phase1                         # 3データセット×7拡張×2モデル×3seeds = 126 runs
make phase2                         # 12データセット×5学習比率×… = 2520 runs

# RQ2: 被験者数削減(F-8〜F-10)。UCI HAR は初回実行時に自動取得(公式の被験者非跨り分割)
uv run python scripts/run_subject_experiment.py --config config/experiments/subject_count.yaml
uv run python scripts/run_subject_experiment.py --config config/experiments/negative_control.yaml

# RQ2 外的妥当性: WISDM v1.1(36名)での追試(F-12)。初回実行時に自動取得(CC BY、seed-0 で pool24/test12 に分割)
uv run python scripts/run_subject_experiment.py --config config/experiments/wisdm_subject_count.yaml
uv run python scripts/run_subject_experiment.py --config config/experiments/wisdm_negative_control.yaml

# 補足分析(F-11): 拡張効果と汎化ギャップの相関
uv run python scripts/analyze_gap.py --config config/experiments/h2_gap.yaml

make all-report                     # 監査 → 集計 → report/dist/index.html を自動生成
```

所要時間の目安(4コア CPU): `make phase1` 約1.5h、`make phase2` 約2h、被験者数実験は軽量版で数十分〜。目標性能 τ=0.90 は結果を見る前に [`artifacts/pre_registration.md`](artifacts/pre_registration.md) へ事前登録しています(事後設定の禁止)。**再現手順・データ前処理の補足と不定性・「エイヤッと決めた」判断は、レポートの「再現・前処理ノート」タブにまとめています。**

## やっていないこと・今後の課題

本研究の結論は以下の範囲に限定されます。これらは重要な留保であり、レポートの「限界」セクションに対応します。

- **被験者数削減は統計的に未確定**。被験者数実験の反復は各被験者数で3回のみで N* の信頼区間が広く、拡張手法の削減は none や negative control と CI が重なります。確度には反復数の増加(例: 各点10回以上)と検出力設計が必要です(最優先の未解決点)。
- **被験者データは UCI HAR(30名)と WISDM v1.1(36名)の2件**。ただし2データセットで再現できたのは「削減を確定できない(帰無)」の一点のみで、最良手法の順位は入れ替わります(UCI HAR は DTW 最良、WISDM は SMOTE 最良)。WISDM への展開は前処理を各データセットに合わせた追試(UCI HAR 非正規化 vs WISDM 窓ごと z 正規化)であり、同一枠組みの厳密再現ではありません。第3の被験者データ・別ドメインへの一般化は未検証です。加えて WISDM はクラス不均衡で、点推定の手法順位は accuracy 固有です(macro-F1 / balanced accuracy では変わりうる)。
- **拡張強度は各手法1点固定**(ratio, sigma, alpha, k 等)。「手法の効果」と「強度選択の効果」が交絡しており、強度スイープでの分離は未実施です。
- **negative control は label_shuffle 1本のみ**。純ノイズ拡張・時間シャッフル等の対照拡充は今後の課題です。
- **原論文からの実装差分**があります(mixup は同一クラス限定・ハードラベル、DTW はペアワイズ簡略版、raw-SMOTE は全クラス適用)。数値の厳密一致ではなく傾向の追試を目的としています(詳細は [`artifacts/deviations.md`](artifacts/deviations.md))。
- **目標 τ=0.90** は 6クラス行動認識では比較的容易に到達する水準で、より難しい目標では削減の様相が変わりえます。
- 1D-CNN の**ハイパーパラメータ探索は未実施**(拡張条件間は固定条件で公平に比較)。生成モデル系拡張(TimeGAN 等)や MiniRocket 無効の機序(特徴量空間分析)も未着手です。

## リポジトリ構成

| ディレクトリ | 役割 |
|---|---|
| `config/` | データセット・拡張・モデル・実験条件(YAML。実験条件はコードに埋め込まない) |
| `src/signal_aug/` | 研究コード本体(data / augmentations / models / experiments / evaluation / reporting) |
| `scripts/` | 実験実行・集計・監査・レポート生成のエントリポイント |
| `runs/` | run_id 単位の manifest / metrics / predictions / logs |
| `artifacts/` | 事前登録・知見(findings)・意思決定ログ・再現ノート等 |
| `report/` | 論文形式の静的 HTML レポート(`runs/`・`artifacts/` から自動生成) |
| `references/` | 参考文献・追試対象の定義 |
| `tests/` | unit / integration / regression |

## 方法論上の制約(リーク防止)

- test データを fit・拡張・合成元・early stopping・ハイパラ/手法選択のいずれにも使用しない([`docs/spec.md`](docs/spec.md) §8)。
- 被験者ID付きデータでは同一被験者を train/test に跨がせない(group split をコードで検査)。
- レポートの数値は `runs/`・`artifacts/` から自動生成し、HTML への手入力は行わない。

## 主要参考文献

- B. K. Iwana and S. Uchida, "An empirical survey of data augmentation for time series classification," *PLOS ONE*, 2021.
- A. Dempster, D. F. Schmidt, and G. I. Webb, "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification," *KDD*, 2021.
- G. Forestier, F. Petitjean, H. A. Dau, G. I. Webb, and E. Keogh, "Generating synthetic time series to augment sparse datasets," *ICDM*, 2017.
- D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, "A Public Domain Dataset for Human Activity Recognition Using Smartphones," *ESANN*, 2013.
- H. A. Dau et al., "The UCR Time Series Archive," *IEEE/CAA J. Autom. Sinica*, 2019.

全参考文献はレポートの参考文献セクション([`report/assets/data/references.json`](report/assets/data/references.json))を参照してください。データセットは UCR アーカイブおよび UCI HAR(CC BY 4.0)を利用しています。
