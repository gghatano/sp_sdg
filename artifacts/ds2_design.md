# DS-2 実験設計・事前登録案 — PAMAP2 を加えた「削減率 vs 母集団被験者数」横断評価

- 作成日: 2026-07-19 / レーン: Research / Design / 対応 issue: #21 DS-2(#12 H-CORE への追加証拠、#23 レッドチームの教訓を反映)
- 位置づけ: **設計・事前登録の草案**。実データ取得・ローダ実装・run は Implementation レーンが行う。本ファイルは判断材料と実装タスクを提示するもの。
- 一次情報の裏取り状況は §1 各項目に URL 併記。**概値は「概値」と明記**。

---

## 0. 狙いと defensible な主張の範囲(先に結論)

- 目的: PAMAP2(被験者 ~9名)/ UCI HAR(30名)/ WISDM v1.1(36名)の3点で「必要実被験者数の削減率」を並べ、母集団被験者数レンジで削減率がどう変わるかを**記述的に**評価する。
- **defensible な主張の上限**(#23 の教訓):
  - UCI HAR / WISDM は既に反復3回で **CI が広く帰無**(手法順位はノイズ支配)。PAMAP2 は N≤~7 でさらに CI が広くなる見込み。したがって DS-2 で言えるのは基本的に **記述的**な主張に限る。
  - 3データセット=**3点**では回帰は引けない。「点 + CI + 定性考察」に留める。傾き・相関の定量主張は不可。
  - 3データセットはタスク難度・クラス数・センサ・前処理・目標が異なり、削減率の差は**多数の交絡を含む**。「被験者数レンジが削減率を規定する」という因果主張はしない。
  - 現実的に最も defensible な結論の型: 「測定可能なレンジ(~7〜36名)では、拡張による被験者数削減率は**どのデータでも帰無と区別できず**、母集団サイズに沿った系統的トレンドは**検出されない**」= 帰無の外的妥当性の追加証拠。H-CORE(#12)へは「個体間情報の不足を個体内拡張で埋める効果は、母集団サイズを変えても確認できない」方向の証拠として接続。

---

## 1. 成果物1: PAMAP2 一次情報

出典(一次): UCI ML Repository #231 <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>(DOI:10.24432/C5NW2H)。原著論文2件:
- Reiss, A., & Stricker, D. (2012). Introducing a New Benchmarked Dataset for Activity Monitoring. *Proc. 16th Int. Symp. on Wearable Computers (ISWC 2012)*, pp. 108–109. IEEE.(データセット提示)
- Reiss, A., & Stricker, D. (2012). Creating and Benchmarking a New Dataset for Physical Activity Monitoring. *Proc. 5th Int. Conf. on PErvasive Technologies Related to Assistive Environments (PETRA 2012)*, Article 40. ACM.(ベンチマーク詳細)

### 1.1 被験者・活動
- **被験者数: 9名**(ID 101–109)。うち女性1名、左利き1名で偏りあり(裏取り: UCI ページ、Reiss & Stricker 2012)。
- **活動: 全18種**。うち **protocol 12種**(全被験者が同一プロトコルで実施)+ **optional 6種**(一部被験者のみ)。
  - protocol 12: lying(1), sitting(2), standing(3), walking(4), running(5), cycling(6), Nordic walking(7), ascending stairs(12), descending stairs(13), vacuum cleaning(16), ironing(17), rope jumping(24)。
  - optional 6: watching TV(9), computer work(10), car driving(11), folding laundry(18), house cleaning(19), playing soccer(20)。
  - **文献の標準サブセット = protocol 12活動**(裏取り: UCI activityID マッピング、PerceptionNet arXiv:1811.00170 等で 12-activity subset が慣例)。**DS-2 も 12活動 protocol サブセットを既定**とする。
- **transient = activityID 0**("other/transient"): 活動遷移区間。**除外が標準**(窓を作らない)。
- **被験者ごとの活動欠け**: 全被験者が全 protocol 活動を行ったわけではない。特に **subject 109 は optional をほぼ実施せず protocol も短く、多くの文献で除外または注意対象**(概況。裏取りは UCI ページの注記+慣例。正確な per-subject 活動有無はローダで実データ集計して確定する必要=Implementation タスク)。rope jumping(24)を欠く被験者もいる(概値)。

### 1.2 センサ・信号仕様
- **3× Colibri 無線 IMU**(装着位置: 利き手の**手首(hand)**・**胸(chest)**・利き足の**足首(ankle)**)、**各 100 Hz**。
- **心拍計 ~9 Hz**(bpm)。IMU と非同期・低レートのため **HR 列は大半が NaN**(裏取り: UCI「Missing Values: Yes」、列レイアウト)。
- **ファイル形式**: `.dat`(半角スペース区切りテキスト)、被験者ごと1ファイル(`Protocol/subjectXXX.dat`、`Optional/subjectXXX.dat`)。**全 54 列**: col1=timestamp(s)、col2=activityID、col3=heart rate(bpm)、col4–20=IMU hand、col21–37=IMU chest、col38–54=IMU ankle。
- **各 IMU 17 列**: temperature×1、**3D 加速度 ±16g ×3**、**3D 加速度 ±6g ×3**、3D ジャイロ×3、3D 磁気×3、**orientation(クォータニオン)×4**。
- **orientation 列(各 IMU 4列)は公式に無効**(「invalid in this data collection」)。**使用禁止**(裏取り: UCI ページ注記)。
- サンプル総数: **概値 約 3.85M サンプル**(生時系列)。
- **ライセンス: CC BY 4.0**(UCI 配布ページに明示。裏取り済み。data 取得時に `data/metadata/pamap2.json` へライセンス条項と DOI を保存すること)。

### 1.3 前処理上の既知の落とし穴
1. **HR の NaN**: 9 Hz サンプル&IMU との非同期で col3 は大半 NaN。**HR チャネルは既定で不使用**(前方補完も可だがリーク・非同期の扱いが増える)。
2. **無線ドロップアウト由来の NaN**: IMU 列にもデータ落ちの NaN が散発。→ **選択チャネルに NaN を含む窓は破棄**(または連続ブロック内の短い欠損のみ線形補間)。既定は「NaN を含む窓を破棄」(単純・リークなし)。
3. **orientation 4列/IMU は無効** → チャネル選択から除外必須。
4. **加速度が 16g と 6g の二重**: 同一軸を2レンジで記録。**通常はどちらか一方を採用**(6g はレンジ内で解像度が高いが飽和しうる。**16g を既定**=飽和回避、慣例)。両採用はチャネル重複で冗長。
5. **被験者ごとの活動欠け / subject 109**: N を数える際、活動が欠ける被験者は「その活動クラスのサポートが 0」。少 N・クラス欠落は macro-F1 を不安定化。**subject 109 の扱い(除外 or 保持)は要判断**(§設計 decision)。
6. **サンプリング差**: 100 Hz は UCI HAR(50 Hz)/WISDM(20 Hz)より高い。窓長・周波数の違いは横断比較の交絡。**ダウンサンプリングで窓の実時間長を揃える**のが定石(下記)。
7. **rWISDM 相当の PAMAP2 修正版**: WISDM には rWISDM(Chereshnev et al. 2023, arXiv:2305.10222)という前処理注意の修正版があるが、**PAMAP2 について同等の広く使われる「修正版データセット」は本調査では確認できず(不明)**。Zenodo に前処理済み配布(例: `zenodo.org/records/834467` PAMAP2 preprocessed v0.3.0)はあるが一次配布ではなく、前処理内容が固定されているため**追試の透明性の観点から一次 `.dat` からの自前前処理を推奨**。

### 1.4 妥当な既定前処理案(config 化する条件)
- チャネル選択(既定案A: **加速度のみ 9ch** = 3 IMU × 3軸 ±16g 加速度)。理由: WISDM(加速度3ch)/UCI HAR(加速度+ジャイロ6ch)との**加速度ドメイン整合**、モデル `cnn1d_har` の入力を単純化。代替案B: 加速度+ジャイロ 18ch(hand/chest/ankle × accel16g + gyro)で情報量重視。**チャネル数は横断比較の交絡**なので案の固定と注記が必要(§decision)。
- ダウンサンプリング: **100 Hz → 33.3 Hz**(1/3 間引き。慣例。裏取り: 複数実装が 33.3 Hz 採用)。
- 窓化: 連続 (subject, activity) ブロック内で **非重複窓**、transient(0)除外、単一ラベル。窓長は実時間で他データと近づける(例: **~5.0 s = 168 サンプル@33.3Hz**。WISDM は10s、UCI HAR は2.56s。完全一致は不可能=交絡として明記)。
- 正規化: **窓ごと・チャネルごと z 正規化**(WISDM ローダと同じくリークなし。生加速度スケール吸収)。
- group split: **被験者単位**。窓は被験者を跨がない(spec §8)。overlap ガード必須(WISDM 実装 `_split_subjects` と同型)。

---

## 2. 成果物2: DS-2 実験設計案

### 2.1 PAMAP2 被験者数削減グリッド

**最大の制約: 被験者が ~9名**しかいない。test を被験者非跨りで確保すると pool が極小になる。

- **test 確保案(推奨)**: 固定 held-out **K=3 被験者**(seed-0 シャッフルで選定、WISDM と同手続き)。**pool = 残り 6 名**(subject 109 を除外する場合 pool=5)。
  - `subject_counts` 格子 = **{2, 3, 4, 5, 6}**(pool=6 の場合)。N=6 は「pool 全員」で subset 反復が消える(seed の学習ゆらぎのみ)。小 N ほど組合せが多く反復に意味が出る。
  - `repeats = 5`(WISDM/UCI HAR は3。**PAMAP2 は N が細かく取れず点が少ないので反復を増やして各点の CI を締める**)。
- **代替案(統計安定重視)**: **3-fold の被験者 CV**(9名を 3×3、各 fold で test3/pool6)で N* を fold 平均。単一固定 test よりばらつきに頑健だが、**UCI HAR/WISDM は単一固定 test なので設計非対称**になる(横断比較の一貫性 vs 安定性のトレードオフ)。→ §decision。
- 拡張6種(既存と統一): **none / oversample / scaling / mixup / dtw / smote**。model = **cnn1d_har**(既存と同一、比較可能性のため)。
- **negative control**: **label_shuffle**(#23 の再解釈どおり「元データ保持+ラベルノイズの悲観的対照」)。純量水増しの基準は **oversample**。
- config は既存 `wisdm_subject_count.yaml` と同形(`dataset_params` に window/split_seed/n_test_subjects/normalize/downsample_hz/channels を明示)。

**run 本数(案・pool=6, N∈{2,3,4,5,6}, repeats=5)**: 5 N × 5 repeats × 6 augs = **150 runs** + negative control 5 N × 5 = 25 = **合計 175 runs**(概算。実測は Implementation)。

### 2.2 事前登録すべき項目(結果を見る前に固定)

`artifacts/pre_registration.md` に「PAMAP2 DS-2」節を**1 run 前に**追記する。UCI HAR の 0.90・WISDM の 0.80 は**流用しない**。

1. **target 選択ルール(数値ではなく手続き)**:
   **「pool 全員(N=pool 最大, aug=none)の held-out test 主指標の平均から 0.05 を引き、0.05 刻みで切り捨てた値」を target とする。**
   - 例: full-pool none baseline が 0.72 なら target = 0.65。0.83 なら 0.75。
   - この規則は **none baseline のみ**に依存し、拡張手法の比較結果に依存しない=手法有利化の余地なし。曲線の急峻部に target を置け、天井/床を避ける。
   - 補足: full-pool none を測る run は「target 決定用の登録済み手続き」であり、これ自体が結果チェリーピックにならないよう **target 確定前に他手法の結果を参照しない**ことを登録に明記。
2. **subject_count 格子**: {2,3,4,5,6}(pool=6 時)。pool を変える判断(109 除外等)は登録時に確定。
3. **test 被験者の選び方**: `split_seed=0` のソート済み ID シャッフルで先頭 K=3 を test(WISDM と同手続き、再現可能)。
4. **主指標**: PAMAP2 は 12活動でクラス不均衡(lying/ironing 長、rope jumping 短)。**macro-F1 を主指標**、accuracy を副次(#23 V2 の教訓: 不均衡では両指標併記、順位反転がありうる)。→ ただし UCI HAR/WISDM は accuracy 主。横断比較は **両指標で計算**し、PAMAP2 の見出しは macro-F1、横断図は両版を用意(§decision)。
5. **反復数・seed**: repeats と subset 選択 seed を登録。
6. **停止・注意条件**: full-pool none が target 手続きの結果 floor 付近(例 target 差<0.05 を確保できない=曲線が寝ている)なら N* 推定不能を明記(§3)。

### 2.3 横断分析(削減率 vs 母集団被験者数)の設計

- **(a) 目標が各データ独立でも比較可能にする提示**:
  - 横軸 = 母集団(pool)被験者数、縦軸 = **削減率 = 1 − N*(aug)/N*(none)**(**無次元**なので target が違っても比較可能)。
  - target は各データ独立の**同一ルール**(§2.2-1 の「full-pool none −0.05 切り捨て」)で決めることで、**target 決定手続きを3データ間で統一**(数値は違ってよい)。これが「独立でも比較可能」の要。
  - 「等価節約被験者数 N*(none)−N*(aug)」はスケール依存なので**横断の主軸にはしない**(参考記載のみ)。
- **(b) 交絡の明示(査読者 #6 対策として図注に必ず記載)**: クラス数(12 / 6 / 6)、センサ(3IMU / 加速度+ジャイロ / 加速度)、チャネル数、サンプリング&窓実時間、target 絶対値、pool 上限(6 / 21 / 24)、反復数。**これらが交絡するため削減率差の原因を被験者数に帰属できない**旨を明記。
- **(c) H-CORE(#12)接続**: 「個体内拡張が個体間情報の不足を補う」仮説に対し、母集団サイズを 3 水準に振っても削減効果が現れないなら「個体内拡張は個体間分散を代替しにくい」方向の追加証拠。逆に小母集団ほど削減率が大きい兆候があれば H-CORE と整合。**ただし 3 点・広 CI では示唆止まり**。
- **(d) defensible な主張の範囲**: **回帰・相関係数は出さない**。「3点 + 各点 bootstrap CI + 定性考察」。主張は「測定レンジで削減率は帰無と区別できず、母集団サイズに沿う系統トレンドは検出されない(または〜の弱い兆候があるが CI 内)」に限定。強い因果・外挿は明示的に否定する注記を付す。

---

## 3. 成果物3: 実装コスト・リスク棚卸し

### 実装コスト(Implementation レーン向け)
- **PAMAP2 ローダ新規**(`src/signal_aug/data/pamap2.py`、WISDM ローダと同型で実装):
  - `.dat` パース(スペース区切り、54列、Protocol/ ディレクトリの subject101–109)。
  - transient(activityID 0)除外、**12 protocol 活動**へのマッピング。
  - **チャネル選択**(orientation 4/IMU 除外必須、HR 除外、加速度±16g 9ch 既定)。
  - **NaN 処理**(選択チャネルに NaN を含む窓を破棄)。
  - **100→33.3 Hz ダウンサンプル**、連続 (subject,activity) 非重複窓、窓ごと z 正規化。
  - **被験者単位 group split**(seed-0、K=3 held-out)、**overlap ガード**、`data/metadata/pamap2.json` 書き出し(WISDM の `_record_metadata` と同じ drift 検査)。
- **dispatch 登録**: `data/subject_datasets.py` の `SUBJECT_LOADERS` に `"PAMAP2": load_pamap2` を追加(既存2件と同パターン)。
- **config 新規**: `config/experiments/pamap2_subject_count.yaml`(`wisdm_subject_count.yaml` と同形、`dataset_params` に downsample_hz/channels を追加)。
- **集計**: `reduction.py` / aggregate は既に dataset 別汎用化済み(state.md)なので `reduction_pamap2` 追加で流用可の見込み。横断図(3点)生成は新規の可能性。
- **per-subject 活動サポートの実測**: subject 109 と rope jumping 欠けの正確な有無は**実データ集計で確定**(ローダ実装時に metadata へ記録)。

### 停止条件(spec §11)に触れる懸念
- **N* が推定不能な場合**: pool≤6 で N 格子が {2..6} と粗く、full-pool none が target を(ルール上 −0.05 の余裕はあるが)満たしても、**曲線が寝ていて target 交差点が単一で定まらない/外挿になる**リスク。→ 「N* 推定不能」を結果として明記し、削減率は算出せず**定性記述に格下げ**(過大主張回避)。
- **クラス欠落による macro-F1 不安定**: 少 N で特定活動のサポートが 0 になり macro-F1 が跳ねる。→ 反復増(repeats=5)+ CI 提示で緩和、それでも不安定なら停止条件として記録。
- **被験者9名は spec/レーンの「10名以上が望ましい」下限を割る**(phase3 ノート §C4 の指摘)。→ DS-2 は「主張を張る主対象」ではなく**横断の少 N 端点(記述用)**という位置づけを明記して進める。

---

## 4. Implementation へ渡す実装タスク(箇条書き)

1. `src/signal_aug/data/pamap2.py` を WISDM ローダと同型で新規作成: `.dat` パース → transient 除外 → 12 protocol マッピング → チャネル選択(orientation/HR 除外、±16g 加速度 9ch 既定)→ NaN 窓破棄 → 100→33.3Hz ダウンサンプル → (subject,activity) 非重複窓 → 窓ごと z 正規化 → seed-0 で K=3 held-out group split(overlap ガード)→ `load_pamap2()` が `(pool, test)` SubjectSplits を返す。
2. `data/subject_datasets.py` の `SUBJECT_LOADERS` に `"PAMAP2"` を登録。
3. `data/metadata/pamap2.json` を WISDM と同じ drift 検査つきで書き出し(**ライセンス CC BY 4.0・DOI・channels・downsample_hz・per-subject 活動サポート・window checksum** を記録)。
4. `config/experiments/pamap2_subject_count.yaml` を `wisdm_subject_count.yaml` 同形で作成(`dataset_params`: window/split_seed=0/n_test_subjects=3/normalize=per_window_z/downsample_hz=33/channels、`subject_counts`・`repeats`・`augmentations`・target は §5 decision 確定後)。
5. ダウンロード + checksum 記録(`make download` 相当)、per-subject 活動サポートを実測して metadata と本ノートへ反映(subject 109・rope jumping 欠けの真値確定)。
6. reduction 集計に `reduction_pamap2` を追加、横断図(PAMAP2/UCI HAR/WISDM の削減率 vs pool 被験者数、accuracy 版と macro-F1 版)を生成。
7. `judgment_calls.yaml`(J-PAMAP2-*)・`preprocessing_notes.yaml`・`reproduction_steps.yaml`・`deviations.md`(原論文 vs 実装差)を更新。
8. **実験開始前に**コード変更をコミット(CLAUDE.md)。full-pool none baseline を先に走らせ target をルールで確定 → `pre_registration.md` に「PAMAP2 DS-2」節を **1 グリッド run 前に**追記。

## 5. 統括が判断すべき設計上の選択肢

1. **test 確保方式**: (A) 単一固定 held-out K=3 / pool6(UCI HAR・WISDM と手続き対称・横断比較が一貫)vs (B) 3-fold 被験者 CV(N* が安定するが設計非対称)。**推奨=A**(横断比較の一貫性優先、反復増で安定性を補う)。要判断。
2. **subject 109 の扱い**: 除外(pool=5、N∈{2..5}、クラス欠落リスク減)vs 保持(pool=6、少 N 分解能↑)。**推奨=実データの per-subject 活動サポートを見て決定**(109 が rope jumping 等を欠くなら除外)。ただし「結果精度を見て決める」のではなく**活動サポート(データ有無)で機械的に決める**ので事前登録可。要判断。
3. **チャネル選択と主指標の横断整合**: チャネルは (A) 加速度のみ 9ch(WISDM/HAR 加速度と整合)vs (B) 加速度+ジャイロ 18ch(情報量)。主指標は PAMAP2 を macro-F1、UCI HAR/WISDM は accuracy 主だが横断は両指標で提示。**推奨=加速度9ch + 横断は両指標併記**。チャネル数差は交絡として図注明記。要判断。

---

## 参考文献(情報系水準)
- Reiss, A., & Stricker, D. (2012). Introducing a New Benchmarked Dataset for Activity Monitoring. *Proc. ISWC 2012*, pp. 108–109. IEEE.
- Reiss, A., & Stricker, D. (2012). Creating and Benchmarking a New Dataset for Physical Activity Monitoring. *Proc. PETRA 2012*, Article 40. ACM.
- データセット: Reiss, A. (2012). PAMAP2 Physical Activity Monitoring [Dataset]. UCI ML Repository. DOI:10.24432/C5NW2H. <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring>(License: CC BY 4.0)
- (前処理注意・WISDM 側の類例)Chereshnev, R., et al. (2023). rWISDM: Repaired WISDM. arXiv:2305.10222.(PAMAP2 について同等の広く使われる修正版は本調査では**確認できず=不明**)
