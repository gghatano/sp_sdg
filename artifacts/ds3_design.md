# DS-3 実験設計・事前登録案 — WESAD(非HAR 生理信号)での被験者数削減 外的妥当性

- 作成日: 2026-07-20 / レーン: Research / Design / 対応 issue: #21 DS-3(#12 H-CORE への別軸証拠、#23 レッドチームの教訓を反映)
- 位置づけ: **設計・事前登録の草案**。実データ取得・`data/wesad.py` 実装・run は Implementation レーンが行う。本ファイルは判断材料と実装タスクを提示するもの。
- 一次情報の裏取り状況は §1 各項目に URL 併記。**概値は「概値」と明記**。原論文の記述と二次情報で食い違う点は「要実データ確認」と明記した(Implementation が DL 時に真値を確定)。

---

## 0. 狙いと defensible な主張の範囲(先に結論)

- 目的: これまで HAR(UCI HAR 30名 / WISDM v1.1 36名 / PAMAP2 ~9名)で得た「**拡張による必要実被験者数の削減は帰無**」という結論が、**信号種を HAR の運動信号から非HAR の生理信号(情動・ストレス)に変えても成り立つか**を確認する。DS-2(母集団サイズ軸)とは独立の、**信号種軸**の外的妥当性。
- **defensible な主張の上限**(#23 の教訓):
  - WESAD は被験者 **15名**・被験者あたり有効窓数が少ない(§1.4 で概算 数十窓)。UCI HAR/WISDM 同様に **反復の CI が広く帰無**になる可能性が高く、言えるのは基本的に**記述的**な主張。
  - WESAD は **HAR とタスク・信号種・クラス数が異なる**(情動3クラス vs 行動6クラス、生理信号 vs 加速度)。よって §6.5 の「削減率 vs 母集団被験者数」HAR 横断図に**同列で載せない**(§2.4)。DS-3 は「**信号種を変えても削減効果は帰無か**」という別軸の証拠として、別表・別図で提示する。
  - 最も defensible な結論の型: 「非HAR の生理信号(WESAD 情動分類)でも、拡張による被験者数削減率は**帰無と区別できない**」= 帰無の外的妥当性を信号種軸で補強。H-CORE(#12)へは「個体内拡張が個体間分散を代替しにくい傾向は信号種に依らない」方向の追加証拠として接続。逆に WESAD でのみ削減が出れば「効果は信号種依存」という重要な反証になる。

---

## 1. 成果物1: WESAD 一次情報

**出典(一次)**:
- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. *Proc. 20th ACM Int. Conf. on Multimodal Interaction (ICMI 2018)*, pp. 400–408. ACM. DOI:10.1145/3242969.3242985. 論文 PDF: <https://ubi29.informatik.uni-siegen.de/usi/pdf/ubi_icmi2018.pdf>
- 著者配布ページ(readme・DL): <https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>(Sciebo 経由 zip)
- ミラー: UCI ML Repository #465 <https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection>

### 1.1 被験者・タスク
- **被験者数: 15名**(裏取り: 論文・配布ページ・UCI いずれも "15 subjects")。ファイルは被験者ごと `SX.pkl`。
- **被験者ID**: 配布データは概ね **S2–S17 の範囲**で、途中に欠番があり実データは **15名分**。欠番の通説は「**S1 はパイロット**、**もう1名(S12 と広く記載)がセンサ不具合で除外**」だが、**正確な欠番ID・除外理由は二次情報で揺れがあり要実データ確認**(Implementation が `SX.pkl` の実在ファイル集合で真値を確定し metadata に記録)。「S4 除外」とする二次記述も見たが**確度低・不採用**。
- **標準タスク(文献)**: (a) **3クラス baseline / stress / amusement**(原論文の主タスク、報告 accuracy 概値 ~93%)と (b) **binary stress vs non-stress**(原論文・追試多数、報告 F1 概値 ~95%)。どちらも WESAD 標準。※報告値は**手作り特徴 + 被験者混在/LOSO 評価**での値で、本研究の raw-CNN + 被験者非跨り分割では動作点は下がる(target 設定に利用、§2.3)。

### 1.2 センサ・信号仕様(裏取り: 論文 §3、配布 readme)
- **胸: RespiBAN Professional**。全チャネル **700 Hz**。ECG・**EDA**(皮膚電気活動)・**EMG**(筋電)・**RESP**(呼吸)・**TEMP**(体温)・**3軸 ACC**(加速度)。
- **手首: Empatica E4**。**BVP 64 Hz**・**EDA 4 Hz**・**TEMP 4 Hz**・**ACC 32 Hz**。
- **ラベル同期**: `label` 配列は **胸 700 Hz に同期**してサンプルされる(裏取り: readme「label: … sampled at 700 Hz」)。手首 E4 の各チャネルはレートが異なるため、モダリティ横断で使う場合はリサンプル整合が要る(本設計は胸単独なので不要、§2.2)。

### 1.3 プロトコル状態とラベルコード(裏取り: 論文 §3、複数二次情報)
- 状態(概値の時間):**baseline**(中立読書 概値 ~20分)/ **amusement**(コメディ動画 概値 ~392秒)/ **stress**(Trier Social Stress Test: スピーチ+暗算 概値 ~10分)/ **meditation**(誘導瞑想 概値 ~7分×2)。※論文 PDF 抽出では各 ~5分とも読めたが、広く引用される値(baseline 20分・amusement 392秒・TSST 10分)を採用し**概値**と明記。**有効窓数の見積り(§1.4)に効くため、Implementation が実 `label` 配列から状態別サンプル数を実測して確定**。
- **ラベルコード**(裏取り: readme・複数実装): **0 = transient / not defined**、**1 = baseline**、**2 = stress**、**3 = amusement**、**4 = meditation**、**5/6/7 = 無視(条件間の休憩・SSSQ 記入等)**。
- 3クラスタスク = ラベル {1,2,3} のみ採用、{0,4,5,6,7} は窓化しない。binary = stress(2)vs non-stress({1,3,4} など、meditation を non-stress に含めるかは文献で揺れ→§2.1 で明記)。

### 1.4 ファイル形式・サイズ・前処理の落とし穴
- **形式**: 被験者ごと `SX.pkl`。中身は dict(`signal` → `chest`/`wrist` → 各モダリティ配列、`label`(700Hz)、`subject`)。**Python2 pickle のため `pickle.load(f, encoding="latin1")` が必須**(既知の落とし穴。UTF-8 デフォルトだと `UnicodeDecodeError`)。
- **サイズ**: 圧縮 概値 ~2.5 GB(配布ページ)。展開はより大。DL は直リンク zip(下記ライセンス)。
- **前処理の落とし穴**:
  1. **高サンプリング 700 Hz → ダウンサンプル要**。生 700Hz を cnn1d_har にそのまま載せると窓が巨大。共通レートへ間引き必要(§2.2)。チャネルで実効帯域が違う(ECG は QRS で高域、EDA/TEMP は数 Hz 未満の緩変化)。ダウンサンプル前に**アンチエイリアス LPF** を掛けるのが定石(単純間引きは折返し歪みを生む)。
  2. **モダリティ選択(胸 vs 手首)**: 原論文の高性能は胸。手首 E4 はレート不揃い・低品質。DS-3 は**胸を既定**(§2.2)。
  3. **チャネル種別**: 胸 ACC は**運動(motion)信号 = HAR 的**。DS-3 の主眼「**非HAR 生理信号**」を濁すため **ACC は既定で除外**し、生理信号 ECG/EDA/EMG/RESP/TEMP に限定する(§2.2)。
  4. **有効窓数が少ない**(削減の分解能に直結): 3クラス採用時、被験者あたり使える時間は概値 baseline ~20分 + stress ~10分 + amusement ~6.5分 ≈ 37分。**60秒・非重複窓なら被験者あたり概値 ~36窓**、**30秒・非重複なら ~73窓**、うち **amusement(少数派)は 30秒で ~13窓/名**。→ 窓長・重複が N* 推定の分解能を左右する(§2.2, §3 停止条件)。
  5. **クラス不均衡**: baseline ≫ stress > amusement(3クラス)。binary は stress が少数派でさらに不均衡。→ **主指標は macro-F1、accuracy 併記**(#23 V2)。
  6. **transient/meditation の扱い**: 3クラスでは {0,4,5,6,7} を捨てる。捨てる区間が長い(meditation ~14分)ので、窓数を稼ぎたい誘惑があるが、**タスク定義を曖昧にしないため 3クラスは {1,2,3} 厳守**。
  7. **状態境界の transient 混入**: 状態遷移直後は生理反応が定常でない。連続 (subject, label) ブロック内でのみ窓化し、**ラベルが切り替わる境界を跨ぐ窓は作らない**(WISDM/PAMAP2 ローダと同型)。

### 1.5 ライセンス・利用規約(**HAR 3データと異なる。重要**)
- **ライセンス文(裏取り: 配布ページ)**: 「*You may use this data for scientific, non-commercial purposes, provided that you give credit to the owners when publishing any work based on this data.*」
  = **学術・非商用に限る + 帰属表示**。**CC BY 4.0 ではない**(UCI HAR/WISDM/PAMAP2 は CC BY 4.0 で商用可だが、WESAD は**非商用限定**)。
- **同意フロー**: PhysioNet のような credentialed access / DUA 署名は**不要**。配布ページ / UCI から**直リンク zip を自動 DL 可**(Implementation の DL 可否に支障なし)。
- **再配布**: ライセンスは「利用」を許すが**再配布を明示的には許諾していない**。→ **生データはリポジトリに含めない**。`data/metadata/wesad.json` に **checksum・DOI・ライセンス条項・per-subject 窓数のみ**を記録する(方針は spec §11 準拠、生データ非コミット)。
- 本研究は**学術・非商用**なので利用条件を満たす。レポート・論文で**帰属表示(Schmidt et al. 2018 引用)必須**。

---

## 2. 成果物2: DS-3 実験設計案

### 2.1 タスク選択(既定案 + 代替案)
- **既定案: 3クラス baseline / stress / amusement**。根拠:
  - 原論文の主タスクで**信号種軸の外的妥当性**に最も直球。クラス数3で HAR(6クラス)と完全一致はしないが情報量がある。
  - binary(stress vs non-stress)は報告 F1 ~95% と**天井が高く**、raw-CNN・被験者非跨りでも高めに出やすい → target を曲線急峻部に置きにくい(床/天井回避が難)。3クラス(~93% but 特徴+混在評価)は被験者非跨り raw-CNN で**より低い動作点**に落ち、削減曲線の急峻部に target を置きやすい。
- **代替案: binary stress vs non-stress**。文献での最頻タスク。3クラスの full-pool none baseline が**天井 or 床に張り付いて target 手続きが機能しない**場合の切替候補。binary の non-stress に **meditation(4)を含めるか**は要登録(既定: 含めない = {baseline(1), amusement(3)} vs stress(2)。理由: meditation は非情動・生理的に特殊で交絡)。→ §5 decision。

### 2.2 モダリティ / チャネル / ダウンサンプル(既定案)
- **既定: 胸 RespiBAN の生理信号 5ch = ECG / EDA / EMG / RESP / TEMP**。胸 ACC(運動)と手首 E4 は除外。理由: DS-3 の主眼が「**非HAR 生理信号**」であり、ACC を入れると HAR 的運動情報が混ざり主張が濁る。マルチモーダル拡張は DS-5 の主題なので DS-3 は**過度な多チャネル化を避け 5ch の defensible な既定**とする。
- **cnn1d_har 入力形**: (5ch, window) の raw 窓。WISDM(3ch)/PAMAP2(9ch)と同じ 1D-CNN 入力形式で、チャネル数のみ異なる。
- **ダウンサンプル**: **700 Hz → 概値 ~70 Hz(1/10 間引き、事前に反エイリアス LPF)** を既定案。理由: EDA/TEMP/RESP は数 Hz 未満で 70Hz で十分、ECG は QRS 微細形状は落ちるが**心拍リズム・HR 変動**は保持でき、ストレス識別の主情報は残る。**代替: 64 Hz(手首 BVP と整合)/ 100 Hz(ECG 形状保持重視・窓大)**。ダウンサンプルレートと LPF は judgment call(J-WESAD-DS)として記録。→ §5 decision。
- **窓化**: 連続 (subject, label) ブロック内で **非重複窓**、境界跨ぎ無し、単一ラベル。**窓長 既定 = 30 秒(~70Hz で 2100 サンプル)**。理由: 60秒(生理特徴の慣例)だと被験者あたり ~36窓と少なく N* 分解能が落ちる;30秒で ~73窓に増え、生理的にもストレス反応検出に十分。**窓が足りず N* 不能なら 30秒/50%重複(train 内のみ、被験者非跨りは保持)を judgment call として許容**(重複は窓相関を増やすので注記)。→ §5 decision。
- **正規化**: **窓ごと・チャネルごと z 正規化**(WISDM/PAMAP2 と同じ、リークなし)。生理信号はチャネル間スケール差が大きい(ECG mV vs EDA μS vs TEMP °C)ので必須。

### 2.3 被験者数削減グリッド(15名)
- **test 確保(既定案)**: 固定 held-out **K=5 被験者**(seed-0 シャッフル、WISDM/PAMAP2 と同手続き)。**pool = 残り 10 名**。K/N=5/15 ≈ **1/3** で WISDM(12/36)・PAMAP2(3/9)と held-out 比率が揃い、横断の手続き一貫性を保つ。
- **`subject_counts` 格子 = {2,3,4,5,6,7,8,9,10}**(pool=10)。N=10 は pool 全員で subset 反復消滅(学習ゆらぎのみ)。
- **`repeats = 5`**(PAMAP2 と同様、被験者が少なく点が粗いので各点 CI を締める。UCI HAR/WISDM は 3)。
- **拡張6種**(既存統一): none / oversample / scaling / mixup / dtw / smote。**model = cnn1d_har**(既存と同一、比較可能性)。
- **negative control**: **label_shuffle**(#23 の再解釈: 元データ保持 + ラベルノイズの悲観的対照)。純量水増しの基準は oversample。
- **run 本数(概算)**: 9 N × 5 repeats × 6 augs = 270 + label_shuffle 9×5 = 45 → **概値 ~315 runs**(N=10 は反復1のため実効はやや少)。実測は Implementation。
- **窓数の事前見積り(N* 分解能チェック)**: full-pool(N=10, 30秒非重複)で概値 ~730窓、うち amusement ~130窓。小 N(N=2)では ~146窓・amusement ~26窓 = **CNN 学習の下限に近い**。→ §3 停止条件で明示。

### 2.4 事前登録すべき項目(結果を見る前に固定)
`artifacts/pre_registration.md` に「WESAD DS-3」節を **1 グリッド run 前に**追記。**UCI HAR 0.90 / WISDM 0.80 / PAMAP2 の値は流用しない**。
1. **target 選択ルール(数値でなく手続き、既存と同一)**: 「**full-pool(N=10, aug=none, 5 repeats)の held-out test 主指標(macro-F1)の平均から 0.05 を引き、0.05 刻みで切り捨てた値**」を target とする。none baseline のみに依存 = 手法有利化の余地なし。target 確定前に他手法の結果を参照しないことを登録に明記。
2. **主指標**: **macro-F1 を主**(3クラス不均衡)、**accuracy を副次併記**(#23 V2: 不均衡では順位反転がありうる)。
3. **subject_count 格子**: {2..10}(pool=10)。K=5 held-out。
4. **test 被験者の選び方**: `split_seed=0` のソート済み ID シャッフルで先頭 K=5 を test(WISDM/PAMAP2 と同手続き、再現可能)。
5. **反復数・seed**: repeats=5 と subset 選択 seed を登録。
6. **停止・注意条件**: full-pool none baseline が target 手続きで有効マージン(≥0.05)を確保できない/曲線が寝て交差点が単一に定まらない → **N* 推定不能を明記し削減率は算出せず定性記述に格下げ**(§3)。

### 2.5 HAR 横断図(§6.5)との関係 — **同列に載せない**
- WESAD は**信号種(生理 vs 運動)・タスク(情動 vs 行動)・クラス数**が HAR 3データと異なる。§6.5 の「削減率 vs 母集団被験者数」HAR 横断図に**同じ系列として載せない**。
- 提示方法: (a) **別表/別図**で「HAR 3データ(運動信号)」と「WESAD(生理信号)」を**信号種で層別**して並置、または (b) 横断図に載せる場合は**信号種・タスク・クラス数の違いを交絡として図注で厳格に注記**し、点は別マーカー・別凡例。
- **defensible な主張**: 「測定レンジで WESAD(生理信号)でも削減率は帰無と区別できず、HAR で見た帰無は**信号種を変えても崩れない**」。**因果・外挿はしない**(3データ + WESAD の 4点、広 CI、多交絡)。

---

## 3. 成果物3: 実装コスト・リスク

### 実装コスト(Implementation レーン向け)
- **`src/signal_aug/data/wesad.py` 新規**(WISDM/PAMAP2 ローダと同型):
  - zip DL + checksum、被験者ごと `SX.pkl` を **`pickle.load(f, encoding="latin1")`** でパース。
  - 胸生理 5ch(ECG/EDA/EMG/RESP/TEMP)選択(ACC・手首は除外)。
  - ラベル {1,2,3} 抽出({0,4,5,6,7} 除外)、状態境界跨ぎ無しの (subject,label) ブロック窓化。
  - **反エイリアス LPF → 700→~70Hz ダウンサンプル**、30秒非重複窓、窓ごと z 正規化。
  - **NaN/欠損処理**(選択チャネルに NaN を含む窓は破棄。E4 は使わないので主に無いが安全側でガード)。
  - **被験者単位 group split**(seed-0、K=5 held-out)、**overlap ガード**(pool/test 被験者非跨り)。
  - `load_wesad()` が `(pool, test)` SubjectSplits を返す。
- **dispatch 登録**: `src/signal_aug/data/subject_datasets.py` の `SUBJECT_LOADERS` に `"WESAD": load_wesad` を追加。
- **`data/metadata/wesad.json`**: WISDM と同じ drift 検査つきで **ライセンス(非商用+帰属)・DOI・source・channels・sampling(700→DSレート)・window・split_seed・実在被験者ID・per-subject/クラス別窓数・window checksum** を記録。
- **config 新規**: `config/experiments/wesad_subject_count.yaml`(`wisdm_subject_count.yaml` 同形。`dataset_params`: window / split_seed=0 / n_test_subjects=5 / normalize=per_window_z / downsample_hz / channels / modality=chest)。`target_metric: macro_f1`、`target_value` は full-pool none baseline 後にルールで確定(プレースホルダ)。
- **集計**: reduction 集計に `reduction_wesad` を追加(dataset 別汎用化済みなら流用可)。**信号種層別の別表/別図**は新規の可能性。
- **judgment_calls.yaml**(J-WESAD-*: DSレート/窓長/重複/チャネル/欠番ID確定)・`preprocessing_notes.yaml`・`reproduction_steps.yaml`・`deviations.md`(原論文=特徴+混在評価 vs 本実装=raw-CNN+被験者非跨りの差)を更新。

### ライセンス / 規約リスク
- **非商用限定 + 再配布非許諾** → **生データをリポジトリに置かない**(checksum + metadata のみ)。DL スクリプトは直リンク取得(credentialed access 不要 = DL 自体は自動化可)。レポート/論文で **Schmidt et al. 2018 の帰属表示必須**。CC 系より条件が厳しい点を state / metadata に明記。

### 停止条件(spec §11)該当リスク
- **窓数不足で N* 推定不能**: 被験者 15名・小 N での窓数が CNN 学習下限に近い(§2.3)。full-pool none が **target マージン ≥0.05 を確保できない/曲線が寝る**なら N* 算出せず定性記述に格下げ(過大主張回避)。
- **クラス欠落 / 少数派 amusement の不安定**: 小 N で amusement サポートが薄く macro-F1 が跳ねる。反復5 + bootstrap CI で緩和、それでも不安定なら停止条件として記録。
- **欠番ID・状態時間の不確定**: 実 `SX.pkl` / `label` から真値を確定するまでは概値。DL 後に metadata へ実測値を記録し本ノートを追記(WISDM の実装追記と同運用)。
- **ライセンス**: 非商用は満たすが**再配布不可** → 生データ非コミット厳守(これ自体は停止条件でなく運用制約)。

---

## 4. Implementation へ渡す実装タスク(箇条書き)
1. `src/signal_aug/data/wesad.py` を WISDM/PAMAP2 同型で新規作成: zip DL+checksum → `SX.pkl` を `encoding="latin1"` でパース → 胸生理5ch 選択 → ラベル{1,2,3}抽出({0,4,5,6,7}除外)→ (subject,label) 境界跨ぎ無し窓化 → 反エイリアス LPF + 700→~70Hz ダウンサンプル → 30秒非重複窓 → 窓ごと z 正規化 → NaN 窓破棄 → seed-0 で K=5 held-out group split(overlap ガード)→ `load_wesad()` が `(pool,test)` SubjectSplits を返す。
2. `subject_datasets.py` の `SUBJECT_LOADERS` に `"WESAD"` を登録。
3. `data/metadata/wesad.json` を drift 検査つきで書き出し(**ライセンス=非商用+帰属・DOI・channels・DSレート・実在被験者ID・per-subject/クラス別窓数・window checksum**)。
4. `config/experiments/wesad_subject_count.yaml` を `wisdm_subject_count.yaml` 同形で作成(`dataset_params`: window/split_seed=0/n_test_subjects=5/normalize=per_window_z/downsample_hz/channels/modality=chest。`subject_counts`{2..10}・`repeats`5・`augmentations`6種・target は §5 decision 確定後)。
5. DL + checksum 記録、**実 `SX.pkl` から欠番ID・状態別サンプル数・per-subject/クラス別窓数を実測**して metadata と本ノート §1 に反映(欠番ID・状態時間の真値確定)。
6. reduction 集計に `reduction_wesad` を追加、**信号種で層別した別表/別図**(HAR 運動信号 vs WESAD 生理信号の削減率)を生成。§6.5 HAR 横断図には同列で載せない。
7. `judgment_calls.yaml`(J-WESAD-*)・`preprocessing_notes.yaml`・`reproduction_steps.yaml`・`deviations.md` を更新。
8. **実験開始前に**コード変更をコミット(CLAUDE.md)。full-pool none baseline を先に走らせ target をルールで確定 → `pre_registration.md` に「WESAD DS-3」節を **1 グリッド run 前に**追記。

## 5. 統括が判断すべき設計上の選択肢
1. **タスク**: (A) **3クラス baseline/stress/amusement** vs (B) binary stress vs non-stress。**推奨=A**(原論文主タスク、床/天井回避が容易、情報量)。B は full-pool none が天井/床に張り付く場合の切替候補(non-stress に meditation を含めるかも要登録、既定は含めない)。要判断。
2. **モダリティ/チャネル**: (A) **胸生理5ch(ECG/EDA/EMG/RESP/TEMP)** vs (B) 胸生理5ch+胸ACC(HAR的運動が混ざる) vs (C) 手首 E4。**推奨=A**(「非HAR 生理信号」の主張を濁さない、DS-5 のマルチモーダルと切り分け)。要判断。
3. **ダウンサンプル + 窓長/重複**: DSレート (A) ~70Hz 既定 / (B) 64Hz / (C) 100Hz;窓長 (A) **30秒非重複** 既定 / (B) 60秒(慣例だが窓少) / (C) 30秒50%重複(窓増だが相関増)。**推奨=~70Hz + 30秒非重複**、窓不足で N* 不能なら 50%重複を judgment call で許容。要判断。
4. **test 分割 K**: (A) **K=5 / pool10**(held-out 1/3、WISDM/PAMAP2 と手続き一貫、推奨)vs (B) K=4 / pool11(pool レンジ↑だが test 小で指標分散↑)。**推奨=A**。要判断。

---

## 参考文献(情報系水準)
- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. *Proc. 20th ACM Int. Conf. on Multimodal Interaction (ICMI 2018)*, pp. 400–408. ACM. DOI:10.1145/3242969.3242985.
- データセット: Schmidt et al. (2018). WESAD (Wearable Stress and Affect Detection) [Dataset]. UCI ML Repository #465 / 著者配布(Univ. Siegen)。<https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection> / <https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html>。License: **scientific / non-commercial use with attribution**(CC BY ではない、要帰属)。
