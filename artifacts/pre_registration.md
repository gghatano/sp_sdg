# 事前登録(Pre-registration) — Phase 4/5 被験者数削減評価

登録日: 2026-07-12(Phase 4 の被験者数学習曲線を **1 run も実行する前**に登録)

本ファイルは spec §8「目標性能の事後設定の禁止」および §5「目標性能事前登録」を満たすため、被験者数削減評価の判断基準を実験結果を見る前に固定するものである。登録後の変更は decision_log.md に理由とともに記録し、変更後の結果は「事後解析」として明示する。

## 対象データ

- データセット: UCI HAR(主対象)。ライセンス CC BY 4.0(data/metadata/uci_har.json に記録)
- 固定 held-out test: 公式 test の 9 被験者(pool と被験者非跨り)
- 学習プール: 公式 train の 21 被験者

## 主要仮説

H(削減): Phase 2 で 1D-CNN において有意だった拡張(mixup, dtw, smote, oversample, scaling)は、目標性能の達成に必要な**実被験者数を削減**する。

## 事前登録した目標性能

- 指標: **test accuracy**
- 目標値: **0.90**
- 根拠: Anguita et al. (2013, ESANN) が 561 次元特徴 + SVM で約 0.90 を報告。生信号 + 深層モデルは 0.90–0.95 が相場。結果に合わせて調整していない、文献由来の丸い値。
- 副次指標: macro-F1(参考記録のみ。主判定は accuracy)

## 評価手続き(事前確定)

1. 被験者数 N ∈ {3, 6, 9, 12, 15, 18, 21} について、各 N で 3 回(異なる被験者部分集合、seed 制御)学習し、固定 test 精度を測る
2. 手法ごとに学習曲線(精度 対 被験者数)を描く
3. 目標 0.90 を最初に超える被験者数 N*(手法)を、曲線の線形補間で推定
4. **削減率 = 1 − N*(aug) / N*(none)**、等価実被験者数 = N*(none) − N*(aug)
5. 3 反復の分散から bootstrap で N* の信頼区間を付す
6. negative control: ラベルを破壊した拡張(将来 P5 で追加)が削減を示さないことを確認

## 事前に定めた停止・注意条件

- baseline(none)が 21 被験者でも 0.90 に届かない場合、目標値の妥当性を再検討し decision_log に記録(結果を見てからの下方修正はしない。上限確認のみ)
- 被験者部分集合の選択は seed 固定で再現可能にする
- 拡張は各被験者部分集合の学習データにのみ適用し、test には一切適用しない

---

# 事前登録(追加) — WISDM v1.1 被験者数削減の再現(issue #21 DS-1)

登録日: 2026-07-18(WISDM の被験者数 run を **1 run も実行する前**に登録)

UCI HAR 単一データセットに立脚する RQ2 の結論(F-8〜F-10)の外的妥当性を確認するため、第2の被験者ID付きデータ WISDM v1.1 で被験者数削減を再現する。UCI HAR の登録内容を流用せず、本データ用に独立して固定する。

## 対象データ

- データセット: WISDM v1.1(Kwapisz et al. 2011)。ライセンス CC BY 4.0(data/metadata/wisdm.json に記録)
- 3チャネル(スマホ加速度 x/y/z)、6活動(Walking/Jogging/Upstairs/Downstairs/Sitting/Standing)、約20Hz
- 公式 split が無いため、seed-0 シャッフルで **12 被験者を固定 held-out test**(pool と非跨り)、残り **24 被験者を学習プール**とする
- 前処理: (subject,activity) 連続区間内で 200 サンプル(約10秒)非重複窓、窓ごと・チャネルごとに z 正規化(リーク無し)

## 事前登録した目標性能

- 指標: **test accuracy**
- 目標値: **0.80**
- 根拠: WISDM v1.1 は手作り特徴 + 被験者混在評価で ~0.90+ の文献値(Kwapisz et al.)だが、生信号 CNN + **被験者分離**評価では相場が下がる。天井/床を避け学習曲線の最急部を狙う round 値として 0.80 を採用。**UCI HAR の 0.90 は意図的に流用しない**(issue #21)。結果に合わせた調整ではない。

## 評価手続き(事前確定)

UCI HAR と同一。被験者数 N ∈ {3,6,9,12,15,18,21,24} で各3反復、目標 0.80 を最初に超える N* を線形補間で推定、削減率 = 1 − N*(aug)/N*(none)、bootstrap で CI。

## 事前に定めた停止・注意条件

- baseline(none)が 24 被験者でも 0.80 に届かない場合、目標値の妥当性を再検討し decision_log に記録(下方修正はしない。上限確認のみ)
- 反復3回のため N* の CI は広い見込み。検出力確保(反復増)は issue #7 候補1 の後続課題として据え置く
- negative control(label_shuffle)は UCI HAR と同様に別 config で追試予定

---

# 事前登録(追加) — PAMAP2 被験者数削減の横断追加(issue #21 DS-2)

登録日: 2026-07-19(PAMAP2 の被験者数グリッドを **1 run も実行する前**、かつ target 数値を算出する前に登録)

横断比較「拡張による必要実被験者数の削減率 vs 母集団被験者数」の**少 N 端点(~7-9名)**として第3の被験者ID付きデータ PAMAP2 を追加する。UCI HAR(0.90)/ WISDM(0.80)の target は**流用しない**。本データ用に独立して手続きを固定する。設計の全体像は `artifacts/ds2_design.md`。

## 対象データ

- データセット: PAMAP2 Physical Activity Monitoring(Reiss & Stricker 2012)。ライセンス **CC BY 4.0**、DOI:10.24432/C5NW2H(data/metadata/pamap2.json に記録)。archive sha256=76b3580b…add352
- チャネル: **加速度 ±16g のみ 9ch**(3 IMU=hand/chest/ankle × x/y/z)。orientation(無効)・HR・±6g・ジャイロ・磁気は除外(WISDM/UCI HAR の加速度ドメインと整合、±16g で飽和回避)
- **12 protocol 活動**のみ(transient=0 と optional 6活動は除外)
- 100Hz → **33.3Hz**(stride-3 間引き)、**168サンプル(~5.0s)非重複窓**、(subject,activity) 連続区間内、窓ごと・チャネルごと z 正規化(リーク無し)、選択チャネルに NaN を含む窓は破棄
- **test = 固定 held-out K=3 被験者**(seed-0 シャッフル、WISDM と同手続き)。pool = 残り被験者。**subject 109 の除外/保持は精度ではなく実データの活動サポート(データ有無)で機械的に決定**(下記ルール、事前登録可)

## subject 109 の機械的除外ルール(結果精度に非依存)

判定基準(結果を見る前に固定): **「protocol 12活動のうち欠ける数が他被験者と比べ著しい、または総窓数が極端に少ない被験者は除外する」**。この判定は per-subject 活動サポート(窓数)のみに依存し、モデル精度には一切依存しない。

実測(data/metadata/pamap2.json の per_subject_activity_support、window=168/33.3Hz):subject 109 は**総窓数 10・12活動中 1活動のみ(rope_jumping のみ)**、他被験者(101–108)の 247–301 窓・8–12 活動と比べ桁違いに欠損(生ファイルも 3.9MB 対 118–207MB)。ルールにより **109 を除外**。→ 残り 8 被験者、test K=3 = **[103,104,105]**、pool 5 = **[101,102,106,107,108]**、**subject_counts = {2,3,4,5}**。

注意(事前記録): 除外後、**rope_jumping(label 11)は極めて希少**(pool の 101 に1窓・102 に3窓のみ、test [103,104,105] には 0 窓)。macro-F1 は test に存在する 11 活動上で実効的に計算され、rope_jumping はクラス欠落として macro-F1 を不安定化しうる。少 N でのクラス欠落は反復増(repeats=5)+ CI 提示で緩和し、それでも不安定なら停止条件として記録する。

## 主指標と target 選択ルール(数値ではなく手続き)

- **主指標 = macro-F1**(PAMAP2 は 12活動でクラス不均衡=lying/ironing 長・rope_jumping/running/walking 短。#23 V2 の教訓で不均衡では両指標併記)。**accuracy を副次併記**。横断図は両指標版を用意
- **target 選択ルール**: **「full-pool(N = pool 最大 = 5, aug = none)の held-out test macro-F1 の平均から 0.05 を引き、0.05 刻みで切り捨てた値」を target とする。**
  - 例: full-pool none baseline が 0.72 なら target = 0.65、0.83 なら 0.75。
  - このルールは **none baseline のみ**に依存し、拡張手法の比較結果に依存しない(手法有利化の余地なし)。曲線の急峻部に target を置き天井/床を避ける。
  - full-pool none を測る run は「target 決定用の登録済み手続き」であり、**target 確定前に他手法(oversample/scaling/mixup/dtw/smote/label_shuffle)の結果を一切参照しない**。
  - UCI HAR 0.90 / WISDM 0.80 は**流用しない**。

### 確定 target(baseline 実測後、ルール適用)

full-pool none baseline(N=5, aug=none, 5反復、seed 0-4、git_dirty=false)の held-out test 実測: **macro-F1 平均 = 0.7656**(sd 0.0369、min 0.706 / max 0.799)、accuracy 平均 = 0.7958(sd 0.0176)。

登録ルール適用: `floor_0.05(0.7656 − 0.05) = floor_0.05(0.7156)` = **target = 0.70(macro_f1)**。target 上のマージン +0.066(>0.05、floor 付近ではなく健全)。ルールは 766f803(数値算出前)でコミット済み、none baseline のみに依存し拡張手法の結果に非依存。config の target_value=0.70 は本節確定後に記入(実験前コミット)。

## 評価手続き(事前確定)

1. subject_count N ∈ {2,3,4,5} について、各 N で **5 反復**(異なる被験者部分集合、seed 制御。N=5 は pool 全員のため反復は seed の学習ゆらぎのみ)学習し、固定 test の macro-F1 / accuracy を測る
2. 手法ごとに学習曲線(macro-F1 対 被験者数)を描く
3. 上記ルールで確定した target を最初に超える N*(手法)を線形補間で推定
4. **削減率 = 1 − N*(aug)/N*(none)**(無次元。target が UCI HAR/WISDM と異なっても横断比較可)。等価節約被験者数はスケール依存のため参考記載のみ
5. 5 反復の分散から bootstrap で N* の CI を付す
6. negative control(**label_shuffle**)が削減を示さないことを確認。純量水増しの基準は **oversample**

## 事前に定めた停止・注意条件

- **N* 推定不能**: pool≤5 で N 格子が {2..5} と粗く、曲線が寝て target 交差点が単一に定まらない/外挿になる場合は「N* 推定不能」を明記し、削減率は算出せず**定性記述に格下げ**(過大主張回避)
- full-pool none がルール適用後に floor 付近(target 差<0.05 を確保できない)なら N* 推定不能として記録
- クラス欠落(特に rope_jumping)による macro-F1 不安定は反復増 + CI で緩和、それでも不安定なら停止条件として記録
- 被験者 8名(除外後)は「10名以上が望ましい」下限付近。DS-2 は**主張を張る主対象ではなく横断の少 N 端点(記述用)**という位置づけ。3データ=3点では回帰・相関は出さず「点 + CI + 定性考察」に留める
- 拡張は各被験者部分集合の学習データにのみ適用し、test には一切適用しない

---

# 事前登録(追加) — WESAD 被験者数削減の外的妥当性(信号種軸, issue #21 DS-3)

登録日: 2026-07-20(WESAD の被験者数グリッドを **1 run も実行する前**、かつ target 数値を算出する前に登録)

これまでの帰結「拡張による必要実被験者数の削減は帰無」は **HAR(運動信号・行動認識)3データ**(UCI HAR 30名 / WISDM 36名 / PAMAP2 ~9名)に立脚する。本節はこれを **信号種を運動信号から非HAR の生理信号(情動・ストレス)に変えても成り立つか**で検証する。DS-2(母集団サイズ軸)とは独立の**信号種軸**の外的妥当性。設計の全体像は `artifacts/ds3_design.md`。UCI HAR(0.90)/ WISDM(0.80)/ PAMAP2(0.70)の target は**流用しない**。本データ用に独立して手続きを固定する。

## 対象データ

- データセット: WESAD(Schmidt et al. 2018, ICMI)。**ライセンス = 学術・非商用 + 帰属表示(CC BY ではない)・再配布不可**、DOI:10.1145/3242969.3242985。生データは非コミット、`data/metadata/wesad.json` に checksum・実在被験者ID・クラス別窓数のみ記録。archive sha256=5e15d260…8fd71c(著者配布 Sciebo zip、2,249,444,501 bytes。動的生成 zip のため archive hash は再現しない可能性があり、真の再現アンカーは窓 checksum)
- チャネル: **胸 RespiBAN 生理 5ch = ECG/EDA/EMG/RESP/TEMP**。胸 ACC(運動=HAR的)・手首 E4 は除外(「非HAR 生理信号」の主張のため)
- タスク: **3クラス baseline(label 1)/ stress(2)/ amusement(3)**。label {0,4,5,6,7}(transient/meditation/ignore)は破棄し 0..2 へ写像
- 700Hz → **70Hz(反エイリアス LPF → factor-10 間引き。単純 stride は ECG/EMG 高域を折り返すため decimate を使用)**、**2100サンプル(~30s)非重複窓**、(subject,label) 連続区間内、窓ごと・チャネルごと z 正規化(リーク無し)、選択チャネルに NaN を含む窓は破棄
- **test = 固定 held-out K=5 被験者**(seed-0 シャッフル、WISDM/PAMAP2 と同手続き)。pool = 残り 10名。**実測**: 15名 [S2–S17、S1(パイロット)と S12 は配布データに欠番]、seed-0 で test=[2,4,5,13,14]・pool=[3,6,7,8,9,10,11,15,16,17]。**subject_counts = {2,3,4,5,6,7,8,9,10}**

## クラス不均衡と窓数(実測、結果精度に非依存のデータ記述)

`data/metadata/wesad.json`(window=2100/70Hz)実測: 被験者あたり総窓数 ~70–75(**baseline ~38–39 / stress ~20–24 / amusement 12**)。pool 728窓 / test 359窓。**amusement が少数派(全被験者で 12窓/名)**で、小 N でのサポートが薄い(N=2 で train amusement ~24窓)。→ **主指標 = macro-F1**(#23 V2 の教訓で不均衡は両指標併記)、**accuracy 併記**。30秒非重複でも被験者あたり ~70窓確保でき、amusement も各被験者に必ず存在するため、当初懸念した窓不足による打ち切りリスクは相対的に低い(それでも少数派の跳ねは反復5+CI で緩和)。

## 主指標と target 選択ルール(数値ではなく手続き)

- **主指標 = macro-F1**、**accuracy を副次併記**。横断図は両指標版を用意
- **target 選択ルール**: **「full-pool(N = pool 最大 = 10, aug = none, 5反復)の held-out test macro-F1 の平均から 0.05 を引き、0.05 刻みで切り捨てた値」を target とする。**
  - このルールは **none baseline のみ**に依存し、拡張手法(oversample/scaling/mixup/dtw/smote/label_shuffle)の結果に依存しない(手法有利化の余地なし)。曲線急峻部に target を置き天井/床を回避。
  - full-pool none を測る run は「target 決定用の登録済み手続き」であり、**target 確定前に他手法の結果を一切参照しない**。
  - **UCI HAR 0.90 / WISDM 0.80 / PAMAP2 0.70 は流用しない。**

### 確定 target(baseline 実測後、ルール適用)

full-pool none baseline(N=10, aug=none, 5反復、seed 0-4、git_dirty=false)の held-out test 実測: **macro-F1 平均 = 0.4835**(sd 0.0218、5反復: 0.484/0.453/0.507/0.473/0.500)、**accuracy 平均 = 0.5053**。

登録ルール適用: `floor_0.05(0.4835 − 0.05) = floor_0.05(0.4335)` = **target = 0.40(macro_f1)**。target 上のマージン +0.084(>0.05、floor 付近ではなく健全)。ルールは b4c0b34(数値算出前)でコミット済み、none baseline のみに依存し拡張手法の結果に非依存。config の target_value=0.40 は本節確定後に記入(実験前コミット)。

**重要(near-chance の明示)**: この 3クラス full-pool none baseline は **near-chance** である(accuracy ~0.50 は多数派クラス率 0.54 を下回る。macro-F1 0.48 は多数派 baseline 予測の macro-F1 より高く、モデルは弱いながら学習はしている)。生信号 5ch + 1D-CNN + **窓ごと z 正規化**を生理信号にそのまま流用したことが弱い動作点の一因の疑いがある(z 正規化は EDA/TEMP の絶対レベルというストレス識別の主要手がかりを消す。judgment_calls J-WESAD-NORM / deviations D-WESAD に記録)。ユーザー判断で **HAR 3データとの比較可能性のため枠組み(前処理・モデル・評価)は一切変更せず、事前登録どおり 3クラス・前処理変更なしで完遂**する。この near-chance は「効果は信号種依存(生理信号では動作点が低い)」という DS-3 の結論の一部として誠実に扱い、target を下げ過ぎたのではなくルール適用結果である。near-chance baseline 上で N* 交差が定まらない/左側打ち切りになる場合は削減率を算出せず「N* 推定不能」を明記する(過大主張回避)。

## 評価手続き(事前確定)

1. subject_count N ∈ {2,3,4,5,6,7,8,9,10} について、各 N で **5 反復**(異なる被験者部分集合、seed 制御。N=10 は pool 全員のため反復は seed の学習ゆらぎのみ)学習し、固定 test の macro-F1 / accuracy を測る
2. 手法ごとに学習曲線(macro-F1 対 被験者数)を描く
3. 上記ルールで確定した target を最初に超える N*(手法)を線形補間で推定
4. **削減率 = 1 − N*(aug)/N*(none)**(無次元。target が HAR 3データと異なっても横断比較可)。等価節約被験者数はスケール依存のため参考記載のみ
5. 5 反復の分散から bootstrap で N* の CI を付す
6. negative control(**label_shuffle**)が削減を示さないことを確認。純量水増しの基準は **oversample**

## 事前に定めた停止・注意条件

- **N* 推定不能**: 曲線が寝て target 交差点が単一に定まらない/外挿になる、または full-pool none がルール適用後に floor 付近(target 差<0.05 を確保できない)なら「N* 推定不能」を明記し、削減率は算出せず**定性記述に格下げ**(過大主張回避)
- 少数派 amusement のクラス欠落/不安定による macro-F1 の跳ねは反復5 + CI で緩和、それでも不安定なら停止条件として記録
- **§6.5 HAR 横断図には同列で載せない**: WESAD は信号種(生理 vs 運動)・タスク(情動 vs 行動)・クラス数(3 vs 6/12)・チャネル数(5)が HAR 3データと異なる。信号種で層別した別表/別図で提示し、横断図に載せる場合は交絡を図注で厳格注記・別マーカー。因果・外挿はしない(4点・広 CI・多交絡)
- 拡張は各被験者部分集合の学習データにのみ適用し、test には一切適用しない
