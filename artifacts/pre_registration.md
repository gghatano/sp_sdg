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
