# Phase 3: 被験者ID付き公開信号データセット候補調査

- 作成日: 2026-07-12
- レーン: Research / Design
- **後日の実装結果(2026-07-19 追記)**: 本ノートは候補調査時点の記録で、WISDM は 2019 版(Weiss et al., 51名)を予備対象候補として検討していた。実際の被験者数削減の追試(Phase 6, F-12)では **WISDM v1.1(Kwapisz et al. 2011, 36名 = pool24 + test12)** を採用した。以下の「51名」記述は当時の候補検討時の値であり、実使用データではない。
- 目的: 「所定の分類性能に必要な実被験者数をデータ拡張でどれだけ削減できるか」(RQ2, RESEARCH_PLAN.md)の評価基盤となる、**被験者ID付き・被験者単位 group split 可能な公開信号分類データセット**を選定する。
- 前提: UCR はサンプル≠被験者のため本フェーズでは不適(spec §8)。本フェーズは調査とドキュメント化のみ(ダウンロードは行わない)。
- ライセンス方針: 研究利用可否が確認できないものは断定せず「要確認」と明記する(spec §11 停止条件「ライセンス不明」)。

---

## 1. 候補一覧(比較表)

| # | 名称 | ドメイン/タスク | 被験者数 | 信号仕様 | サンプル/規模 | ライセンス | group split | 主/予備 |
|---|---|---|---|---|---|---|---|---|
| C1 | UCI HAR (Human Activity Recognition Using Smartphones) | ウェアラブル行動認識(6クラス) | 30 | 3軸加速度+3軸ジャイロ, 50 Hz, 2.56 s窓(128点)/50%重複 | 10,299 窓インスタンス | CC BY 4.0(明示) | 可(被験者ID列 + 21/9 の被験者別公式分割) | **主対象候補** |
| C2 | WISDM Smartphone and Smartwatch Activity and Biometrics (2019) | ウェアラブル行動認識(18クラス) | 51 | スマホ+スマートウォッチの加速度+ジャイロ, 20 Hz | 生時系列(被験者×活動×3分)+区間特徴 | CC BY 4.0(明示) | 可(被験者ID付与, 各被験者は明確に分離) | **予備対象候補** |
| C3 | PTB-XL(12誘導心電図) | 生体信号 ECG 診断分類(多ラベル/5 superclass) | 18,885 patients | 12誘導 ECG, 500 Hz(100 Hz版併存), 各10 s, 16-bit/1μV | 21,837 レコード | CC BY 4.0(明示) | 可(`patient_id` 付与, 公式10-fold は患者非跨り) | 予備(ドメイン多様化用) |
| C4 | PAMAP2 Physical Activity Monitoring | ウェアラブル行動認識(18活動) | 9 | 3× IMU(手首/胸/足首)100 Hz + 心拍計 ~9 Hz | 約 3.85M サンプル(生時系列) | CC BY 4.0(明示) | 可(被験者101–109) | 参考(被験者数<10) |

---

## 2. 候補ごとの詳細

### C1. UCI HAR — Human Activity Recognition Using Smartphones(主対象候補)

- **ドメイン/タスク**: ウェアラブル(腰装着スマホ)慣性センサによる日常行動認識。6クラス(WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)の多クラス分類。
- **被験者数**: 30名(19–48歳)。
- **信号仕様**: Samsung Galaxy S II を腰に装着。3軸線形加速度+3軸角速度を 50 Hz で取得。2.56 秒(128 サンプル)固定幅スライディング窓・50%オーバーラップ。
- **サンプル/規模**: 10,299 窓インスタンス。生信号版(`RawData`)と 561 次元手作り特徴版の両方が配布される。学習曲線には生信号版を用いるのが望ましい(拡張手法 jitter/scaling 等を時系列に直接適用可能なため)。
- **ライセンス・入手元**: Creative Commons Attribution 4.0 International(CC BY 4.0)を UCI 配布ページが明示。入手元: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **group split の可否**: 可。`subject_train.txt` / `subject_test.txt` に窓ごとの被験者IDが付与され、公式には train=21名 / test=9名の**被験者非跨り分割**が用意されている。被験者数を段階的に増やす学習曲線(Phase 4/5)に直接利用できる。
- **長所**: (a) 被験者IDと被験者別公式分割が最初から整備され、group split の実装リスクが最小。(b) CC BY 4.0 で研究利用が明確。(c) 加速度+ジャイロというウェアラブル信号は追試対象 T2(Um et al. 2017 の jitter/scaling)と整合し、拡張手法の妥当性検証と直結。(d) ベースライン相場が豊富で監査時の sanity check が容易。
- **懸念**: (a) 被験者数30は学習曲線を描くには十分だが上限が中程度(WISDM の51より狭い)。(b) 6クラスは比較的易しく高精度に飽和しやすいため、少被験者条件での差を見るには train 側被験者数を細かく刻む設計が要る。(c) 561次元特徴版はリークに注意(特徴抽出が全体で行われている可能性があるため、生信号版を採用し前処理は train のみで fit する)。
- **出典**: Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. In *Proc. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013)*, Bruges, Belgium, pp. 437–442. データセット: Reyes-Ortiz et al. (2013), UCI Machine Learning Repository, DOI:10.24432/C54S4K.

### C2. WISDM Smartphone and Smartwatch Activity and Biometrics(2019)(予備対象候補)

- **ドメイン/タスク**: スマホ+スマートウォッチによる行動認識(および生体認証)。18種の日常活動の多クラス分類。
- **被験者数**: 51名(各被験者が18活動を各3分実施)。
- **信号仕様**: スマホおよびスマートウォッチ搭載の加速度計+ジャイロを 20 Hz で取得。生時系列(raw)と区間統計特徴の双方を配布。
- **サンプル/規模**: 51被験者 × 18活動 × 3分 の生時系列。窓化により多数のインスタンスを生成可能。
- **ライセンス・入手元**: CC BY 4.0(UCI 配布ページが明示)。入手元: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
- **group split の可否**: 可。被験者IDが全レコードに付与されており、被験者単位での hold-out / K-fold(被験者非跨り)が容易。被験者数51は UCI HAR より広く、学習曲線の横軸レンジを拡大できる。
- **長所**: (a) 被験者数51は候補中(HAR系)で最多で、必要被験者数推定の分解能が高い。(b) スマホ/ウォッチの2デバイス・複数センサで拡張手法の一般化検証にも使える。(c) CC BY 4.0 で明確。
- **懸念**: (a) 生データにラベル欠損・タイムスタンプ不整合が報告されており(rWISDM: Chereshnev らによる修正版報告, arXiv:2305.10222)、前処理の品質管理が必要。(b) 18クラス・20 Hz とタスク設定が UCI HAR と異なるため、両者を同一パイプラインに載せるにはローダの抽象化が要る。
- **出典**: Weiss, G. M., Yoneda, K., & Hayajneh, T. (2019). Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living. *IEEE Access*, 7, 133190–133202. データセット: Weiss (2019), UCI Machine Learning Repository, DOI:10.24432/C5HK59.

### C3. PTB-XL — 12誘導心電図データセット(予備/ドメイン多様化)

- **ドメイン/タスク**: 生体信号(ECG)。12誘導心電図の診断分類。5つの診断 superclass(NORM, MI, STTC, CD, HYP)を含む多ラベル分類が標準タスク。
- **被験者数**: 18,885 patients(`patient_id` 付与)。年齢0–95歳(中央値62)、男52%/女48%。
- **信号仕様**: 12誘導 ECG、500 Hz(100 Hz ダウンサンプル版併存)、各10秒、16-bit・1μV/LSB。
- **サンプル/規模**: 21,837 レコード(1患者が複数レコードを持つ場合あり)。
- **ライセンス・入手元**: Creative Commons Attribution 4.0 International Public License(CC BY 4.0、PhysioNet 配布ページが明示。オープンアクセスであり credentialed access ではない)。入手元: https://physionet.org/content/ptb-xl/1.0.1/ (DOI:10.13026/x4td-x982)
- **group split の可否**: 可。各レコードに `patient_id` が付き、公式に推奨される 10-fold(`strat_fold`)は同一患者が複数 fold に跨らないよう構成されている。被験者(患者)単位分割の要件を満たす。
- **長所**: (a) 患者数が桁違いに多く、被験者数学習曲線を広範囲(数十〜数千)で描ける。(b) 生体信号ドメインでの一般化検証(Phase 6/7 の別データ検証)に好適。(c) CC BY 4.0 かつ pseudonymize 済みで利用条件が明確。
- **懸念**: (a) 多ラベル問題であり、HAR系(多クラス)と評価指標・パイプラインが異なる(主対象を HAR にする場合は予備位置づけが妥当)。(b) 1患者あたりレコード数が少ない(多くは1–2件)ため、「被験者を増やす」軸は活動認識とは意味合いが異なる(患者=1系列に近い)。(c) 医療データのため臨床的妥当性の議論は本研究のスコープ外と切り分ける必要。
- **出典**: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), Article 154. DOI:10.1038/s41597-020-0495-6.

### C4. PAMAP2 — Physical Activity Monitoring(参考)

- **ドメイン/タスク**: マルチIMU+心拍によるウェアラブル行動認識。18活動の多クラス分類。
- **被験者数**: 9名(被験者ID 101–109。女性1名・左利き1名のみと偏りあり)。
- **信号仕様**: 3× Colibri 無線IMU(手首・胸・足首、各100 Hz)+ 心拍計(約9 Hz)。
- **サンプル/規模**: 約 3.85M サンプル(生時系列)。窓化により多数インスタンス化可能。
- **ライセンス・入手元**: CC BY 4.0(UCI 配布ページが明示)。入手元: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring (DOI:10.24432/C5NW2H)
- **group split の可否**: 可(被験者ID 101–109)。ただし**被験者9名は学習曲線を描く規模として不足**(タスク要件「10名以上」を満たさない)。
- **長所**: マルチセンサ・高サンプルレートで信号が豊富。ライセンス明確。
- **懸念**: 被験者数9はレーンの下限要件(10名以上が望ましい)を割り込むため、主対象・予備対象には非推奨。センサ配置多様性の補助検証にとどめる。
- **出典**: Reiss, A., & Stricker, D. (2012). Introducing a New Benchmarked Dataset for Activity Monitoring. In *Proc. 16th International Symposium on Wearable Computers (ISWC 2012)*, pp. 108–109. IEEE. データセット: Reiss (2012), UCI Machine Learning Repository, DOI:10.24432/C5NW2H.

---

## 3. 推奨

### 主対象: **C1 — UCI HAR (Human Activity Recognition Using Smartphones)**

理由:
1. **group split の即応性が最も高い**。窓ごとの被験者IDと、被験者非跨りの公式 train(21名)/test(9名)分割が最初から用意されており、Phase 4/5 の被験者単位学習曲線をリーク無く実装できる(spec §8 準拠)。
2. **ライセンスが明確(CC BY 4.0)**で、spec §11 の「ライセンス不明」停止条件に抵触しない。
3. **拡張手法との整合**。加速度+ジャイロのウェアラブル信号は追試対象 T2(Um et al. 2017, jitter/scaling)と同ドメインであり、Phase 2 で選定する 3–5 拡張手法をそのまま持ち込め、RQ2 の評価が一貫する。
4. **被験者30名は学習曲線を描くのに十分**、かつベースライン相場が豊富で監査(Audit レーン)が容易。

### 予備対象: **C2 — WISDM Smartphone and Smartwatch Activity and Biometrics (2019)**

理由:
1. **被験者51名**で主対象より横軸レンジが広く、必要被験者数推定・削減率評価の分解能を高められる。
2. **同一ドメイン(ウェアラブルHAR)**のため主対象と同一の拡張手法・評価パイプラインを流用でき、結果の一般化を同条件で確認できる(H2「効果はデータセット依存」の検証に好適)。
3. **CC BY 4.0 で利用条件が明確**。懸念(生データの品質)は前処理で管理可能。

### ドメイン多様化の予備(Phase 6/7 用): C3 — PTB-XL
- HARに偏らない生体信号(ECG)ドメインでの一般化検証用。患者ID・公式 fold で group split 可、CC BY 4.0。多ラベル問題のため評価系は別途整備が必要。

---

## 4. ライセンス・利用条件の確認状況

| データセット | ライセンス | 確認状況 |
|---|---|---|
| UCI HAR | CC BY 4.0 | 一次配布ページ(UCI)で明示・確認済み |
| WISDM (2019) | CC BY 4.0 | 一次配布ページ(UCI)で明示・確認済み |
| PTB-XL | CC BY 4.0 | 一次配布ページ(PhysioNet)で明示・確認済み。オープンアクセス(非 credentialed) |
| PAMAP2 | CC BY 4.0 | 一次配布ページ(UCI)で明示・確認済み |

いずれも CC BY 4.0(帰属表示による研究利用可)であることを一次情報で確認した。**「要確認」に留まる項目は現時点で無い**。データ取得時には各配布ページの最新ライセンス表記を再確認し、`data/metadata/` にライセンス条項と DOI を保存すること。

## 5. 出典一覧(情報系論文水準)

- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. *Proc. ESANN 2013*, pp. 437–442.
- Weiss, G. M., Yoneda, K., & Hayajneh, T. (2019). Smartphone and Smartwatch-Based Biometrics Using Activities of Daily Living. *IEEE Access*, 7, 133190–133202.
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(1), 154.
- Reiss, A., & Stricker, D. (2012). Introducing a New Benchmarked Dataset for Activity Monitoring. *Proc. ISWC 2012*, pp. 108–109. IEEE.
- (参考・品質) Chereshnev, R. et al. (2023). rWISDM: Repaired WISDM, a Public Dataset for Human Activity Recognition. arXiv:2305.10222.
</content>
</invoke>
