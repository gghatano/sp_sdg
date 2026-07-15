---
description: 再現・前処理ノート(再現手順/前処理の不定性/エイヤッ判断/実装差分)を最新化し、レポート「再現・前処理ノート」タブへ反映する
---

目的: 本研究を別環境で再現・移植できる状態を保つ。レポートの「再現・前処理ノート」タブは
以下の artifacts ファイルから自動生成される(HTML 直接編集は禁止)。コード・config・実験条件を
変えたら、対応するファイルを更新してからレポートを再生成する。

## 単一の情報源(これらを更新する)

- `artifacts/reproduction_steps.yaml` — 再現手順(環境・step ごとのコマンドと説明・不変条件)
- `artifacts/preprocessing_notes.yaml` — データ前処理の補足と不定性(処理 / 対象 / 内容 / 不定性)
- `artifacts/judgment_calls.yaml` — 「エイヤッと決めた」判断(決定 / 理由 / 他の選択肢 / 結論への影響 / 事前登録有無 / source)
- `artifacts/deviations.md` — 原論文・標準手法からの実装差分(箇条書き)

## いつ更新するか

- 前処理(正規化・窓化・分割・被験者選択など src/signal_aug/data/)を変えたとき → preprocessing_notes.yaml
- 実験手順・Makefile ターゲット・config を変えたとき → reproduction_steps.yaml
- 一意に定まらない設定を判断で決めた/変えたとき(目標値・反復数・手法選定・ハイパラ等) → judgment_calls.yaml と decision_log.md の両方
- 原論文の再現から外れる実装にしたとき → deviations.md(追試/独自の区別は reproduction_notes.md にも記録)

## 手順

1. 変更に対応する上記ファイルを更新する。judgment_calls は必ず sensitivity(結論への影響)を書く。事前登録済みの項目は `preregistered: true` を保ち、事後変更は decision_log.md に理由を残す。
2. `make all-report` でレポートを再生成(runs/・artifacts/ から自動生成)。
3. `uv run pytest tests/regression/test_report_sections.py -q` で必須セクション(repro-steps / repro-preprocessing / repro-judgment / repro-deviations)を確認。
4. 可能なら第3タブをヘッドレスで描画し、手順・前処理表・判断カード・差分が表示されることを目視確認。
5. 内容の正しさは複数ペルソナ(教授=正しさ/引用、新規参入院生=前提知識・わかりやすさ、実務エンジニア=そのコマンドで本当に再現できるか)でレビューしてから確定する。

## 原則

- 再現に効く「不定性」と「判断の余地」を隠さない。読者が別環境・別データへ移植する際に効く注意点を必ず sensitivity / uncertainty 欄に書く。
- 数値・結果は手入力しない。手順とコマンドは実在の Makefile ターゲット・config パスと一致させる。
