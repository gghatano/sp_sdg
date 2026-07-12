# CLAUDE.md

信号データ拡張・論文追試プロジェクトの作業規約。詳細仕様は `docs/spec.md`、実行計画は `docs/plan.md`。

## プロジェクト目的

公開信号データ(UCR)でデータ拡張の効果を追試し、被験者ID付きデータで「所定性能に必要な実被験者数をデータ拡張でどれだけ削減できるか」を評価する。

## Phase 順序

Phase 0(基盤)→ 1(UCR最小追試)→ 2(横断比較)→ 3(被験者データ選定)→ 4(学習曲線)→ 5(削減評価)→ 6–7(改善・統合)。
Phase 3 の調査のみ Phase 2 と並行可。停止条件(spec §11)に該当したら次 Phase へ進まない。

## 必須ルール

- **サブエージェントは最大 3 並列**(Research / Implementation / Audit の 3 レーン)
- **test データ利用禁止**: fit・拡張・合成元・early stopping・ハイパラ選択・手法選択のいずれにも test を使わない(spec §8)
- **config 駆動**: 実験条件は `config/*.yaml` のみで定義。コードへの埋め込み禁止
- **run manifest 必須**: 全 run に一意 run_id + manifest(spec §7)。同一 run_id への並行書き込み禁止
- **state / task queue 更新義務**: 作業の開始・完了時に `artifacts/state.md` と `artifacts/task_queue.yaml` を更新する。task_queue.yaml が唯一の順序管理簿
- **同一失敗の再試行上限は 2 回**。それでも失敗したら停止して記録
- **HTML への手入力禁止**: レポートは `make all-report` で `runs/`・`artifacts/` から自動生成
- **原論文追試と独自検証を区別**して記録する(reproduction_notes.md / deviations.md)

## ドキュメント品質(レポート・文書共通)

- 日本語で簡潔に。本文の簡潔さを保ち、詳細は折りたたみ(details)や付録・別ファイルへ
- 参考文献は情報系論文水準で記載(`report/assets/data/references.json` で管理)
- この領域に馴染みのないメンバーが読んでも文脈が分かるように書く
- 重要文書の作成・大幅更新時は、複数ペルソナ(例: 教授視点=正しさ・引用、新規参入院生視点=わかりやすさ・前提知識、実務エンジニア視点=再現手順)でレビューしてから確定する
- AI 出力っぽい冗長な定型句・過剰な箇条書き・不自然な網羅性を避ける

## よく使うコマンド

```
make test        # 全テスト
make smoke       # ネットワーク不要の smoke 実験(Phase 0 品質ゲート)
make download    # UCR データ取得 + checksum 記録
make phase1      # Phase 1 実験格子(resume 対応)
make all-report  # 監査 → 集計 → HTML レポート生成
make validate    # artifacts/ の整合性検査
```
