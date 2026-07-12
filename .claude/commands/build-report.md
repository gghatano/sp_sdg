---
description: HTML レポートを再生成し、ペルソナレビューを実施する
---

1. `make all-report` でレポートを再生成
2. tests/regression を実行して必須 section を確認
3. 教授視点(正しさ・引用)、新規参入院生視点(わかりやすさ)、実務エンジニア視点(再現手順)でレビュー
4. 指摘はテンプレート(report/src/report.template.html)または元データの修正で反映し、HTML 直接編集はしない
