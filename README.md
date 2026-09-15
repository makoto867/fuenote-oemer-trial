# FueNote oemer trial server

FueNoteからoemerの認識精度を試すための、独立したFastAPIサーバーです。楽譜画像を受け取り、MusicXMLと音名・ドレミのJSONを返します。

## Endpoints

- `GET /` — 手動アップロード用の簡易画面
- `GET /health` — ヘルスチェック
- `POST /v1/recognize` — 画像認識（multipartの`file`）
- `POST /v1/recognize?response_format=musicxml` — MusicXMLファイルを直接返す

## Local Docker run

```bash
docker build -t fuenote-oemer-trial .
docker run --rm -p 10000:10000 fuenote-oemer-trial
```

最初の認識時だけ、oemerが公式チェックポイントをダウンロードするため長くかかります。oemerは処理中にグローバル状態を使うので、サーバーは1ワーカーかつ同時認識1件に制限しています。

## Render

- Runtime: Docker
- Health check path: `/health`
- Plan: Free（メモリ不足時はStarter以上へ変更）
- Region: Singapore

データベースやAPIキーは不要です。
