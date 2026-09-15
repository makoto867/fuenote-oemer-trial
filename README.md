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

無料プランの512MB内で試験できるよう、推論パッチを1件ずつ統合する低メモリ実行を使用します。認識アルゴリズムとMusicXML生成はoemerのままです。

## Render

- Runtime: Python 3.10（自動作成時）
- Health check path: `/health`
- Plan: Free（メモリ不足時はStarter以上へ変更）
- Region: Singapore

Build command:

```bash
pip install -r requirements.txt && pip install --no-deps oemer==0.1.8
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

`Dockerfile`は、後でDocker方式へ切り替えて比較するときのために残しています。

データベースやAPIキーは不要です。
