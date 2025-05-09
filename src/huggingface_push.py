#!/usr/bin/env python3
"""
upload_to_hf.py

手元にある学習済みパラメータファイルだけを Hugging Face Hub に
ワンコマンドでアップロードするスクリプトです。

事前準備:
  pip install huggingface_hub
  huggingface-cli login
"""

import argparse
from huggingface_hub import HfApi, HfFolder


def main():
    parser = argparse.ArgumentParser(
        description="Upload pretrained model params to Hugging Face Hub with custom repo paths"
    )
    parser.add_argument(
        "repo_id",
        help="Hugging Face repo ID (例: username/model-name)"
    )
    parser.add_argument(
        "model_file",
        help="アップロードする学習済み重みファイルのパス (例: pytorch_model.bin)"
    )
    parser.add_argument(
        "--model_repo_path",
        help="モデル重みをリポジトリ内に置くパス (デフォルト: pytorch_model.bin)",
        default="pytorch_model.bin"
    )
    parser.add_argument(
        "--config_file",
        help="（任意）config.json のパス",
        default=None
    )
    parser.add_argument(
        "--config_repo_path",
        help="config.json をリポジトリ内に置くパス (デフォルト: config.json)",
        default="config.json"
    )
    parser.add_argument(
        "--readme_file",
        help="（任意）README.md のパス",
        default=None
    )
    parser.add_argument(
        "--readme_repo_path",
        help="README.md をリポジトリ内に置くパス (デフォルト: README.md)",
        default="README.md"
    )
    args = parser.parse_args()

    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise RuntimeError("Please run `huggingface-cli login` first to authenticate.")

    # 1) リポジトリがなければ作成（exist_ok=True）
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=token
    )

    # 2) モデルファイルをアップロード
    print(f"Uploading model file: {args.model_file} -> {args.model_repo_path}")
    api.upload_file(
        path_or_fileobj=args.model_file,
        path_in_repo=args.model_repo_path,
        repo_id=args.repo_id,
        repo_type="model",
        token=token
    )

    # 3) 任意で config.json をアップロード
    if args.config_file:
        print(f"Uploading config file: {args.config_file} -> {args.config_repo_path}")
        api.upload_file(
            path_or_fileobj=args.config_file,
            path_in_repo=args.config_repo_path,
            repo_id=args.repo_id,
            repo_type="model",
            token=token
        )

    # 4) 任意で README.md をアップロード
    if args.readme_file:
        print(f"Uploading README: {args.readme_file} -> {args.readme_repo_path}")
        api.upload_file(
            path_or_fileobj=args.readme_file,
            path_in_repo=args.readme_repo_path,
            repo_id=args.repo_id,
            repo_type="model",
            token=token
        )

    print(f"\n✅ Successfully uploaded to https://huggingface.co/{args.repo_id}/resolve/main/")


if __name__ == "__main__":
    main()
