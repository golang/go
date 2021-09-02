#! /bin/bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

STD_PATH=src/crypto/ed25519/internal/edwards25519/field
LOCAL_PATH=curve25519/internal/field
LAST_SYNC_REF=$(cat $LOCAL_PATH/sync.checkpoint)

git fetch https://go.googlesource.com/go master

if git diff --quiet $LAST_SYNC_REF:$STD_PATH FETCH_HEAD:$STD_PATH; then
    echo "No changes."
else
    NEW_REF=$(git rev-parse FETCH_HEAD | tee $LOCAL_PATH/sync.checkpoint)
    echo "Applying changes from $LAST_SYNC_REF to $NEW_REF..."
    git diff $LAST_SYNC_REF:$STD_PATH FETCH_HEAD:$STD_PATH | \
        git apply -3 --directory=$LOCAL_PATH
fi
