#! /bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: merge.sh <target branch> <source revision>"
    echo ""
    echo "example: merge.sh dev.boringcrypto master"
    echo "         merge.sh dev.boringcrypto.go1.10 go1.10.7"
    exit 1
fi

TARGET="$1"
SOURCE="$2"
WORKTREE="$(mktemp -d)"
BRANCH="boring/merge-$TARGET-$(date +%Y%m%d%H%M%S)"

git fetch
git worktree add --track -b "$BRANCH" "$WORKTREE" "origin/$TARGET"

cd "$WORKTREE"
export GIT_GOFMT_HOOK=off
git merge --no-commit "$SOURCE" || echo "Ignoring conflict..."
[[ -f VERSION ]] && git rm -f VERSION
git commit -m "all: merge $SOURCE into $TARGET"

if ! git log --format=%B -n 1 | grep "\[$TARGET\] "; then
    echo "The commit does not seem to be targeting the BoringCrypto branch."
    exit 1
fi

git codereview mail -r katie@golang.org,filippo@golang.org -trybot HEAD
cd - && git worktree remove "$WORKTREE"
