#! /bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: <target branch> [<target branch> ...]"
    echo ""
    echo "example: release.sh dev.boringcrypto.go1.11 dev.boringcrypto.go1.12"
    exit 1
fi

set -x
WORKTREE="$(mktemp -d)"
BRANCH="boring/release-$(date +%Y%m%d%H%M%S)"

git fetch
git worktree add --track -b "$BRANCH" "$WORKTREE" origin/dev.boringcrypto

cd "$WORKTREE/src"
./make.bash

cd ../misc/boring
for branch in "$@"; do
    ./build.release "origin/$branch"
done
./build.docker

git add RELEASES
git commit -m "misc/boring: add new releases to RELEASES file"
git codereview mail -r dmitshur@golang.org,filippo@golang.org

rm *.tar.gz
cd - && git worktree remove "$WORKTREE"
