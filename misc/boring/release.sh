#! /bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: <target branch> [<target branch> ...]"
    echo ""
    echo "example: release.sh dev.boringcrypto.go1.11 dev.boringcrypto.go1.12"
    exit 1
fi

# Check that the Docker daemon is available.
docker ps > /dev/null

WORKTREE="$(mktemp -d)"
BRANCH="boring/release-$(date +%Y%m%d%H%M%S)"

git fetch
git worktree add --track -b "$BRANCH" "$WORKTREE" origin/dev.boringcrypto

cd "$WORKTREE/src"
./make.bash

cd ../misc/boring
for branch in "$@"; do
    ./build.release "origin/$branch"
    ./build.docker
done

git add RELEASES
git commit -m "misc/boring: add new releases to RELEASES file"
git codereview mail -r katie@golang.org,roland@golang.org,filippo@golang.org -trust

rm *.tar.gz
cd - && git worktree remove "$WORKTREE"
