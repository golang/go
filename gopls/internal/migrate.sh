#!/bin/bash
#
# Copyright 2022 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Migrates the internal/lsp directory to gopls/internal/lsp. Run this script
# from the root of x/tools to migrate in-progress CLs.
#
# See golang/go#54509 for more details. This script may be deleted once a
# reasonable amount of time has passed such that all active in-progress CLs
# have been rebased.

set -eu

# A portable -i flag. Darwin requires two parameters.
# See https://stackoverflow.com/questions/5694228/sed-in-place-flag-that-works-both-on-mac-bsd-and-linux
# for more details.
sedi=(-i)
case "$(uname)" in
  Darwin*) sedi=(-i "")
esac

# mvpath moves the directory at the relative path $1 to the relative path $2,
# moving files and rewriting import paths.
#
# It uses heuristics to identify import path literals, and therefore may be
# imprecise.
function mvpath() {
  # If the source also doesn't exist, it may have already been moved.
  # Skip so that this script is idempotent.
  if [[ ! -d $1 ]]; then
    echo "WARNING: skipping nonexistent source directory $1"
    return 0
  fi

  # git can sometimes leave behind empty directories, which can change the
  # behavior of the mv command below.
  if [[ -d $2 ]] || [[ -f $2 ]]; then
    echo "ERROR: destination $2 already exists"
    exit 1
  fi

  mv $1 $2

  local old="golang.org/x/tools/$1"
  local new="golang.org/x/tools/$2"

  # Replace instances of the old import path with the new. This is imprecise,
  # but we are a bit careful to avoid replacing golang.org/x/tools/foox with
  # golang.org/x/tools/barx when moving foo->bar: the occurrence of the import
  # path must be followed by whitespace, /, or a closing " or `.
  local replace="s:${old}\([[:space:]/\"\`]\):${new}\1:g"
  find . -type f \( \
    -name ".git" -prune -o \
    -name "*.go" -o \
    -name "*.in" -o \
    -name "*.golden" -o \
    -name "*.hlp" -o \
    -name "*.md" \) \
    -exec sed "${sedi[@]}" -e $replace {} \;
}

mvpath internal/lsp/diff internal/diff
mvpath internal/lsp/fuzzy internal/fuzzy
mvpath internal/lsp/debug/tag internal/event/tag
mvpath internal/lsp/bug internal/bug
mvpath internal/lsp gopls/internal/lsp
