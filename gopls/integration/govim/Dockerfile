# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# govim requires a more recent version of vim than is available in most
# distros, so we build from their base image.
FROM govim/govim:latest-vim

# We use a pinned version of govim so that this build is repeatable, and so
# that we're not sensitive to test breakages in govim.
# TODO(findleyr): Once a version of govim has been tagged containing
# https://github.com/govim/govim/pull/629, switch this to @latest.
ENV GOPROXY=https://proxy.golang.org GOPATH=/go VIM_FLAVOR=vim
WORKDIR /src

# Clone govim. In order to use the go command for resolving latest, we download
# a redundant copy of govim to the build cache using `go mod download`.
RUN GOVIM_VERSION=$(go mod download -json github.com/govim/govim@latest | jq -r .Version) && \
    git clone https://github.com/govim/govim /src/govim && cd /src/govim && \
    git checkout $GOVIM_VERSION
