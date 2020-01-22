# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# govim requires a more recent version of vim than is available in most
# distros, so we build from their base image.
FROM govim/govim:go1.13.5_vim_v8.1.2414_v1

# We use a pinned version of govim so that this build is repeatable, and so
# that we're not sensitive to test breakages in govim.
# TODO(findleyr): Once a version of govim has been tagged containing
# https://github.com/govim/govim/pull/629, switch this to @latest.
ENV GOPROXY=https://proxy.golang.org GOPATH=/go VIM_FLAVOR=vim \
    GOVIM_VERSION=v0.0.27-0.20200122085353-96e61d22452d
WORKDIR /src

# Get govim, as well as all dependencies for the ./cmd/govim tests. In order to
# use the go command's module version resolution, we use `go mod download` for
# this rather than git. Another approach would be to use a temporary module and
# `go get -t -u github.com/govim/govim`, but that may cause test dependencies
# to be upgraded.
RUN GOVIM_DIR=$(go mod download -json github.com/govim/govim@$GOVIM_VERSION | jq -r .Dir) && \
    cp -r $GOVIM_DIR /src/govim && cd /src/govim && \
    go list -test -deps ./cmd/govim
