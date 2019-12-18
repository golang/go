# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# govim requires a more recent version of vim than is available in most
# distros, so we build from their base image.
FROM govim/govim:go1.13.5_vim_v8.1.2414_v1

# Get a copy of govim in order to run its integration tests. We use a pinned
# version so that this build is repeatable, and so that we're not sensitive to
# test breakages in govim.
# TODO(findleyr): Once a version of govim has been tagged containing
# https://github.com/govim/govim/pull/629, switch this to @latest.
ENV GOPROXY=https://proxy.golang.org GOPATH=/go VIM_FLAVOR=vim
WORKDIR /src/mod
RUN go mod init mod && go get -t github.com/govim/govim@v0.0.27-0.20191220164001-63ce556bb69e
