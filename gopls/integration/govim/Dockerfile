# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# govim requires a more recent version of vim than is available in most
# distros, so we build from their base image.
FROM govim/govim:latest-vim
ARG GOVIM_REF

ENV GOPROXY=https://proxy.golang.org GOPATH=/go VIM_FLAVOR=vim
WORKDIR /src

# Clone govim. In order to use the go command for resolving latest, we download
# a redundant copy of govim to the build cache using `go mod download`.
RUN git clone https://github.com/govim/govim /src/govim && cd /src/govim && \
    git checkout $GOVIM_REF
