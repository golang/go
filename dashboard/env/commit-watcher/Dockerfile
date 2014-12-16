# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Commit watcher for Go repos.

FROM debian:wheezy
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/build-commit-watcher.sh /scripts/
RUN GO_REV=go1.4 WATCHER_REV=a54d0066172 /scripts/build-commit-watcher.sh && test -f /usr/local/bin/watcher
