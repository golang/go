# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Commit watcher for Go repos.

# Wheezy has Mercurial 2.2.2 and we need Mercurial >= 2.8, sid has 3.1.
FROM debian:sid
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/build-commit-watcher.sh /scripts/
RUN GO_REV=go1.4 WATCHER_REV=6735829fe0 /scripts/build-commit-watcher.sh && test -f /usr/local/bin/watcher
