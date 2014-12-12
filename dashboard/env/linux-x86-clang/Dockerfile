# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# gobuilders/linux-x86-clang for building with clang instead of gcc.

FROM debian:wheezy
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /sources/clang-deps.list /etc/apt/sources.list.d/

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/build-go-builder.sh /scripts/
RUN GO_REV=go1.4 BUILDER_REV=6735829f /scripts/build-go-builder.sh && test -f /usr/local/bin/builder

ENV CC /usr/bin/clang
