# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# gobuilders/linux-x86-gccgo for 32- and 64-bit gccgo.

FROM debian:wheezy
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/install-gold.sh /scripts/
RUN /scripts/install-gold.sh

ADD /scripts/install-gccgo-builder.sh /scripts/
RUN /scripts/install-gccgo-builder.sh && test -f /usr/local/bin/builder