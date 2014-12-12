# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# gobuilders/linux-x86-nacl for 32- and 64-bit nacl.
#
# We need more modern libc than Debian stable as used in base, so we're
# using Ubuntu LTS here.
#
# TODO(bradfitz): make both be Ubuntu? But we also want Debian, Fedora, 
# etc coverage., so deal with unifying these later, once there's a plan
# or a generator for them and the other builders are turned down.

FROM ubuntu:trusty
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/build-go-builder.sh /scripts/
RUN GO_REV=go1.4 BUILDER_REV=6735829f /scripts/build-go-builder.sh && test -f /usr/local/bin/builder

ADD build-command.pl /usr/local/bin/

ENV PATH /usr/local/bin:$GOROOT/bin:$PATH
