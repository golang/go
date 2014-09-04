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

RUN echo cachebuster4
RUN apt-get update && apt-get upgrade -y

# For running curl to get the hg starter tarballs (faster than hg clone).
RUN apt-get install -y --no-install-recommends curl ca-certificates

# For building Go's bootstrap 'dist' program:
RUN apt-get install -y --no-install-recommends gcc libc6-dev

# For interacting with the Go source & subrepos:
RUN apt-get install -y --no-install-recommends mercurial git-core

# For 32-bit nacl:
RUN apt-get install -y --no-install-recommends libc6-i386 libc6-dev-i386 lib32stdc++6 gcc-multilib

ENV GOPATH /gopath
ENV GOROOT /goroot

RUN mkdir -p $GOROOT
RUN cd $GOROOT && curl -s https://storage.googleapis.com/gobuilder/go-snap.tar.gz | tar x --no-same-owner -zv

ENV GO_TOOLS $GOPATH/src/code.google.com/p/go.tools
RUN mkdir -p $GO_TOOLS
RUN cd $GO_TOOLS $ && curl -s https://storage.googleapis.com/gobuilder/go.tools-snap.tar.gz | tar x --no-same-owner -zv

RUN cd $GOROOT && hg pull && hg update -C b8ff0ec2a724
RUN cd $GOROOT/src && ./make.bash
RUN mkdir -p /usr/local/bin

RUN cd $GO_TOOLS && hg pull && hg update -C f9e37a924785
ENV GOBIN /usr/local/bin
RUN /goroot/bin/go install code.google.com/p/go.tools/dashboard/builder

RUN cd /usr/local/bin && curl -s -O https://storage.googleapis.com/gobuilder/sel_ldr_x86_32 && chmod +x sel_ldr_x86_32
RUN cd /usr/local/bin && curl -s -O https://storage.googleapis.com/gobuilder/sel_ldr_x86_64 && chmod +x sel_ldr_x86_64

RUN ln -s $GOROOT/misc/nacl/go_nacl_386_exec /usr/local/bin/
RUN ln -s $GOROOT/misc/nacl/go_nacl_amd64p32_exec /usr/local/bin/
ADD build-command.pl /usr/local/bin/

ENV PATH /usr/local/bin:$GOROOT/bin:$PATH
