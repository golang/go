# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Base builder image: gobuilders/linux-x86-base

FROM debian:wheezy
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

# For running curl to get the hg starter tarballs (faster than hg clone).
RUN apt-get install -y --no-install-recommends curl ca-certificates

# Optionally used by some net/http tests:
RUN apt-get install -y --no-install-recommends strace 

# For building Go's bootstrap 'dist' program:
RUN apt-get install -y --no-install-recommends gcc libc6-dev

# For 32-bit builds:
# TODO(bradfitz): move these into a 386 image that derives from this one.
RUN apt-get install -y --no-install-recommends libc6-dev-i386 gcc-multilib

# For interacting with the Go source & subrepos:
RUN apt-get install -y --no-install-recommends mercurial git-core

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
