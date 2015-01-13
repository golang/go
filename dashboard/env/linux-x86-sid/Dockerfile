# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM debian:sid
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

ADD /scripts/install-apt-deps.sh /scripts/
RUN /scripts/install-apt-deps.sh

ADD /scripts/build-go-builder.sh /scripts/
RUN GO_REV=go1.4 BUILDER_REV=d79e0375a /scripts/build-go-builder.sh && test -f /usr/local/bin/builder

RUN mkdir -p /go1.4-386   && (curl --silent https://storage.googleapis.com/golang/go1.4.linux-386.tar.gz   | tar -C /go1.4-386   -zxv)
RUN mkdir -p /go1.4-amd64 && (curl --silent https://storage.googleapis.com/golang/go1.4.linux-amd64.tar.gz | tar -C /go1.4-amd64 -zxv)
