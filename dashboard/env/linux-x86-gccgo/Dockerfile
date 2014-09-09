# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# gobuilders/linux-x86-gccgo for 64-bit gccgo.

FROM gobuilders/linux-x86-base
MAINTAINER golang-dev <golang-dev@googlegroups.com>

ENV DEBIAN_FRONTEND noninteractive

# For using numeric libraries within GCC.
RUN apt-get install -y --no-install-recommends libgmp10-dev libmpc-dev libmpfr-dev

# For building binutils and gcc from source.
RUN apt-get install -y --no-install-recommends make g++ flex bison

# gccgo uses the Gold linker from binutils.
ENV BINUTILS_VERSION binutils-2.24
RUN curl -s http://ftp.gnu.org/gnu/binutils/$BINUTILS_VERSION.tar.gz | tar x --no-same-owner -zv
RUN mkdir binutils-objdir
RUN cd binutils-objdir && ../$BINUTILS_VERSION/configure --enable-gold --enable-plugins --prefix=/opt/gold
RUN cd binutils-objdir && make -sj && make install -sj

# For running the extended gccgo testsuite
RUN apt-get install -y --no-install-recommends dejagnu