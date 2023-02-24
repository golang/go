# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Run this using build.sh.

ARG ubuntu=ubuntu
FROM $ubuntu:focal

RUN mkdir /boring
WORKDIR /boring

ENV LANG=C
ENV LANGUAGE=

# Following NIST submission draft dated July 3, 2021.
# This corresponds to boringssl.googlesource.com/boringssl tag fips-20210429.
ENV ClangV=12
RUN apt-get update && \
        apt-get install --no-install-recommends -y cmake xz-utils wget unzip ca-certificates clang-$ClangV python

# Download, validate, unpack, build, and install Ninja.
ENV NinjaV=1.10.2
ENV NinjaH=ce35865411f0490368a8fc383f29071de6690cbadc27704734978221f25e2bed
RUN \
	wget https://github.com/ninja-build/ninja/archive/refs/tags/v$NinjaV.tar.gz && \
	echo "$NinjaH v$NinjaV.tar.gz" >sha && sha256sum -c sha && \
	tar -xzf v$NinjaV.tar.gz && \
	rm v$NinjaV.tar.gz && \
	cd ninja-$NinjaV && \
	CC=clang-$ClangV CXX=clang++-$ClangV ./configure.py --bootstrap && \
	mv ninja /usr/local/bin/

# Download, validate, unpack, and install Go.
ARG GOARCH
ENV GoV=1.16.5
ENV GoHamd64=b12c23023b68de22f74c0524f10b753e7b08b1504cb7e417eccebdd3fae49061
ENV GoHarm64=d5446b46ef6f36fdffa852f73dfbbe78c1ddf010b99fa4964944b9ae8b4d6799
RUN \
	eval GoH=\${GoH$GOARCH} && \
	wget https://golang.org/dl/go$GoV.linux-$GOARCH.tar.gz && \
	echo "$GoH go$GoV.linux-$GOARCH.tar.gz" >sha && sha256sum -c sha && \
	tar -C /usr/local -xzf go$GoV.linux-$GOARCH.tar.gz && \
	rm go$GoV.linux-$GOARCH.tar.gz && \
	ln -s /usr/local/go/bin/go /usr/local/bin/

# Download, validate, and unpack BoringCrypto.
ENV BoringV=853ca1ea1168dff08011e5d42d94609cc0ca2e27
ENV BoringH=a4d069ccef6f3c7bc0c68de82b91414f05cb817494cd1ab483dcf3368883c7c2
RUN \
	wget https://commondatastorage.googleapis.com/chromium-boringssl-fips/boringssl-$BoringV.tar.xz && \
	echo "$BoringH boringssl-$BoringV.tar.xz" >sha && sha256sum -c sha && \
	tar xJf boringssl-$BoringV.tar.xz

# Build BoringCrypto.
ADD build-boring.sh /boring/build-boring.sh
RUN /boring/build-boring.sh

# Build Go BoringCrypto syso.
# build.sh copies it back out of the Docker image.
ADD goboringcrypto.h /boring/godriver/goboringcrypto.h
ADD build-goboring.sh /boring/build-goboring.sh
RUN /boring/build-goboring.sh
