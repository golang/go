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

# Following the Security Policy for FIPS 140 certificate #4735.
# https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp4735.pdf
# This corresponds to boringssl.googlesource.com/boringssl tag fips-20220613.
RUN apt-get update && \
        apt-get install --no-install-recommends -y xz-utils wget unzip ca-certificates python lsb-release software-properties-common gnupg make libssl-dev faketime

# Install Clang.
ENV ClangV=14
RUN \
	wget https://apt.llvm.org/llvm.sh && \
	chmod +x llvm.sh && \
	./llvm.sh $ClangV

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

# Download, validate, unpack, build, and install Cmake.
ENV CmakeV=3.22.1
ENV CmakeH=0e998229549d7b3f368703d20e248e7ee1f853910d42704aa87918c213ea82c0
RUN \
	wget https://github.com/Kitware/CMake/releases/download/v$CmakeV/cmake-$CmakeV.tar.gz && \
	echo "$CmakeH cmake-$CmakeV.tar.gz" >sha && sha256sum -c sha && \
	tar -xzf cmake-$CmakeV.tar.gz && \
	rm cmake-$CmakeV.tar.gz && \
	cd cmake-$CmakeV && \
	CC=clang-$ClangV CXX=clang++-$ClangV ./bootstrap && \
	make && make install

# Download, validate, unpack, and install Go.
ARG GOARCH
ENV GoV=1.18.1
ENV GoHamd64=b3b815f47ababac13810fc6021eb73d65478e0b2db4b09d348eefad9581a2334
ENV GoHarm64=56a91851c97fb4697077abbca38860f735c32b38993ff79b088dac46e4735633
RUN \
	eval GoH=\${GoH$GOARCH} && \
	wget https://golang.org/dl/go$GoV.linux-$GOARCH.tar.gz && \
	echo "$GoH go$GoV.linux-$GOARCH.tar.gz" >sha && sha256sum -c sha && \
	tar -C /usr/local -xzf go$GoV.linux-$GOARCH.tar.gz && \
	rm go$GoV.linux-$GOARCH.tar.gz && \
	ln -s /usr/local/go/bin/go /usr/local/bin/

# Download, validate, and unpack BoringCrypto.
ENV BoringV=0c6f40132b828e92ba365c6b7680e32820c63fa7
ENV BoringH=62f733289f2d677c2723f556aa58034c438f3a7bbca6c12b156538a88e38da8a
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
