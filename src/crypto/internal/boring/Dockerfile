# Copyright 2020 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This Docker image builds goboringcrypto_linux_amd64.syso according to the
# Security Policy. To use it, build the image, run it, and then extract
# /boring/godriver/goboringcrypto_linux_amd64.syso.
#
#   $ podman build -t goboring:140sp3678 .
#   $ podman run -it --name goboring-140sp3678 goboring:140sp3678
#   $ podman cp goboring-140sp3678:/boring/godriver/goboringcrypto_linux_amd64.syso syso
#   $ sha256sum syso/goboringcrypto_linux_amd64.syso # compare to docker output
#
# The podman commands may need to run under sudo to work around a subuid/subgid bug.

FROM ubuntu:focal

RUN mkdir /boring
WORKDIR /boring

# Following 140sp3678.pdf [0] page 19, install clang 7.0.1, Go 1.12.7, and
# Ninja 1.9.0, then download and verify BoringSSL.
#
# [0]: https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3678.pdf

RUN apt-get update && \
        apt-get install --no-install-recommends -y cmake xz-utils wget unzip ca-certificates clang-7
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.9.0/ninja-linux.zip && \
        unzip ninja-linux.zip && \
        rm ninja-linux.zip && \
        mv ninja /usr/local/bin/
RUN wget https://golang.org/dl/go1.12.7.linux-amd64.tar.gz && \
        tar -C /usr/local -xzf go1.12.7.linux-amd64.tar.gz && \
        rm go1.12.7.linux-amd64.tar.gz && \
        ln -s /usr/local/go/bin/go /usr/local/bin/

RUN wget https://commondatastorage.googleapis.com/chromium-boringssl-fips/boringssl-ae223d6138807a13006342edfeef32e813246b39.tar.xz
RUN [ "$(sha256sum boringssl-ae223d6138807a13006342edfeef32e813246b39.tar.xz | awk '{print $1}')" = \
        3b5fdf23274d4179c2077b5e8fa625d9debd7a390aac1d165b7e47234f648bb8 ]

ADD goboringcrypto.h /boring/godriver/goboringcrypto.h
ADD build.sh /boring/build.sh

ENTRYPOINT ["/boring/build.sh"]
