#!/bin/bash
# Copyright 2017 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
id
date
echo http_proxy=$http_proxy
export LANG=C
unset LANGUAGE
apt-get update
apt-get install --no-install-recommends -y cmake clang-4.0 golang-1.8-go ninja-build xz-utils
